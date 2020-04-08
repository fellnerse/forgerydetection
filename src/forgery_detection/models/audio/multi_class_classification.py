import logging

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.resnet import _resnet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import resnet18
from torchvision.models.video import r2plus1d_18

from forgery_detection.models.audio.utils import ContrastiveLoss
from forgery_detection.models.mixins import PretrainedNet
from forgery_detection.models.mixins import SupervisedNet
from forgery_detection.models.utils import SequenceClassificationModel

logger = logging.getLogger(__file__)


class AudioNet(SequenceClassificationModel):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__(
            num_classes=num_classes, sequence_length=8, contains_dropout=False
        )
        self.r2plus1 = r2plus1d_18(pretrained=pretrained)
        self.r2plus1.layer3 = nn.Identity()
        self.r2plus1.layer4 = nn.Identity()
        self.r2plus1.fc = nn.Identity()

        self.resnet = resnet18(pretrained=pretrained, num_classes=1000)
        self.resnet.layer3 = nn.Identity()
        self.resnet.layer4 = nn.Identity()
        self.resnet.fc = nn.Identity()

        self.relu = nn.ReLU()
        self.out = nn.Sequential(
            nn.Linear(256, 50), nn.ReLU(), nn.Linear(50, self.num_classes)
        )

    def forward(self, x):
        # def forward(self, video, audio):
        video, audio = x  # bs x 8 x 3 x 112 x 112 , bs x 8 x 29

        video = video.transpose(1, 2)
        video = self.r2plus1(video)

        audio = self.resnet(audio.unsqueeze(1).expand(-1, 3, -1, -1))

        flat = torch.cat((video, audio), dim=1)
        out = self.out(self.relu(flat))
        return out


class AudioNetFrozen(AudioNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_requires_grad_for_module(self.r2plus1, requires_grad=False)
        self._set_requires_grad_for_module(self.resnet, requires_grad=False)


class PretrainedAudioNet(
    PretrainedNet("/data/hdd/model_checkpoints/audionet/13_epochs/model.ckpt"), AudioNet
):
    pass


class AudioNetLayer2Unfrozen(PretrainedAudioNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_requires_grad_for_module(self.r2plus1, requires_grad=False)
        self._set_requires_grad_for_module(self.r2plus1.layer2, requires_grad=True)
        self._set_requires_grad_for_module(self.resnet, requires_grad=False)
        self._set_requires_grad_for_module(self.resnet.layer2, requires_grad=True)


class AudioOnly(SequenceClassificationModel):
    def __init__(self, pretrained=True):
        super().__init__(num_classes=5, sequence_length=8, contains_dropout=False)

        self.resnet = _resnet(
            "resnet18",
            BasicBlock,
            [1, 1, 1, 1],
            pretrained=False,
            progress=True,
            num_classes=1000,
        )
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resnet.layer4 = nn.Identity()
        self.resnet.fc = nn.Linear(256, 5)

    def forward(self, x):
        _, audio = x
        del _
        return self.resnet(audio.unsqueeze(1))


class Audio2ExpressionNet(nn.Module):
    def __init__(self, T, input_dim=29):
        super(Audio2ExpressionNet, self).__init__()

        self.T = T

        self.convs = nn.Sequential(
            nn.Conv1d(input_dim, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.02),
            nn.Conv1d(32, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.02),
            nn.Conv1d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.02),
            nn.Conv1d(64, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.02),
            # nn.Conv1d(64, 64, 3, stride=2, padding=1),
            # nn.LeakyReLU(0.02),
            # nn.Conv1d(64, 64, 3, stride=2, padding=1),
            # nn.LeakyReLU(0.02), needed for mfcc_features_stacked.npy
        )

        self.fcs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02),
            nn.Linear(64, 64),
        )

        self.filter = nn.Sequential(
            nn.Conv1d(T, T, 3, stride=2, padding=1),  # [b, 8, 32]
            nn.LeakyReLU(0.02),
            nn.Conv1d(T, T, 5, stride=4, padding=2),  # [b, 8, 8]
            nn.LeakyReLU(0.02),
            nn.Conv1d(T, T, 5, stride=4, padding=2),  # [b, 8, 2]
            nn.LeakyReLU(0.02),
            nn.Conv1d(T, T, 2, stride=1, padding=0),  # [b, 8, 1]
            nn.LeakyReLU(0.02),
            nn.Flatten(),
            nn.Linear(T, T),
            nn.Sigmoid(),
        )

    def forward(self, audio):
        # input shape: [b, T, 16, 29]
        x = audio.permute(1, 0, 3, 2)  # shape: [T, b, 29, 16]

        # Per frame expression estimation
        z = []
        for window in x:
            z.append(self.fcs(self.convs(window)))
        z = torch.stack(z, dim=1)  # shape: [b, 8, 2048]

        # Filtering
        if z.shape[1] > 1:
            f = self.filter(z).unsqueeze(2)  # shape: [b, 8, 1]
            y = torch.mul(z, f)  # shape: [b, 8, 64]
            y = y.sum(1)  # shape: [b, 64]
        else:
            y = z[:, 0]

        return y  # .view(-1, self.T, 64)  # .view(-1, 4, 512)  # shape: [b, 4, 512]


class FrameNet(SequenceClassificationModel):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__(
            num_classes=num_classes, sequence_length=8, contains_dropout=False
        )

        self.resnet = resnet18(pretrained=pretrained, num_classes=1000)  # 64
        self.resnet.layer2 = nn.Identity()
        self.resnet.layer3 = nn.Identity()
        self.resnet.layer4 = nn.Identity()
        self.resnet.fc = nn.Identity()

        self.audio_extractor = Audio2ExpressionNet(1)  # want output for each frame

        self.out = nn.Sequential(
            nn.Linear(
                self.sequence_length * 2 * 64, 50
            ),  # maybe convolution over time would make sense here
            nn.LeakyReLU(0.02),
            nn.Linear(50, self.num_classes),
        )

    def forward(self, x):
        video, audio = x  # bs x 8 x 3 x 112 x 112 , bs x 8 x 16 x 29
        bs = video.shape[0]
        video, audio = (
            video.view(-1, *video.shape[2:]),
            audio.view(-1, *audio.shape[2:]).unsqueeze(1),
        )

        video_out = self.resnet(video).view(bs, -1)  # b x 8 * 64
        audio_out = self.audio_extractor(audio).view(bs, -1)  # b x 8 * 64

        combined_features = torch.cat((video_out, audio_out), dim=1)

        out = self.out(combined_features)
        return out


class SimilarityNet(SequenceClassificationModel):
    def __init__(self, num_classes=5, sequence_length=8, pretrained=True):
        super().__init__(
            num_classes=num_classes,
            sequence_length=sequence_length,
            contains_dropout=False,
        )
        self.r2plus1 = r2plus1d_18(pretrained=pretrained)
        self.r2plus1.layer2 = nn.Identity()
        self.r2plus1.layer3 = nn.Identity()
        self.r2plus1.layer4 = nn.Identity()
        self.r2plus1.fc = nn.Identity()

        self.audio_extractor = resnet18(pretrained=pretrained, num_classes=1000)
        self.audio_extractor.layer2 = nn.Identity()
        self.audio_extractor.layer3 = nn.Identity()
        self.audio_extractor.layer4 = nn.Identity()
        self.audio_extractor.fc = nn.Identity()

        self.c_loss = ContrastiveLoss(1)

        self.log_class_loss = False

    def _shuffle_audio(self, audio: torch.Tensor):
        with torch.no_grad():
            middle = audio.shape[0] // 2
            audio[:middle] = audio[torch.randperm(middle)]
            return audio

    def _get_targets(self, logits: torch.Tensor):
        nb_items_neg = logits.shape[0] // 2
        return torch.cat(
            (
                torch.zeros((nb_items_neg,)).long(),
                torch.ones((logits.shape[0] - nb_items_neg,)).long(),
            )
        ).to(logits.device)

    def forward(self, x):
        video, audio = x  # bs x 8 x 3 x 112 x 112 , bs x 8 x 16 x 29
        # def forward(self, video, audio):
        audio = self._shuffle_audio(audio)
        audio = (
            audio.reshape((audio.shape[0], -1, 13)).unsqueeze(1).expand(-1, 3, -1, -1)
        )

        video = video.permute(0, 2, 1, 3, 4)
        return self.r2plus1(video), self.audio_extractor(audio)

    def loss(self, logits, labels):
        video_logits, audio_logits = logits
        return self.c_loss(video_logits, audio_logits, labels)

    def training_step(self, batch, batch_nb, system):
        x, target = batch

        pred = self.forward(x)
        target = self._get_targets(target)
        loss = self.loss(pred, target)
        lightning_log = {"loss": loss}

        tensorboard_log = {"loss": {"train": loss}}
        if self.log_class_loss or True:
            class_loss = self.loss_per_class(pred[0], pred[1], target)
            tensorboard_log["class_acc_train"] = {
                str(idx): val for idx, val in enumerate(class_loss)
            }
            # self.log_embedding(pred[0], pred[1], target, system)
            self.log_class_loss = False

        return tensorboard_log, lightning_log

    def aggregate_outputs(self, outputs, system):
        # if there are more then one dataloader we ignore the additional data
        if len(system.val_dataloader()) > 1:
            outputs = outputs[0]

        with torch.no_grad():
            video_logits = torch.cat([x["pred"][0] for x in outputs], 0)
            audio_logtis = torch.cat([x["pred"][1] for x in outputs], 0)

            # i will have to change this so because the targets are wrong
            # here several batches are concatenated, and after that the targets are calculated
            # but we have to calculate the targets before, or do all the stuff batchwise
            target_list = [x["target"] for x in outputs]
            calculated_targets = [self._get_targets(x) for x in target_list]
            target = torch.cat(calculated_targets, dim=0)

            loss_mean = self.loss((video_logits, audio_logtis), target)

            class_loss = self.loss_per_class(video_logits, audio_logtis, target)

            # tensorboard only works if number of samples logged does not change ->
            # ignore pre training routine
            # if system.global_step:
            # self.log_embedding(video_logits, audio_logtis, target, system)

            tensorboard_log = {
                "loss": loss_mean,
                "class_acc": {str(idx): val for idx, val in enumerate(class_loss)},
            }
        # if system.global_step > 0:
        self.log_class_loss = True

        return tensorboard_log, {}

    def loss_per_class(self, video_logits, audio_logits, targets):
        class_loss = torch.zeros((2,))
        class_counts = torch.zeros((2,))
        distances = (video_logits - audio_logits).pow(2).sum(1)
        for target, logit in zip(targets, distances):
            class_loss[target] += logit
            class_counts[target] += 1
        return class_loss / class_counts

    @staticmethod
    def get_meta_data(target: torch.Tensor):
        def _get_prefixed_meta_data(_target, prefix):
            return list(map(lambda x: str(x) + prefix, target))

        # return _get_prefixed_meta_data(target, "_v")
        return _get_prefixed_meta_data(target, "_v") + _get_prefixed_meta_data(
            target, "_a"
        )

    @staticmethod
    def get_colour(target, video=True):
        base_value = 255 if video else 128
        colour_dict = [
            torch.ones((3, 32, 32))
            * torch.tensor([base_value * 1, base_value * 0, base_value * 0])
            .unsqueeze(1)
            .unsqueeze(1),
            torch.ones((3, 32, 32))
            * torch.tensor([base_value * 0, base_value * 1, base_value * 1])
            .unsqueeze(1)
            .unsqueeze(1),
            torch.ones((3, 32, 32))
            * torch.tensor([base_value * 1, base_value * 0, base_value * 1])
            .unsqueeze(1)
            .unsqueeze(1),
            torch.ones((3, 32, 32))
            * torch.tensor([base_value * 0, base_value * 1, base_value * 0])
            .unsqueeze(1)
            .unsqueeze(1),
            torch.ones((3, 32, 32))
            * torch.tensor([base_value * 0, base_value * 0, base_value * 1])
            .unsqueeze(1)
            .unsqueeze(1),
        ]
        return torch.stack(list(map(lambda x: colour_dict[x], target)))

    def log_embedding(
        self,
        video_logits: torch.Tensor,
        audio_logits: torch.Tensor,
        target: torch.Tensor,
        system,
    ):
        # metadata = self.get_meta_data(target)
        # label_img = torch.cat(
        #     (self.get_colour(target, video=True), self.get_colour(target, video=False))
        # )
        # system.logger.experiment.add_embedding(
        #     torch.cat((video_logits, audio_logits)),
        #     metadata=metadata,
        #     label_img=label_img,
        #     global_step=0,
        # )
        metadata = self.get_meta_data(target)[: target.shape[0]]
        label_img = self.get_colour(target, video=True)
        system.logger.experiment.add_embedding(
            video_logits,
            metadata=metadata,
            label_img=label_img,
            global_step=system.global_step,
        )


class PretrainedSimilarityNet(
    PretrainedNet(
        "/home/sebastian/log/debug/version_117/checkpoints/_ckpt_epoch_2.ckpt"
    ),
    SimilarityNet,
):
    pass
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.c_loss = ContrastiveLoss(2)


class SyncNet(SimilarityNet):
    def __init__(self, num_layers_in_fc_layers=1024, *args, **kwargs):
        super().__init__(sequence_length=5, *args, **kwargs)

        self.__nFeatures__ = 24
        self.__nChs__ = 32
        self.__midChs__ = 32
        self.num_layers_in_fc_layers = num_layers_in_fc_layers

        self.netcnnaud = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),
            nn.Conv2d(192, 384, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.Conv2d(256, 512, kernel_size=(5, 4), padding=(0, 0)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.netfcaud = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        )

        self.netcnnlip = nn.Sequential(
            nn.Conv3d(3, 96, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=0),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.Conv3d(
                96, 256, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 1, 1)
            ),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.Conv3d(256, 512, kernel_size=(1, 6, 6), padding=0),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )

        self.netfclip = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        )

        self.r2plus1 = nn.Sequential(self.netcnnlip, nn.Flatten(), self.netfclip)
        self.audio_extractor = nn.Sequential(
            self.netcnnaud, nn.Flatten(), self.netfcaud
        )

    def forward(self, x):
        video, audio = x  # bs x 5 x 3 x 112 x 112 , bs x 5 x 4 x 13
        # def forward(self, video, audio):
        audio = self._shuffle_audio(audio)
        audio = (audio.reshape((audio.shape[0], -1, 13)).unsqueeze(1)).transpose(-2, -1)

        video = video.permute(0, 2, 1, 3, 4)
        return self.r2plus1(video), self.audio_extractor(audio)


class SimilarityNetClassification(SupervisedNet(128, 2), PretrainedSimilarityNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.classifier = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(128, 50),
            # nn.Dropout(p=0.5),
            nn.LeakyReLU(0.02),
            nn.Linear(50, 2),
        )

        self._set_requires_grad_for_module(self.r2plus1, requires_grad=False)
        self._set_requires_grad_for_module(self.audio_extractor, requires_grad=False)

    def _shuffle_audio(self, audio: torch.Tensor):
        return audio

    def _get_targets(self, targets: torch.Tensor):
        return (targets == 4).long()

    def forward(self, x):
        video, audio = x  # bs x 8 x 3 x 112 x 112 , bs x 8 x 16 x 29

        audio = (
            audio.reshape((audio.shape[0], -1, 13)).unsqueeze(1).expand(-1, 3, -1, -1)
        )

        video = video.permute(0, 2, 1, 3, 4)
        embedding = self.r2plus1(video), self.audio_extractor(audio)
        concat_embedding = torch.cat(embedding, dim=1)
        return embedding, self.classifier(concat_embedding)

    def loss(self, logits, labels):
        embedding, predictions = logits
        total_loss = F.cross_entropy(predictions, labels)

        return total_loss

    def training_step(self, batch, batch_nb, system):
        x, target = batch

        pred = self.forward(x)
        target = self._get_targets(target)
        loss = self.loss(pred, target)
        lightning_log = {"loss": loss}
        acc = self.calculate_accuracy(pred[1], target)
        tensorboard_log = {"loss": {"train": loss}, "acc": {"train": acc}}

        return tensorboard_log, lightning_log

    def aggregate_outputs(self, outputs, system):
        # if there are more then one dataloader we ignore the additional data
        if len(system.val_dataloader()) > 1:
            outputs = outputs[0]

        with torch.no_grad():
            video_logits = torch.cat([x["pred"][0][0] for x in outputs], 0)
            audio_logtis = torch.cat([x["pred"][0][1] for x in outputs], 0)
            predictions = torch.cat([x["pred"][1] for x in outputs], 0)

            target = torch.cat([x["target"] for x in outputs], 0)
            target = self._get_targets(target)
            loss_mean = self.loss(((video_logits, audio_logtis), predictions), target)

            class_loss = self.loss_per_class(video_logits, audio_logtis, target)

            class_accuracies = system.log_confusion_matrix(target, predictions)
            acc = self.calculate_accuracy(predictions, target)
            # tensorboard only works if number of samples logged does not change ->
            # ignore pre training routine
            # if system.global_step:
            #     self.log_embedding(video_logits, audio_logtis, target, system)

            tensorboard_log = {
                "loss": loss_mean,
                "class_acc": {str(idx): val for idx, val in enumerate(class_loss)},
                "class_acc2": class_accuracies,
                "acc": acc,
            }

        return tensorboard_log, {}

    def loss_per_class(self, video_logits, audio_logits, targets):
        class_loss = torch.zeros((2,))
        class_counts = torch.zeros((2,))
        distances = (video_logits - audio_logits).pow(2).sum(1)
        for target, logit in zip(targets, distances):
            class_loss[target] += logit
            class_counts[target] += 1
        return class_loss / class_counts
