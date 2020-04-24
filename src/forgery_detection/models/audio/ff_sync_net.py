import logging

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.video import r2plus1d_18

from forgery_detection.lightning.logging.const import NAN_TENSOR
from forgery_detection.lightning.logging.const import VAL_ACC
from forgery_detection.models.audio.similarity_stuff import PretrainedSyncNet
from forgery_detection.models.audio.utils import ContrastiveLoss
from forgery_detection.models.mixins import PretrainedNet
from forgery_detection.models.utils import SequenceClassificationModel

logger = logging.getLogger(__file__)


class FFSyncNet(SequenceClassificationModel):
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

        self.video_mlp = nn.Sequential(
            nn.Linear(64, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 1024)
        )

        self.sync_net = PretrainedSyncNet()
        self._set_requires_grad_for_module(self.sync_net, requires_grad=False)

        self.audio_extractor = self.sync_net.audio_extractor

        self.c_loss = ContrastiveLoss(20)

        self.log_class_loss = False

    def forward(self, x):
        video, audio = x  # bs x 8 x 3 x 112 x 112 , bs x 8 x 16 x 29
        # syncnet only uses 5 frames
        audio = audio[:, 2:-1]
        audio = (audio.reshape((audio.shape[0], -1, 13)).unsqueeze(1)).transpose(-2, -1)

        video = video.permute(0, 2, 1, 3, 4)

        return self.video_mlp(self.r2plus1(video)), self.audio_extractor(audio)

    def loss(self, logits, labels):
        video_logits, audio_logits = logits
        return self.c_loss(video_logits, audio_logits, labels)

    def training_step(self, batch, batch_nb, system):
        x, (label, audio_shifted) = batch
        audio_shifted: torch.Tensor
        audio_target = label // 4

        # assert torch.all(torch.ones_like(audio_shifted).eq(audio_shifted))

        pred = self.forward(x)
        loss = self.loss(pred, audio_target)
        lightning_log = {"loss": loss}

        tensorboard_log = {"loss": {"train": loss}}
        if self.log_class_loss or True:
            class_loss = self.loss_per_class(pred[0], pred[1], label)
            tensorboard_log["class_loss_train"] = {
                str(idx): val for idx, val in enumerate(class_loss)
            }
            tensorboard_log["class_loss_diff_train"] = {
                str(idx): val - class_loss[4] for idx, val in enumerate(class_loss[:4])
            }
            tensorboard_log["vid_std"] = {"train": torch.std(pred[0])}
            tensorboard_log["aud_std"] = {"train": torch.std(pred[1])}
            self.log_class_loss = False

        return tensorboard_log, lightning_log

    def aggregate_outputs(self, outputs, system):
        if len(system.val_dataloader()) > 1:
            outputs = outputs[0]

        with torch.no_grad():
            video_logits = torch.cat([x["pred"][0] for x in outputs], 0)
            audio_logtis = torch.cat([x["pred"][1] for x in outputs], 0)

            label_list = [x["target"][0] for x in outputs]
            label = torch.cat(label_list, dim=0)

            audio_target = label // 4

            loss_mean = self.loss((video_logits, audio_logtis), audio_target)

            class_loss = self.loss_per_class(video_logits, audio_logtis, label)

            tensorboard_log = {
                "loss": loss_mean,
                "class_loss_val": {str(idx): val for idx, val in enumerate(class_loss)},
                "class_loss_diff_val": {
                    str(idx): val - class_loss[4]
                    for idx, val in enumerate(class_loss[:4])
                },
                "vid_std": torch.std(video_logits),
                "aud_std": torch.std(audio_logtis),
            }
        # if system.global_step > 0:
        self.log_class_loss = True

        return tensorboard_log, {}

    def loss_per_class(self, video_logits, audio_logits, targets):
        class_loss = torch.zeros((5,))
        class_counts = torch.zeros((5,))
        distances = (video_logits - audio_logits).pow(2).sum(1)
        for target, logit in zip(targets, distances):
            class_loss[target] += logit
            class_counts[target] += 1
        return class_loss / class_counts


class PretrainedFFSyncNet(
    PretrainedNet(
        "/home/sebastian/log/runs/TRAIN/ff_sync_net/version_0/_ckpt_epoch_4.ckpt"
    ),
    FFSyncNet,
):
    pass


class FFSyncNetGeneralize(FFSyncNet):
    def loss_without_class(self, logits, labels, classes):
        labels_mask = classes != 0
        if labels_mask.sum() == 0:
            return NAN_TENSOR
        logits = logits[0][labels_mask], logits[1][labels_mask]
        labels = labels[labels_mask]
        video_logits, audio_logits = logits
        return self.c_loss(video_logits, audio_logits, labels)

    def training_step(self, batch, batch_nb, system):
        x, (label, audio_shifted) = batch
        audio_shifted: torch.Tensor
        audio_target = label // 4

        # assert torch.all(torch.ones_like(audio_shifted).eq(audio_shifted))

        pred = self.forward(x)
        loss_without_class = self.loss_without_class(pred, audio_target, label)
        loss = self.loss(pred, audio_target)
        lightning_log = {"loss": loss_without_class}

        tensorboard_log = {
            "loss": {"train": loss, "train_without_0": loss_without_class}
        }
        if self.log_class_loss or True:
            class_loss = self.loss_per_class(pred[0], pred[1], label)
            tensorboard_log["class_loss_train"] = {
                str(idx): val for idx, val in enumerate(class_loss)
            }
            tensorboard_log["class_loss_diff_train"] = {
                str(idx): val - class_loss[4] for idx, val in enumerate(class_loss[:4])
            }
            tensorboard_log["vid_std"] = {"train": torch.std(pred[0])}
            tensorboard_log["aud_std"] = {"train": torch.std(pred[1])}
            self.log_class_loss = False

        return tensorboard_log, lightning_log


class PretrainedFFSyncNetGeneralize(
    PretrainedNet(
        "/home/sebastian/log/runs/TRAIN/ff_sync_net_generalize/version_1/_ckpt_epoch_15.ckpt"
    ),
    FFSyncNet,
):
    pass


class EmbeddingClassifier(SequenceClassificationModel):
    def forward(self, x):
        embeddings = self.ff_sync_net(x)
        cat = torch.cat(embeddings, dim=1)
        out = self.out(cat)
        return out, embeddings

    def training_step(self, batch, batch_nb, system):
        x, (target, audio_shift) = batch

        label = target // 4

        # if the model uses dropout we want to calculate the metrics on predictions done
        # in eval mode before training no the samples
        if self.contains_dropout:
            with torch.no_grad():
                self.eval()
                pred_ = self.forward(x)
                self.train()

        pred, embeddings = self.forward(x)
        loss = self.loss(pred, label)
        lightning_log = {"loss": loss}

        with torch.no_grad():
            train_acc = self.calculate_accuracy(pred, label)
            tensorboard_log = {"loss": {"train": loss}, "acc": {"train": train_acc}}

            class_loss = self.ff_sync_net.loss_per_class(
                embeddings[0], embeddings[1], target
            )
            tensorboard_log["class_loss_train"] = {
                str(idx): val for idx, val in enumerate(class_loss)
            }
            tensorboard_log["class_loss_diff_train"] = {
                str(idx): val - class_loss[4] for idx, val in enumerate(class_loss[:4])
            }

            if self.contains_dropout:
                pred = pred_
                loss_eval = self.loss(pred, label)
                acc_eval = self.calculate_accuracy(pred, label)
                tensorboard_log["loss"]["train_eval"] = loss_eval
                tensorboard_log["acc"]["train_eval"] = acc_eval

        return tensorboard_log, lightning_log

    def aggregate_outputs(self, outputs, system):
        # if there are more then one dataloader we ignore the additional data
        if len(system.val_dataloader()) > 1:
            outputs = outputs[0]

        with torch.no_grad():
            pred = torch.cat([x["pred"][0] for x in outputs], 0)
            label = torch.cat([x["target"][0] for x in outputs], 0)
            target = label // 4

            video_logits = torch.cat([x["pred"][1][0] for x in outputs], 0)
            audio_logtis = torch.cat([x["pred"][1][1] for x in outputs], 0)
            class_loss = self.ff_sync_net.loss_per_class(
                video_logits, audio_logtis, label
            )

            loss_mean = self.loss(pred, target)
            pred = pred.cpu()
            target = target.cpu()
            pred = F.softmax(pred, dim=1)
            acc_mean = self.calculate_accuracy(pred, target)

            # confusion matrix
            class_accuracies = system.log_confusion_matrix(target, pred)

            tensorboard_log = {
                "loss": loss_mean,
                "acc": acc_mean,
                "class_acc": class_accuracies,
                "class_loss_val": {str(idx): val for idx, val in enumerate(class_loss)},
                "class_loss_diff_val": {
                    str(idx): val - class_loss[4]
                    for idx, val in enumerate(class_loss[:4])
                },
            }

            lightning_log = {VAL_ACC: acc_mean}

        return tensorboard_log, lightning_log


class FFSyncNetClassifier(EmbeddingClassifier):
    def __init__(self, num_classes=5, sequence_length=8):
        super().__init__(
            num_classes=num_classes,
            sequence_length=sequence_length,
            contains_dropout=False,
        )

        self.ff_sync_net = PretrainedFFSyncNet()
        self._set_requires_grad_for_module(self.ff_sync_net, requires_grad=False)

        self.out = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(1024 * 2, 50),
            # nn.BatchNorm1d(50),
            # nn.Dropout(p=0.5),
            nn.LeakyReLU(0.02),
            nn.Linear(50, 2),
        )


class FFSyncNetClassifierGeneralze(EmbeddingClassifier):
    def __init__(self, num_classes=5, sequence_length=8):
        super().__init__(
            num_classes=num_classes,
            sequence_length=sequence_length,
            contains_dropout=False,
        )

        self.ff_sync_net = PretrainedFFSyncNetGeneralize()
        self._set_requires_grad_for_module(self.ff_sync_net, requires_grad=False)

        self.out = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(1024 * 2, 50),
            # nn.BatchNorm1d(50),
            # nn.Dropout(p=0.5),
            nn.LeakyReLU(0.02),
            nn.Linear(50, 2),
        )
