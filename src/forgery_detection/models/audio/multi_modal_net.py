import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from torchvision.models.video import r2plus1d_18

from forgery_detection.lightning.logging.confusion_matrix import confusion_matrix
from forgery_detection.models.audio.similarity_stuff import SimilarityNet
from forgery_detection.models.mixins import BinaryEvaluationMixin
from forgery_detection.models.utils import SequenceClassificationModel


class SimilarityNetBigFiltered(SimilarityNet):
    def __init__(self, num_classes=5, sequence_length=8, pretrained=True):
        super().__init__(num_classes=num_classes, sequence_length=sequence_length)

        # self.r2plus1 = nn.Sequential(
        #     r2plus1d_18(pretrained=pretrained),
        #     nn.Linear(512, 512),
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(0.2),
        # )
        # self.r2plus1[0].fc = nn.Identity()
        self.r2plus1 = r2plus1d_18(pretrained=pretrained)
        self.r2plus1.fc = nn.Identity()

        # self.audio_extractor = nn.Sequential(
        #     resnet18(pretrained=pretrained, num_classes=1000),
        #     nn.Linear(512, 512),
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(0.2),
        # )
        # self.audio_extractor[0].fc = nn.Identity()
        self.audio_extractor = resnet18(pretrained=pretrained, num_classes=1000)
        self.audio_extractor.fc = nn.Identity()

        self.filter = nn.Sequential(  # b x 512 x 9
            nn.Conv1d(
                512, 128, kernel_size=3, stride=1, padding=1, bias=True
            ),  # b x 16 x seq_len
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(
                128, 32, kernel_size=3, stride=1, padding=1, bias=True
            ),  # b x 8 x seq_len
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(
                32, 8, kernel_size=3, stride=1, padding=1, bias=True
            ),  # b x 4 x seq_len
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(
                8, 2, kernel_size=3, stride=1, padding=1, bias=True
            ),  # b x 2 x seq_len
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(
                2, 1, kernel_size=3, stride=1, padding=1, bias=True
            ),  # b x 1 x seq_len
            nn.LeakyReLU(0.02, True),
        )

        self.attention = self.attentionNet = nn.Sequential(
            nn.Linear(9, 9, bias=True), nn.Softmax(dim=1)
        )

    def filter_audio(self, audio: torch.Tensor):
        # audio shape: b x 16 x 4 x 13
        bs = audio.shape[0]

        audio = audio.reshape((-1, 8, 4, 13))

        audio = (
            audio.reshape((audio.shape[0], -1, 13))
            .unsqueeze(1)
            .expand(-1, 3, -1, -1)
            .repeat((1, 1, 1, 4))
        )  # (bs*9) x 3 x 32 x 52

        audio: torch.Tensor = self.audio_extractor(audio)  # (bs*9) x 512
        audio = audio.reshape(bs, 9, 512).transpose(2, 1)  # bs x 512 x 9
        weights = self.filter(audio)  # bs x 1 x 8
        attention = self.attention(weights.squeeze()).unsqueeze(-1)  # bs x 9 x 1

        filtered_audio = torch.bmm(audio, attention).squeeze()  # bs x 512

        return filtered_audio

    def forward(self, x):
        video, audio = x  # bs x 8 x 3 x 112 x 112 , bs x 8 x 16 x 29
        # def forward(self, video, audio):

        video = video.permute(0, 2, 1, 3, 4)
        return self.r2plus1(video), self.filter_audio(audio)


class MultiModalNet(BinaryEvaluationMixin, SequenceClassificationModel):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__(num_classes=2, sequence_length=8, contains_dropout=False)
        self.similarity_net = SimilarityNetBigFiltered(num_classes=5, sequence_length=8)
        self.video_extractor = r2plus1d_18(pretrained=True)
        self.video_extractor.fc = nn.Identity()
        self._set_requires_grad_for_module(self.video_extractor, requires_grad=False)

        self.out = nn.Sequential(
            nn.Linear(512 + 512, 50),
            nn.BatchNorm1d(50),
            nn.LeakyReLU(0.2),
            nn.Linear(50, self.num_classes),
        )

    def loss_per_class(self, video_logits, audio_logits, targets):
        class_loss = torch.zeros((5,))
        class_counts = torch.zeros((5,))
        distances = (video_logits - audio_logits).pow(2).sum(1)
        for target, logit in zip(targets, distances):
            class_loss[target] += logit
            class_counts[target] += 1
        return class_loss / class_counts

    def forward(self, x):
        video, audio = x
        video = video.transpose(1, 2)
        video = self.video_extractor(video)

        diff_vid_embedding, diff_aud_embedding = self.similarity_net.forward(x)

        diff: torch.Tensor = diff_vid_embedding - diff_aud_embedding

        # classification should not influence this
        flat = torch.cat((video, diff.detach()), dim=1)
        out = self.out(flat)

        return out, (diff_vid_embedding, diff_aud_embedding)

    def training_step(self, batch, batch_nb, system):
        x, (label, audio_shift) = batch

        target = label // 4

        pred, embeddings = self.forward(x)
        classification_loss = self.loss(pred, target)
        contrastive_loss = self.similarity_net.loss(embeddings, target)
        total_loss = classification_loss + contrastive_loss

        train_acc = self.calculate_accuracy(pred, target)

        lightning_log = {"loss": total_loss}
        tensorboard_log = {
            "loss": {"train": total_loss},
            "classification_loss": classification_loss,
            "constrastive_loss": contrastive_loss,
            "acc": {"train": train_acc},
            "vid_std": torch.std(embeddings[0]),
            "aud_std": torch.std(embeddings[1]),
        }

        class_loss = self.loss_per_class(embeddings[0], embeddings[1], label)
        tensorboard_log["class_loss_train"] = {
            str(idx): val for idx, val in enumerate(class_loss)
        }
        tensorboard_log["class_loss_diff_train"] = {
            str(idx): val - class_loss[4] for idx, val in enumerate(class_loss[:4])
        }

        return tensorboard_log, lightning_log

    def aggregate_outputs(self, outputs, system):
        if len(system.val_dataloader()) > 1:
            outputs = outputs[0]

        with torch.no_grad():
            pred = torch.cat([x["pred"][0] for x in outputs], 0)
            label = torch.cat([x["target"][0] for x in outputs], 0)
            target = label // 4

            video_logits = torch.cat([x["pred"][1][0] for x in outputs], 0)
            audio_logtis = torch.cat([x["pred"][1][1] for x in outputs], 0)
            class_loss = self.loss_per_class(video_logits, audio_logtis, label)

            loss_mean_classification = self.loss(pred, target)
            contrastive_loss = self.similarity_net.loss(
                (video_logits, audio_logtis), label
            )
            total_loss = loss_mean_classification + contrastive_loss
            pred = pred.cpu()
            target = target.cpu()
            pred = F.softmax(pred, dim=1)
            acc_mean = self.calculate_accuracy(pred, target)

            # confusion matrix
            cm = confusion_matrix(label, torch.argmax(pred, dim=1), num_classes=5)
            cm = cm[:, :2]  # this is only binary classification
            cm[0] = torch.sum(cm[:-1], dim=0)
            cm[1] = cm[-1]
            accs = cm.diag() / torch.sum(cm[:2, :2], dim=1)
            class_accuracies = system.log_confusion_matrix(label, pred)
            class_accuracies[list(class_accuracies.keys())[0]] = accs[0]
            class_accuracies[list(class_accuracies.keys())[1]] = accs[1]

            tensorboard_log = {
                "loss": total_loss,
                "acc": acc_mean,
                "classification_loss": loss_mean_classification,
                "constrastive_loss": contrastive_loss,
                "class_loss_val": {str(idx): val for idx, val in enumerate(class_loss)},
                "class_loss_diff_val": {
                    str(idx): val - class_loss[4]
                    for idx, val in enumerate(class_loss[:4])
                },
                "vid_std": torch.std(video_logits),
                "aud_std": torch.std(audio_logtis),
            }

        return tensorboard_log, {}


class MutliModalNetFrozenSimNet(MultiModalNet):
    def __init__(self, num_classes):
        super().__init__(num_classes=2)
        self._set_requires_grad_for_module(
            self.similarity_net.r2plus1, requires_grad=False
        )


# todo here
class MutliModalNetFrozenSimNetNonDetach(MutliModalNetFrozenSimNet):
    def forward(self, x):
        video, audio = x
        video = video.transpose(1, 2)
        video = self.video_extractor(video)

        diff_vid_embedding, diff_aud_embedding = self.similarity_net.forward(x)

        diff: torch.Tensor = diff_vid_embedding - diff_aud_embedding

        # classification should not influence this
        flat = torch.cat((video, diff), dim=1)
        out = self.out(flat)

        return out, (diff_vid_embedding, diff_aud_embedding)


class MultiModalNetFrozenSimNetNonDetachNonFiltered(MutliModalNetFrozenSimNetNonDetach):
    def __init__(self, num_classes):
        super().__init__(num_classes=2)

        def _forward(x):
            video, audio = x  # bs x 8 x 3 x 112 x 112 , bs x 8 x 16 x 29
            # def forward(self, video, audio):
            audio = (
                audio.reshape((audio.shape[0], -1, 13))
                .unsqueeze(1)
                .expand(-1, 3, -1, -1)
            )

            video = video.permute(0, 2, 1, 3, 4)
            return (
                self.similarity_net.r2plus1(video),
                self.similarity_net.audio_extractor(audio),
            )

        self.similarity_net.forward = _forward


class MultiModalNetPretrained50ShiftNonFilter(MutliModalNetFrozenSimNet):
    def __init__(self, num_classes):
        super().__init__(num_classes=2)
        checkpoint = torch.load(
            "/mnt/raid/sebastian/log/debug/version_332/checkpoints/_ckpt_epoch_3.ckpt"
        )
        state_dict = checkpoint["state_dict"]
        self_state = self.similarity_net.state_dict()
        for name, param in state_dict.items():
            self_state[name.replace("model.", "")].copy_(param)

        def _forward(x):
            video, audio = x  # bs x 8 x 3 x 112 x 112 , bs x 8 x 16 x 29
            # def forward(self, video, audio):
            audio = (
                audio.reshape((audio.shape[0], -1, 13))
                .unsqueeze(1)
                .expand(-1, 3, -1, -1)
            )

            video = video.permute(0, 2, 1, 3, 4)
            return (
                self.similarity_net.r2plus1(video),
                self.similarity_net.audio_extractor(audio),
            )

        self.similarity_net.forward = _forward
        self._set_requires_grad_for_module(self.similarity_net, requires_grad=False)


class SimilarityNetBigNonFiltered(SimilarityNetBigFiltered):
    def __init__(self, num_classes):
        super().__init__(num_classes=2)

        def _forward(x):
            video, audio = x  # bs x 8 x 3 x 112 x 112 , bs x 8 x 16 x 29
            # def forward(self, video, audio):
            audio = (
                audio.reshape((audio.shape[0], -1, 13))
                .unsqueeze(1)
                .expand(-1, 3, -1, -1)
            )

            video = video.permute(0, 2, 1, 3, 4)
            return (self.r2plus1(video), self.audio_extractor(audio))

        self.forward = _forward


class Hello:
    def __init__(self):
        self.var1 = "home"

    def get_var1(self):
        return self.var1

    def print_var1(self):
        """Print the var1 in the output stream """
        print(self.get_var1())
