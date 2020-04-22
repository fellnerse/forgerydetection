import logging

import torch
from torch import nn
from torchvision.models.video import r2plus1d_18

from forgery_detection.models.audio.similarity_stuff import PretrainedSyncNet
from forgery_detection.models.audio.utils import ContrastiveLoss
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

            audio_target = label // 5

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
