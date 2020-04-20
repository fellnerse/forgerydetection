import logging

import torch
from torch import nn
from torchvision.models.resnet import _resnet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import resnet18
from torchvision.models.video import r2plus1d_18

from forgery_detection.models.audio.similarity_stuff import PretrainedSyncNet
from forgery_detection.models.mixins import PretrainedNet
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

    def training_step(self, batch, batch_nb, system):
        x, (target, _) = batch
        return super().training_step((x, target), batch_nb, system)

    def aggregate_outputs(self, outputs, system):
        for x in outputs:
            x["target"] = x["target"][0]
        return super().aggregate_outputs(outputs, system)


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


class SyncAudioNet(SequenceClassificationModel):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__(
            num_classes=num_classes, sequence_length=8, contains_dropout=False
        )
        self.r2plus1 = torch.hub.load(
            "moabitcoin/ig65m-pytorch",
            "r2plus1d_34_8_kinetics",
            num_classes=400,
            pretrained=True,
        )
        self.r2plus1.layer3 = nn.Identity()
        self.r2plus1.layer4 = nn.Identity()
        self.r2plus1.fc = nn.Identity()

        self.sync_net = PretrainedSyncNet()
        self._set_requires_grad_for_module(self.sync_net, requires_grad=False)

        self.relu = nn.ReLU()
        self.out = nn.Sequential(
            nn.Linear(128 + 1024, 50), nn.ReLU(), nn.Linear(50, self.num_classes)
        )

    def forward(self, x):
        # def forward(self, video, audio):
        video, audio = x  # bs x 8 x 3 x 112 x 112 , bs x 8 x 29
        # video = x  # bs x 8 x 3 x 112 x 112 , bs x 8 x 29

        video = video.transpose(1, 2)
        video = self.r2plus1(video)

        # syncnet only uses 5 frames
        audio = audio[:, 2:-1]
        audio = (audio.reshape((audio.shape[0], -1, 13)).unsqueeze(1)).transpose(-2, -1)
        audio = self.sync_net.audio_extractor(audio)

        flat = torch.cat((video, audio), dim=1)
        out = self.out(self.relu(flat))
        return out

    def training_step(self, batch, batch_nb, system):
        x, (target, _) = batch
        return super().training_step((x, target), batch_nb, system)

    def aggregate_outputs(self, outputs, system):
        for x in outputs:
            x["target"] = x["target"][0]
        return super().aggregate_outputs(outputs, system)


class PretrainSyncAudioNet(
    PretrainedNet(
        "/mnt/raid/sebastian/log/runs/TRAIN/sync_audio_net/version_1/checkpoints/_ckpt_epoch_23.ckpt"
    ),
    SyncAudioNet,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_requires_grad_for_module(self.r2plus1, requires_grad=False)
