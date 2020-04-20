import logging

import torch
from torch import nn
from torchvision.models.resnet import _resnet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import resnet18

from forgery_detection.models.utils import SequenceClassificationModel

logger = logging.getLogger(__file__)


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
