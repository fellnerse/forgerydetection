import torch
from torch import nn
from torchvision.models.resnet import _resnet
from torchvision.models.resnet import BasicBlock

from forgery_detection.models.utils import SequenceClassificationModel


class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()


class AudioNet(SequenceClassificationModel):
    def __init__(self, pretrained=True):
        super().__init__(num_classes=5, sequence_length=8, contains_dropout=False)
        self.r2plus1 = torch.hub.load(
            "moabitcoin/ig65m-pytorch",
            "r2plus1d_34_8_kinetics",
            num_classes=400,
            pretrained=pretrained,
        )
        self.r2plus1.fc = nn.Linear(512, 256)

        self.resnet = _resnet(
            "resnet18",
            BasicBlock,
            [1, 1, 1, 1],
            pretrained=False,
            progress=True,
            num_classes=256,
        )
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resnet.layer4 = nn.Identity()
        self.resnet.fc = nn.Linear(256, 256)

        self.relu = nn.ReLU()
        self.out = nn.Linear(512, self.num_classes)

    def forward(self, x):
        video, audio = x

        video = video.transpose(1, 2)
        video = self.r2plus1(video)
        audio = self.resnet(audio.unsqueeze(1))

        flat = torch.cat((video, audio), dim=1)
        out = self.out(self.relu(flat))
        return out
