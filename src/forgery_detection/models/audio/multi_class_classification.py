import torch
from torch import nn
from torchvision.models.resnet import _resnet
from torchvision.models.resnet import BasicBlock
from torchvision.models.video import r2plus1d_18

from forgery_detection.models.utils import SequenceClassificationModel


class AudioNet(SequenceClassificationModel):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__(
            num_classes=num_classes, sequence_length=8, contains_dropout=False
        )
        self.r2plus1 = r2plus1d_18(pretrained=pretrained)
        self.r2plus1.layer3 = nn.Identity()
        self.r2plus1.layer4 = nn.Identity()
        self.r2plus1.fc = nn.Identity()

        self.resnet = _resnet(
            "resnet18",
            BasicBlock,
            [1, 1, 1, 1],
            pretrained=False,
            progress=True,
            num_classes=1000,
        )
        self.resnet.layer3 = nn.Identity()
        self.resnet.layer4 = nn.Identity()
        self.resnet.fc = nn.Identity()

        self.relu = nn.ReLU()
        self.out = nn.Sequential(
            nn.Linear(256, 50), nn.ReLU(), nn.Linear(50, self.num_classes)
        )

    def forward(self, x):
        # def forward(self, video, audio):
        video, audio = x  # bs x 8 x 3 x 112 x 112 , bs x 8 x 40

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
