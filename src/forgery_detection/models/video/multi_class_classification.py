import torch
from torch import nn
from torchvision.models import resnet18

from forgery_detection.models.utils import SequenceClassificationModel
from forgery_detection.models.video import resnet_fully_3d


class Resnet183DNoDropout(SequenceClassificationModel):
    def __init__(self, pretrained=True):
        super().__init__(num_classes=5, sequence_length=8, contains_dropout=False)
        self.resnet = resnet18(pretrained=pretrained, num_classes=1000)

        self.resnet.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=(8, 7, 7),
            stride=(1, 2, 2),
            padding=(0, 3, 3),
            bias=False,
        )
        self.resnet.layer4 = nn.Identity()
        self.resnet.fc = nn.Linear(256, self.num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.resnet.conv1(x)
        x = x.squeeze(2)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)

        return x


class Resnet183D(Resnet183DNoDropout):
    def __init__(self, pretrained=True):
        super().__init__(pretrained=pretrained)
        self.contains_dropout = True

        self.resnet.layer1 = nn.Sequential(nn.Dropout2d(0.1), self.resnet.layer1)
        self.resnet.layer2 = nn.Sequential(nn.Dropout2d(0.2), self.resnet.layer2)
        self.resnet.layer3 = nn.Sequential(nn.Dropout2d(0.3), self.resnet.layer3)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256, self.num_classes)
        )


class Resnet183DUntrained(Resnet183D):
    def __init__(self):
        super().__init__(pretrained=False)


class Resnet18Fully3D(SequenceClassificationModel):
    def __init__(self, pretrained=False):
        super().__init__(num_classes=5, sequence_length=8, contains_dropout=True)
        self.resnet = resnet_fully_3d.resnet18(
            pretrained=pretrained,
            sample_size=224 * 2 * 2,
            shortcut_type="A",
            sample_duration=self.sequence_length,
            num_classes=101,
        )
        self.resnet.layer1 = nn.Sequential(nn.Dropout3d(0.1), self.resnet.layer1)
        self.resnet.layer2 = nn.Sequential(nn.Dropout3d(0.2), self.resnet.layer2)
        self.resnet.layer3 = nn.Identity()
        self.resnet.layer4 = nn.Identity()
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256, self.num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.resnet(x)


class Resnet18Fully3DPretrained(Resnet18Fully3D):
    def __init__(self):
        super().__init__(pretrained=True)
