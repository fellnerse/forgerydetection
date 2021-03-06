import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.models.video.resnet import mc3_18

from forgery_detection.models.mixins import PretrainedNet
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
            sample_size=112 * 2 * 2,
            shortcut_type="A",
            sample_duration=self.sequence_length,
            num_classes=101,
        )
        # self.resnet.layer1 = nn.Sequential(nn.Dropout3d(0.1), self.resnet.layer1)
        # self.resnet.layer2 = nn.Sequential(nn.Dropout3d(0.2), self.resnet.layer2)
        self.resnet.layer3 = nn.Identity()
        self.resnet.layer4 = nn.Identity()
        # self.resnet.fc = nn.Sequential(
        #     nn.Dropout(0.5), nn.Linear(256, self.num_classes)
        # )
        self.resnet.fc = nn.Linear(256, self.num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.resnet(x)


class Resnet18Fully3DPretrained(Resnet18Fully3D):
    def __init__(self):
        super().__init__(pretrained=True)


class R2Plus1(SequenceClassificationModel):
    def __init__(self, num_classes=5, sequence_length=8, contains_dropout=False):
        super().__init__(
            num_classes=num_classes,
            sequence_length=sequence_length,
            contains_dropout=contains_dropout,
        )
        self.r2plus1 = torch.hub.load(
            "moabitcoin/ig65m-pytorch",
            "r2plus1d_34_8_kinetics",
            num_classes=400,
            pretrained=True,
        )
        self.r2plus1.fc = nn.Linear(512, self.num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.r2plus1(x)


class R2Plus1Frozen(R2Plus1):
    def __init__(self):
        super().__init__()
        self._set_requires_grad_for_module(self.r2plus1, requires_grad=False)
        self._set_requires_grad_for_module(self.r2plus1.fc, requires_grad=True)


class R2Plus1Small(R2Plus1):
    def __init__(self):
        super().__init__()
        self.contains_dropout = True
        self.r2plus1.layer1 = nn.Sequential(nn.Dropout3d(0.1), self.r2plus1.layer1)
        self.r2plus1.layer2 = nn.Sequential(nn.Dropout3d(0.2), self.r2plus1.layer2)
        self.r2plus1.layer3 = nn.Identity()
        self.r2plus1.layer4 = nn.Identity()
        self.r2plus1.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(128, self.num_classes)
        )


class R2Plus1SmallAudiolike(R2Plus1):
    def __init__(self, num_classes=5, sequence_length=8):
        super().__init__(
            num_classes=num_classes,
            sequence_length=sequence_length,
            contains_dropout=False,
        )

        self.r2plus1.layer3 = nn.Identity()
        self.r2plus1.layer4 = nn.Identity()
        self.r2plus1.fc = nn.Sequential(
            nn.Linear(128, 50), nn.ReLU(), nn.Linear(50, self.num_classes)
        )


class R2Plus1SmallAudioLikePretrain(
    PretrainedNet(
        "/home/sebastian/log/runs/TRAIN/pretrain_r2plus1/version_0/checkpoints/_ckpt_epoch_2.ckpt"
    ),
    R2Plus1SmallAudiolike,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_requires_grad_for_module(self.r2plus1, requires_grad=False)
        self._set_requires_grad_for_module(self.r2plus1.fc, requires_grad=True)


class R2Plus1SmallAudiolikePretrained(
    PretrainedNet(
        "/mnt/raid/sebastian/model_checkpoints/ff/r2plus1_with_mlp/model_epoch_4.ckpt"
    ),
    R2Plus1SmallAudiolike,
):
    pass


class R2Plus1Smallest(R2Plus1):
    def __init__(self, num_classes=5, sequence_length=8):
        super().__init__(
            num_classes=num_classes,
            sequence_length=sequence_length,
            contains_dropout=False,
        )

        self.r2plus1.layer2 = nn.Identity()
        self.r2plus1.layer3 = nn.Identity()
        self.r2plus1.layer4 = nn.Identity()
        self.r2plus1.fc = nn.Sequential(
            nn.Linear(64, 50), nn.ReLU(), nn.Linear(50, self.num_classes)
        )


class MC3(SequenceClassificationModel):
    def __init__(self, num_classes=5, sequence_length=8):
        super().__init__(
            num_classes=num_classes,
            sequence_length=sequence_length,
            contains_dropout=False,
        )
        self.mc3 = mc3_18(pretrained=True)
        self.mc3.layer4 = nn.Identity()
        self.mc3.fc = nn.Linear(256, self.num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.mc3(x)
