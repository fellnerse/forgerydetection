from torch import nn
from torchvision.models import resnet18

from forgery_detection.models.utils import SequenceClassificationModel


class Resnet18(SequenceClassificationModel):
    def __init__(
        self, num_classes=5, sequence_length=1, pretrained=True, contains_dropout=False
    ):
        super().__init__(
            num_classes, sequence_length, contains_dropout=contains_dropout
        )
        self.resnet = resnet18(pretrained=pretrained, num_classes=1000)
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.resnet.forward(x)


class Resnet182D(Resnet18):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.resnet.layer4 = nn.Identity()
        self.resnet.fc = nn.Linear(256, self.num_classes)


class Resnet182d2Blocks(Resnet182D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.resnet.layer3 = nn.Identity()
        self.resnet.fc = nn.Linear(128, self.num_classes)


class Resnet182d1Block(Resnet182d2Blocks):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.resnet.layer2 = nn.Identity()
        self.resnet.fc = nn.Linear(64, self.num_classes)


class Resnet182d1BlockFrozen(Resnet182d1Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_requires_grad_for_module(self.resnet.conv1, requires_grad=False)
        self._set_requires_grad_for_module(self.resnet.bn1, requires_grad=False)
        self._set_requires_grad_for_module(self.resnet.layer1, requires_grad=False)


class Resnet182d2BlocksFrozen(Resnet182d2Blocks):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_requires_grad_for_module(self.resnet.conv1, requires_grad=False)
        self._set_requires_grad_for_module(self.resnet.bn1, requires_grad=False)
        self._set_requires_grad_for_module(self.resnet.layer1, requires_grad=False)
        self._set_requires_grad_for_module(self.resnet.layer2, requires_grad=False)


class Resnet182dFrozen(Resnet182D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_requires_grad_for_module(self.resnet.conv1, requires_grad=False)
        self._set_requires_grad_for_module(self.resnet.bn1, requires_grad=False)
        self._set_requires_grad_for_module(self.resnet.layer1, requires_grad=False)
        self._set_requires_grad_for_module(self.resnet.layer2, requires_grad=False)
        self._set_requires_grad_for_module(self.resnet.layer3, requires_grad=False)


class Resnet18Frozen(Resnet18):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_requires_grad_for_module(self.resnet.conv1, requires_grad=False)
        self._set_requires_grad_for_module(self.resnet.bn1, requires_grad=False)
        self._set_requires_grad_for_module(self.resnet.layer1, requires_grad=False)
        self._set_requires_grad_for_module(self.resnet.layer2, requires_grad=False)
        self._set_requires_grad_for_module(self.resnet.layer3, requires_grad=False)
        self._set_requires_grad_for_module(self.resnet.layer4, requires_grad=False)


class ResidualResnet(Resnet182d2Blocks):
    def __init__(self, **kwargs):
        super().__init__(sequence_length=2, **kwargs)

    def forward(self, x):
        first_frame = x[:, 0, :, :, :]
        second_frame = x[:, 1, :, :, :]
        residual_frame = second_frame - first_frame

        return self.resnet.forward(residual_frame.squeeze(1))


class Resnet18MultiClassDropout(Resnet182D):
    def __init__(self, pretrained=True):
        super().__init__(
            num_classes=5,
            sequence_length=1,
            contains_dropout=True,
            pretrained=pretrained,
        )

        self.resnet.layer1 = nn.Sequential(nn.Dropout2d(0.1), self.resnet.layer1)
        self.resnet.layer2 = nn.Sequential(nn.Dropout2d(0.2), self.resnet.layer2)
        self.resnet.layer3 = nn.Sequential(nn.Dropout2d(0.3), self.resnet.layer3)
        self.resnet.fc = nn.Sequential(nn.Dropout(0.5), self.resnet.fc)


class Resnet18UntrainedMultiClassDropout(Resnet18MultiClassDropout):
    def __init__(self):
        super().__init__(pretrained=False)
