from torch import nn
from torchvision.models import resnet18

from forgery_detection.models.utils import SequenceClassificationModel


class Resnet182D(SequenceClassificationModel):
    def __init__(
        self, num_classes, sequence_length, pretrained, contains_dropout=False
    ):
        super().__init__(
            num_classes, sequence_length, contains_dropout=contains_dropout
        )
        self.resnet = resnet18(pretrained=pretrained, num_classes=1000)

        self.resnet.layer4 = nn.Identity()
        self.resnet.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.resnet.forward(x)


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
