from torch import nn

from forgery_detection.models.utils import PretrainedResnet18


class Resnet18MultiClassDropout(PretrainedResnet18):
    def __init__(self):
        super().__init__(num_classes=5, sequence_length=1, contains_dropout=True)

        self.resnet.layer1 = nn.Sequential(nn.Dropout2d(0.1), self.resnet.layer1)
        self.resnet.layer2 = nn.Sequential(nn.Dropout2d(0.2), self.resnet.layer2)
        self.resnet.layer3 = nn.Sequential(nn.Dropout2d(0.3), self.resnet.layer3)
        self.resnet.fc = nn.Sequential(nn.Dropout(0.5), self.resnet.fc)
