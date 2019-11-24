import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

from forgery_detection.models.utils import PretrainedResnet18


class Resnet183D(PretrainedResnet18):
    def __init__(self,):
        super().__init__(num_classes=5, sequence_length=8, contains_dropout=True)
        self.resnet = resnet18(pretrained=True, num_classes=1000)

        self.resnet.conv1 = nn.Conv3d(8, 64, kernel_size=(3, 7, 7), bias=False)
        self.resnet.layer1 = nn.Sequential(nn.Dropout2d(0.1), self.resnet.layer1)
        self.resnet.layer2 = nn.Sequential(nn.Dropout2d(0.2), self.resnet.layer2)
        self.resnet.layer3 = nn.Sequential(nn.Dropout2d(0.3), self.resnet.layer3)
        self.resnet.layer4 = nn.Identity()
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256, self.num_classes)
        )

    def forward(self, x):
        x = self.resnet.conv1(x).squeeze(2)
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

    def loss(self, logits, labels):
        return F.cross_entropy(logits, labels)
