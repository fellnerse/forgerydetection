from torch import nn
from torchvision import models


class VGG11Binary(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg11_bn = models.vgg11_bn(pretrained=True, num_classes=1000)

        self.vgg11_bn.classifier = nn.Sequential(
            *list(self.vgg11_bn.classifier)[:-1],
            nn.Linear(in_features=4096, out_features=2, bias=True),
        )

    def forward(self, x):
        return self.vgg11_bn.forward(x)


class SqueezeBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.squeeze = models.squeezenet1_1(pretrained=True, num_classes=1000)

        final_conv = nn.Conv2d(512, 2, kernel_size=1)
        self.squeeze.classifier = nn.Sequential(nn.Dropout(p=0.5), final_conv)

    def forward(self, x):
        return self.squeeze.forward(x)


class Resnet18Binary(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True, num_classes=1000)

        self.resnet.layer4 = nn.Identity()
        self.resnet.fc = nn.Linear(256, 2)

    def forward(self, x):
        return self.resnet.forward(x)


class Resnet18BinaryDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True, num_classes=1000)

        self.resnet.layer1 = nn.Sequential(nn.Dropout2d(0.1), self.resnet.layer1)
        self.resnet.layer2 = nn.Sequential(nn.Dropout2d(0.2), self.resnet.layer2)
        self.resnet.layer3 = nn.Sequential(nn.Dropout2d(0.3), self.resnet.layer3)
        self.resnet.layer4 = nn.Identity()
        self.resnet.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(256, 2))

    def forward(self, x):
        return self.resnet.forward(x)
