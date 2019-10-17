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
