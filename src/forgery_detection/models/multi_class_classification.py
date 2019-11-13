from torch import nn
from torchvision import models


class Resnet18MultiClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True, num_classes=1000)

        self.resnet.layer4 = nn.Identity()
        self.resnet.fc = nn.Linear(256, 5)

    def forward(self, x):
        return self.resnet.forward(x)


class Resnet18MultiClassFrozen(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True, num_classes=1000)

        self.resnet.layer4 = nn.Identity()
        self.resnet.fc = nn.Linear(256, 5)

        # freeze everything besides 2. half of last layer
        self._set_requires_grad_for_module(self.resnet.layer1, False)
        self._set_requires_grad_for_module(self.resnet.layer2, False)
        self._set_requires_grad_for_module(self.resnet.layer3[0], False)  # 1. resblock
        # 2. resblock only first half
        self._set_requires_grad_for_module(self.resnet.layer3[1].conv1, False)
        self._set_requires_grad_for_module(self.resnet.layer3[1].bn1, False)
        self._set_requires_grad_for_module(self.resnet.layer3[1].relu, False)

    def _set_requires_grad_for_module(self, module, requires_grad=False):
        for param in module.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        return self.resnet.forward(x)
