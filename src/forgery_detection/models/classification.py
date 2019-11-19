from torch import nn
from torchvision.models import resnet18


class Resnet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = resnet18(pretrained=True, num_classes=1000)

        self.resnet.layer4 = nn.Identity()
        self.resnet.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.resnet.forward(x)

    @staticmethod
    def _set_requires_grad_for_module(module, requires_grad=False):
        for param in module.parameters():
            param.requires_grad = requires_grad
