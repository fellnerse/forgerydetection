from torch import nn
from torchvision.models import resnet18


class SequenceClassificationModel(nn.Module):
    def __init__(self, num_classes, sequence_length, contains_dropout=False):
        super().__init__()
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.contains_dropout = contains_dropout

    @staticmethod
    def _set_requires_grad_for_module(module, requires_grad=False):
        for param in module.parameters():
            param.requires_grad = requires_grad


class PretrainedResnet18(SequenceClassificationModel):
    def __init__(self, num_classes, sequence_length, contains_dropout=False):
        super().__init__(num_classes, sequence_length, contains_dropout)

        self.resnet = resnet18(pretrained=True, num_classes=1000)

        self.resnet.layer4 = nn.Identity()
        self.resnet.fc = nn.Linear(256, num_classes)
        self.eval()

    def forward(self, x):
        return self.resnet.forward(x)
