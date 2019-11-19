from torch import nn

from forgery_detection.models.classification import Resnet18


class Resnet18MultiClass(Resnet18):
    def __init__(self):
        super().__init__(5)


class Resnet18MultiClassSmall(Resnet18):
    def __init__(self):
        super().__init__(5)
        self.resnet.layer3 = nn.Identity()
        self.resnet.fc = nn.Linear(128, 5)


class Resnet18MultiClassDropout(Resnet18MultiClass):
    def __init__(self):
        super().__init__()

        self.resnet.layer1 = nn.Sequential(nn.Dropout2d(0.1), self.resnet.layer1)
        self.resnet.layer2 = nn.Sequential(nn.Dropout2d(0.2), self.resnet.layer2)
        self.resnet.layer3 = nn.Sequential(nn.Dropout2d(0.3), self.resnet.layer3)
        self.resnet.fc = nn.Sequential(nn.Dropout(0.5), self.resnet.fc)


class Resnet18MultiClassFrozen(Resnet18MultiClass):
    def __init__(self):
        super().__init__()

        # freeze everything besides 2. half of last layer
        self._set_requires_grad_for_module(self.resnet.conv1, False)
        self._set_requires_grad_for_module(self.resnet.bn1, False)
        self._set_requires_grad_for_module(self.resnet.relu, False)
        self._set_requires_grad_for_module(self.resnet.maxpool, False)
        self._set_requires_grad_for_module(self.resnet.layer1, False)
        self._set_requires_grad_for_module(self.resnet.layer2, False)
        self._set_requires_grad_for_module(self.resnet.layer3[0], False)  # 1. resblock
        # 2. resblock only first half
        self._set_requires_grad_for_module(self.resnet.layer3[1].conv1, False)
        self._set_requires_grad_for_module(self.resnet.layer3[1].bn1, False)
        self._set_requires_grad_for_module(self.resnet.layer3[1].relu, False)


class Resnet18MultiClassFrozen2(Resnet18MultiClass):
    def __init__(self):
        super().__init__()

        # freeze everything besides last fc
        self._set_requires_grad_for_module(self.resnet.conv1, False)
        self._set_requires_grad_for_module(self.resnet.bn1, False)
        self._set_requires_grad_for_module(self.resnet.relu, False)
        self._set_requires_grad_for_module(self.resnet.maxpool, False)
        self._set_requires_grad_for_module(self.resnet.layer1, False)
        self._set_requires_grad_for_module(self.resnet.layer2, False)
        self._set_requires_grad_for_module(self.resnet.layer3, False)


class Resnet18MultiClassDropoutFrozen(Resnet18MultiClass):
    def __init__(self):
        super().__init__()
        self.resnet.layer3 = nn.Sequential(nn.Dropout2d(0.15), self.resnet.layer3)
        self.resnet.fc = nn.Sequential(nn.Dropout(0.5), self.resnet.fc)

        # freeze everything besides 2. half of last layer
        self._set_requires_grad_for_module(self.resnet.conv1, False)
        self._set_requires_grad_for_module(self.resnet.bn1, False)
        self._set_requires_grad_for_module(self.resnet.relu, False)
        self._set_requires_grad_for_module(self.resnet.maxpool, False)
        self._set_requires_grad_for_module(self.resnet.layer1, False)
        self._set_requires_grad_for_module(self.resnet.layer2, False)
        self._set_requires_grad_for_module(self.resnet.layer3[0], False)  # dropout
        self._set_requires_grad_for_module(
            self.resnet.layer3[1][0], False
        )  # 1. resblock
        # 2. resblock only second half
        self._set_requires_grad_for_module(self.resnet.layer3[1][1].conv1, False)
        self._set_requires_grad_for_module(self.resnet.layer3[1][1].bn1, False)
        self._set_requires_grad_for_module(self.resnet.layer3[1][1].relu, False)
