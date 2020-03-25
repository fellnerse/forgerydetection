import logging

import torch
from torch import nn
from torchvision.models import resnet18

from forgery_detection.lightning.utils import VAL_ACC
from forgery_detection.models.utils import SequenceClassificationModel


logger = logging.getLogger(__file__)


class Resnet18(SequenceClassificationModel):
    def __init__(
        self,
        num_classes=1000,
        sequence_length=1,
        pretrained=True,
        contains_dropout=False,
    ):
        super().__init__(
            num_classes, sequence_length, contains_dropout=contains_dropout
        )
        self.resnet = resnet18(pretrained=pretrained, num_classes=1000)
        if num_classes != 1000:
            old_fc = self.resnet.fc
            self.resnet.fc = nn.Linear(512, num_classes)
            with torch.no_grad():
                min_classes = min(num_classes, old_fc.out_features)
                self.resnet.fc.weight[:min_classes] = old_fc.weight[:min_classes]
                self.resnet.fc.bias[:min_classes] = old_fc.bias[:min_classes]

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


class ResidualResnet(Resnet182D):
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


class Resnet18SameAsInAE(Resnet18):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.resnet.layer4 = nn.Conv2d(256, 16, 3, 1, 1)
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 50), nn.ReLU(), nn.Linear(50, self.num_classes)
        )


class ImageNetResnet(Resnet18):
    def __init__(self, **kwargs):
        kwargs["num_classes"] = 1000
        super().__init__(**kwargs)
        self.ff_classifier = nn.Linear(512, 5)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        imagnet_pred = self.resnet.fc(x)
        ff_pred = self.ff_classifier(x)

        return torch.cat((imagnet_pred, ff_pred), dim=1)

    def _loss(self, pred, target):
        if len(pred):
            return super().loss(pred, target)
        else:
            return torch.zeros((1,)).to(pred.device)

    def loss(self, pred, target):
        ff_mask = target >= 1000
        ff_pred = pred[ff_mask][:, 1000:]
        ff_target = target[ff_mask]

        imagenet_pred = pred[~ff_mask][:, :1000]
        imagenet_target = target[~ff_mask]

        return (
            self._loss(ff_pred, ff_target - 1000),
            self._loss(imagenet_pred, imagenet_target),
        )

    def calculate_accuracy(self, pred, target):
        ff_mask = target >= 1000
        ff_pred = pred[ff_mask][:, 1000:]
        ff_target = target[ff_mask]

        imagenet_pred = pred[~ff_mask][:, :1000]
        imagenet_target = target[~ff_mask]

        if 0 < sum(ff_mask) < len(pred):
            return super().calculate_accuracy(ff_pred, ff_target - 1000) / len(
                ff_mask
            ) + super().calculate_accuracy(imagenet_pred, imagenet_target) / (
                len(pred) - len(ff_mask)
            )

        if sum(ff_mask) == 0:
            return super().calculate_accuracy(imagenet_pred, imagenet_target)
        else:
            return super().calculate_accuracy(ff_pred, ff_target - 1000)

    def training_step(self, batch, batch_nb, system):
        x, target = batch

        pred = self.forward(x)
        ff_loss, imagenet_loss = self.loss(pred, target)
        loss = ff_loss + imagenet_loss
        lightning_log = {"loss": loss}

        with torch.no_grad():
            train_acc = self.calculate_accuracy(pred, target)

            tensorboard_log = {
                "loss": {"train": loss},
                "acc": {"train": train_acc},
                # "ff_acc": {"train": ff_acc_mean},
                "ff_loss": {"train": ff_loss},
                "imagenet_loss": {"train": imagenet_loss},
            }  # todo instead of logging 0 maybe just dont log it -> could logg all stats for ff as well

        return tensorboard_log, lightning_log

    def aggregate_outputs(self, outputs, system):

        imagenet_val, ff_val = outputs
        with torch.no_grad():
            imagenet_acc_mean, imagenet_loss_mean, _ = self._calculate_metrics(
                imagenet_val, cut_off=0
            )
            ff_acc_mean, ff_loss_mean, class_accuracies = self._calculate_metrics(
                ff_val, cut_off=1000, system=system
            )

        tensorboard_log = {
            "imagnet_acc": imagenet_acc_mean,
            "imagenet_loss": imagenet_loss_mean,
            "ff_acc": ff_acc_mean,
            "ff_loss": ff_loss_mean,
            "class_acc": class_accuracies,
        }
        lightning_log = {VAL_ACC: imagenet_acc_mean}

        return tensorboard_log, lightning_log

    def _calculate_metrics(self, output, cut_off=0, system=None):
        pred = torch.cat([x["pred"] for x in output], 0)[:, cut_off : 1000 + cut_off]
        target = torch.cat([x["target"] for x in output], 0) - cut_off
        loss_mean = sum(self.loss(pred, target))
        acc_mean = self.calculate_accuracy(pred, target)

        if system:
            # todo find better way for class acc
            # https://discuss.pytorch.org/t/how-to-find-individual-class-accuracy/6348/2
            return (
                acc_mean,
                loss_mean,
                system.log_confusion_matrix(
                    target.cpu(), torch.argmax(pred, dim=1).cpu()
                ),
            )
        else:
            return acc_mean, loss_mean, None
