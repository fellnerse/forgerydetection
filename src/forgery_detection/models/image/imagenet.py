import abc
import logging

import torch
from torch import nn
from torchvision.models import resnext101_32x8d

from forgery_detection.lightning.logging.const import NAN_TENSOR
from forgery_detection.lightning.logging.const import VAL_ACC
from forgery_detection.models.image.multi_class_classification import Resnet18
from forgery_detection.models.mixins import PretrainedNet
from forgery_detection.models.utils import SequenceClassificationModel


logger = logging.getLogger(__file__)


class ImageNet(SequenceClassificationModel):
    @abc.abstractmethod
    def forward(self, x):
        raise NotImplementedError(
            "This needs to return a tensor with size 1005 for each input sample."
        )

    def _loss(self, pred, target):
        if len(pred):
            return super().loss(pred, target)
        else:
            return NAN_TENSOR

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
            return (
                super().calculate_accuracy(ff_pred, ff_target - 1000),
                super().calculate_accuracy(imagenet_pred, imagenet_target),
            )

        if sum(ff_mask) == 0:
            return (
                NAN_TENSOR,
                super().calculate_accuracy(imagenet_pred, imagenet_target),
            )
        else:
            return super().calculate_accuracy(ff_pred, ff_target - 1000), NAN_TENSOR

    @staticmethod
    def sum_nan_tensors(*args):
        return sum(map(lambda _x: 0 if _x != _x else _x, args))

    def training_step(self, batch, batch_nb, system):
        x, target = batch

        pred = self.forward(x)
        ff_loss, imagenet_loss = self.loss(pred, target)

        loss = self.sum_nan_tensors(ff_loss, imagenet_loss)
        lightning_log = {"loss": loss}

        with torch.no_grad():
            ff_acc, imagenet_acc = self.calculate_accuracy(pred, target)

            tensorboard_log = {
                "loss": {"train": loss},
                "imagnet_acc": {"train": imagenet_acc},
                "imagenet_loss": {"train": imagenet_loss},
            }

            if not ff_acc != ff_acc:
                tensorboard_log["ff_acc"] = {"train": ff_acc}

            if not ff_loss != ff_loss:
                tensorboard_log["ff_loss"] = {"train": ff_loss}

        return tensorboard_log, lightning_log

    def aggregate_outputs(self, outputs, system):
        if len(outputs) != 2:
            raise ValueError(
                "There is a problem with outputs. "
                "Probably the number of val dataloaders is not 2. "
                "The first one needs to give only imagenet images, the 2nd only FF."
            )
        imagenet_val, ff_val = outputs
        with torch.no_grad():
            imagenet_acc_mean, imagenet_loss_mean, _ = self._calculate_metrics(
                imagenet_val, cut_off=0
            )
            ff_acc_mean, ff_loss_mean, class_accuracies = self._calculate_metrics(
                ff_val, cut_off=1000, system=system
            )

        tensorboard_log = {
            "loss": imagenet_loss_mean + ff_loss_mean,
            "imagnet_acc": imagenet_acc_mean,
            "imagenet_loss": imagenet_loss_mean,
            "ff_acc": ff_acc_mean,
            "ff_loss": ff_loss_mean,
            "class_acc": class_accuracies,
        }
        lightning_log = {VAL_ACC: imagenet_acc_mean}

        return tensorboard_log, lightning_log

    def aggregate_outputs_test(self, outputs, system):

        # imagenet_val_loader_batch, ff_val_loader_batch = outputs
        ff_val_loader_batch = outputs
        with torch.no_grad():
            # imagenet_acc_mean, imagenet_loss_mean, _ = self._calculate_metrics(
            #     imagenet_val_loader_batch, cut_off=0
            # )
            ff_acc_mean, ff_loss_mean, class_accuracies = self._calculate_metrics(
                ff_val_loader_batch, cut_off=1000, system=system
            )

        tensorboard_log = {
            # "loss": imagenet_loss_mean + ff_loss_mean,
            # "imagnet_acc": imagenet_acc_mean,
            # "imagenet_loss": imagenet_loss_mean,
            "ff_acc": ff_acc_mean,
            "ff_loss": ff_loss_mean,
            "class_acc": class_accuracies,
        }
        lightning_log = {}  # {VAL_ACC: imagenet_acc_mean}

        return tensorboard_log, lightning_log

    def _calculate_metrics(self, output, cut_off=0, system=None):
        pred = torch.cat([x["pred"] for x in output], 0)[:, cut_off : 1000 + cut_off]
        target = torch.cat([x["target"] for x in output], 0) - cut_off
        loss_mean = self.sum_nan_tensors(*self.loss(pred, target))
        acc_mean = self.sum_nan_tensors(*self.calculate_accuracy(pred, target))

        if system:
            return (
                acc_mean,
                loss_mean,
                system.log_confusion_matrix(target.cpu(), pred.cpu()),
            )
        else:
            return acc_mean, loss_mean, None


class ImageNetResnet(Resnet18, ImageNet):
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


class ImageNetResnet152(ImageNet):
    def __init__(
        self,
        num_classes=1000,
        sequence_length=1,
        pretrained=True,
        contains_dropout=False,
    ):
        num_classes = 1000
        super().__init__(
            num_classes, sequence_length, contains_dropout=contains_dropout
        )
        self.resnet = resnext101_32x8d(pretrained=pretrained, num_classes=1000)
        if num_classes != 1000:
            old_fc = self.resnet.fc
            self.resnet.fc = nn.Linear(2048, num_classes)
            with torch.no_grad():
                min_classes = min(num_classes, old_fc.out_features)
                self.resnet.fc.weight[:min_classes] = old_fc.weight[:min_classes]
                self.resnet.fc.bias[:min_classes] = old_fc.bias[:min_classes]

        self.ff_classifier = nn.Linear(2048, 5)

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


class PretrainFFFCResnet152(SequenceClassificationModel):
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
        self.resnet = resnext101_32x8d(pretrained=pretrained, num_classes=1000)
        if num_classes != 1000:
            old_fc = self.resnet.fc
            self.resnet.fc = nn.Linear(2048, num_classes)
            with torch.no_grad():
                min_classes = min(num_classes, old_fc.out_features)
                self.resnet.fc.weight[:min_classes] = old_fc.weight[:min_classes]
                self.resnet.fc.bias[:min_classes] = old_fc.bias[:min_classes]

        self._set_requires_grad_for_module(self.resnet, requires_grad=False)
        self._set_requires_grad_for_module(self.resnet.fc, requires_grad=True)

    def forward(self, x):
        return self.resnet.forward(x)


class PretrainedFFFCResnet152(
    PretrainedNet(
        "/home/sebastian/log/runs/TRAIN/imagenet_ff/version_4/checkpoints/_ckpt_epoch_2.ckpt"
    ),
    ImageNetResnet152,
):
    pass
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     state_dict = torch.load(
    #         "/data/hdd/model_checkpoints/imagenet_net/resnext/model.ckpt"
    #     )["state_dict"]
    #     self.ff_classifier.load_state_dict(
    #         {
    #             "weight": state_dict["model.resnet.fc.weight"],
    #             "bias": state_dict["model.resnet.fc.bias"],
    #         }
    #     )


class PretrainedImageNetResnet(
    PretrainedNet(
        "/mnt/raid5/sebastian/model_checkpoints/imagenet_net/5_epochs_sgd_1.3e-4/model.ckpt"
    ),
    ImageNetResnet,
):
    pass
