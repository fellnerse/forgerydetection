import logging

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.video import mc3_18
from torchvision.utils import make_grid

from forgery_detection.lightning.logging.const import NAN_TENSOR
from forgery_detection.lightning.logging.const import VAL_ACC
from forgery_detection.models.utils import SequenceClassificationModel

logger = logging.getLogger(__file__)


class ScrambleNet(SequenceClassificationModel):
    def __init__(self, num_classes=2, sequence_length=8, contains_dropout=False):
        super().__init__(num_classes, sequence_length, contains_dropout)

        self.mc3 = mc3_18(pretrained=True)
        self.mc3.layer4 = nn.Identity()
        self.mc3.fc = nn.Linear(256, self.num_classes)

    def forward(self, x):
        self._shuffle_x(x)
        x = x.transpose(1, 2)
        return self.mc3(x)

    def training_step(self, batch, batch_nb, system):
        x, target = batch
        self._set_target_shuffled(target)

        return super().training_step(batch, batch_nb, system)

    def _shuffle_x(self, x):
        middle = x.shape[0] // 2

        # swap two consecutive entries
        idx = torch.randint(0, self.sequence_length - 1, (1,))
        x[:middle, [idx, idx + 1]] = x[:middle, [idx + 1, idx]]

    def _set_target_shuffled(self, target):
        middle = target.shape[0] // 2
        # set second half of batch as not scrambled (aka normal)
        target *= 0
        target[middle:] += 1
        return target

    def aggregate_outputs(self, outputs, system):
        # if there are more then one dataloader we ignore the additional data
        if len(system.val_dataloader()) > 1:
            outputs_ = outputs[0]
        else:
            outputs_ = outputs

        target_classes = torch.cat([x["target"] for x in outputs_], 0)
        for x in outputs_:
            x["scramble_target"] = self._set_target_shuffled(x["target"])

        # log some entries
        x = outputs_[0]["x"]
        x_12 = x[:4].view(-1, 3, 112, 112)
        datapoints = make_grid(
            x_12, nrow=self.sequence_length, range=(-1, 1), normalize=True
        )
        system.logger.experiment.add_image(
            f"reconstruction/val",
            datapoints,
            dataformats="CHW",
            global_step=system.global_step,
        )

        pred = torch.cat([x["pred"] for x in outputs_], 0)
        target = torch.cat([x["scramble_target"] for x in outputs_], 0)

        loss_mean = self.loss(pred, target)
        pred = pred.cpu()
        target = target.cpu()
        pred = F.softmax(pred, dim=1)
        acc_mean = self.calculate_accuracy(pred, target)

        class_accuracies = self._get_class_accuracies(
            pred, target, target_classes, system
        )

        tensorboard_log = {
            "loss": loss_mean,
            "acc": acc_mean,
            "roc_auc": NAN_TENSOR,
            "class_acc": class_accuracies,
        }
        lightning_log = {VAL_ACC: acc_mean}

        return tensorboard_log, lightning_log

    def _get_class_accuracies(self, pred, target, target_classes, system):
        classes = list(system.file_list.class_to_idx.keys())

        class_accuracies = {}

        for class_ in classes:
            value = system.file_list.class_to_idx[class_]
            mask = target_classes == value
            if pred[mask].shape[0]:
                acc_mean = self.calculate_accuracy(pred[mask], target[mask])
            else:
                acc_mean = NAN_TENSOR
            class_accuracies[class_] = acc_mean

        return class_accuracies
