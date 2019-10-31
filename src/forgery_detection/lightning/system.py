from argparse import Namespace
from pathlib import Path
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import confusion_matrix
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from forgery_detection.data.face_forensics.splits import TEST_NAME
from forgery_detection.data.face_forensics.splits import TRAIN_NAME
from forgery_detection.data.face_forensics.splits import VAL_NAME
from forgery_detection.data.utils import get_data
from forgery_detection.lightning.confusion_matrix import plot_cm
from forgery_detection.lightning.confusion_matrix import plot_to_image
from forgery_detection.lightning.utils import _get_fixed_dataloader
from forgery_detection.lightning.utils import calculate_class_weights
from forgery_detection.lightning.utils import DictHolder
from forgery_detection.models.binary_classification import Resnet18Binary
from forgery_detection.models.binary_classification import SqueezeBinary
from forgery_detection.models.binary_classification import VGG11Binary


class Supervised(pl.LightningModule):
    MODEL_DICT = {
        "squeeze": SqueezeBinary,
        "vgg11": VGG11Binary,
        "resnet18": Resnet18Binary,
    }

    def __init__(self, kwargs: Union[dict, Namespace]):
        super(Supervised, self).__init__()

        self.hparams = DictHolder(kwargs)
        self.model = self.MODEL_DICT[self.hparams["model"]]()

        # if we are training we have to log some stuff to hparams
        if self.hparams.pop("train", False):

            # lazily load dataloaders
            self.train_dataloader()
            self.val_dataloader()

            if self.hparams["balance_data"]:
                labels, weights = calculate_class_weights(
                    get_data(Path(self.hparams["data_dir"]) / VAL_NAME)
                )
                self.hparams.add_class_weights(labels, weights)
                self.class_weights = torch.tensor(weights, dtype=torch.float)
            else:
                self.class_weights = None
        else:
            self.train_dataloader()
            self.class_weights = None

    def forward(self, x):
        return self.model.forward(x)

    def loss(self, logits, labels):
        try:
            # classweights  -> multiply each output with weight of target
            #               -> sum result up and divide by sum of these weights
            cross_engropy = F.cross_entropy(logits, labels, weight=self.class_weights)
        except RuntimeError:
            device_index = logits.device.index
            print(f"switching device for class_weights to {device_index}")
            self.class_weights = self.class_weights.cuda(device_index)
            cross_engropy = F.cross_entropy(logits, labels, weight=self.class_weights)
        return cross_engropy

    def training_step(self, batch, batch_nb):
        x, target = batch
        pred = self.forward(x)

        loss_val = self.loss(pred, target)
        train_acc = self._calculate_accuracy(pred, target)

        log = {"loss": loss_val, "acc": train_acc}

        return self._construct_lightning_log(log, suffix="train")

    def validation_step(self, batch, batch_nb):
        x, target = batch
        pred = self.forward(x)

        loss_val = self.loss(pred, target)
        val_acc = self._calculate_accuracy(pred, target)

        return {"loss": loss_val, "acc": val_acc}

    def validation_end(self, outputs):
        # aggregate values from validation step

        val_loss_mean = torch.stack([x["loss"] for x in outputs]).mean()
        val_acc_mean = torch.stack([x["acc"] for x in outputs]).mean()

        log = {"loss": val_loss_mean, "acc": val_acc_mean}

        return self._construct_lightning_log(log, suffix="val")

    def test_step(self, batch, batch_nb):
        x, target = batch
        pred = self.forward(x)

        loss_val = self.loss(pred, target)
        val_acc = self._calculate_accuracy(pred, target)

        cm = confusion_matrix(target.cpu(), torch.argmax(pred, dim=1).cpu())

        if cm.shape != (2, 2):
            cm = [[0, 0], [0, 0]]

        return {"loss": loss_val, "acc": val_acc, "cm": cm}

    def test_end(self, outputs):
        # aggregate values from validation step

        val_loss_mean = torch.stack([x["loss"] for x in outputs]).mean()
        val_acc_mean = torch.stack([x["acc"] for x in outputs]).mean()
        if len(outputs) > 1:
            val_cm_sum = np.stack([x["cm"] for x in outputs]).sum(axis=0)
        else:
            val_cm_sum = outputs[0]["cm"]
        figure = plot_cm(val_cm_sum, class_names=["fake", "real"])
        cm_image = plot_to_image(figure)

        self.logger.experiment.add_image(
            "confusion matrix",
            cm_image,
            dataformats="HWC",
            global_step=self.global_step,
        )

        log = {"loss": val_loss_mean, "acc": val_acc_mean}

        return self._construct_lightning_log(log, suffix="test")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams["lr"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, verbose=True, patience=self.hparams["scheduler_patience"]
        )
        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        train_data = get_data(Path(self.hparams["data_dir"]) / TRAIN_NAME)
        self.hparams.add_dataset_size(len(train_data), TRAIN_NAME)
        return _get_fixed_dataloader(train_data, self.hparams["batch_size"])

    @pl.data_loader
    def val_dataloader(self):
        val_data = get_data(Path(self.hparams["data_dir"]) / VAL_NAME)
        self.hparams.add_dataset_size(len(val_data), VAL_NAME)
        return _get_fixed_dataloader(val_data, self.hparams["batch_size"])

    @pl.data_loader
    def test_dataloader(self):
        test_data = get_data(Path(self.hparams["data_dir"]) / TEST_NAME)
        self.hparams.add_dataset_size(len(test_data), VAL_NAME)
        loader = DataLoader(
            dataset=test_data,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            num_workers=12,
        )
        return loader

    @staticmethod
    def _calculate_accuracy(y_hat, y):
        # todo apply class weights here as well
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = labels_hat.eq(y).float().mean()
        return val_acc

    @staticmethod
    def _construct_lightning_log(
        log: dict, suffix: str = "train", prefix: str = "metrics"
    ):
        fixed_log = {
            f"{prefix}/" + metric: {suffix: value} for metric, value in log.items()
        }
        return {"log": fixed_log, **log}
