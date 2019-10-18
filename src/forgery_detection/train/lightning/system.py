import pytorch_lightning as pl
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from forgery_detection.data.face_forensics.utils import get_data
from forgery_detection.models.binary_classification import SqueezeBinary
from forgery_detection.models.binary_classification import VGG11Binary


class Supervised(pl.LightningModule):
    MODEL_DICT = {"squeeze": SqueezeBinary, "vgg11": VGG11Binary}

    def __init__(self, hparams: dict):
        super(Supervised, self).__init__()
        self.hparams = self._DictHolder(hparams)
        self.batch_size = hparams["batch_size"]
        self.train_data = get_data(hparams["train_data_dir"])
        self.val_data = get_data(hparams["val_data_dir"])

        self.model = self.MODEL_DICT[self.hparams["model"]]()

    def forward(self, x):
        return self.model.forward(x)

    def loss(self, logits, labels):
        cross_engropy = F.cross_entropy(logits, labels)
        return cross_engropy

    def _make_lightning_log(self, log: dict, prefix: str = None):
        if prefix:
            prefixed_log = {prefix + "/" + key: value for key, value in log.items()}
        else:
            prefixed_log = log
        return {"log": prefixed_log, **log}

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)

        loss_val = self.loss(y_hat, y)
        train_acc = self._calculate_accuracy(y_hat, y)

        log = {"loss": loss_val, "acc": train_acc}

        return self._make_lightning_log(log, prefix="train")

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)

        loss_val = self.loss(y_hat, y)
        val_acc = self._calculate_accuracy(y_hat, y)

        return {"loss": loss_val, "acc": val_acc}

    def validation_end(self, outputs):
        # aggregate values from validation step
        val_loss_mean = torch.stack([x["loss"] for x in outputs]).mean()
        val_acc_mean = torch.stack([x["acc"] for x in outputs]).mean()

        log = {"loss": val_loss_mean, "acc": val_acc_mean}

        return self._make_lightning_log(log, prefix="val")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams["lr"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, verbose=True, patience=self.hparams["scheduler_patience"]
        )
        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(
            self.train_data,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            num_workers=8,
        )

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(
            self.val_data,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            num_workers=8,
        )

    @staticmethod
    def _calculate_accuracy(y_hat, y):
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)
        return val_acc

    class _DictHolder:
        def __init__(self, hparams: dict):
            self.__dict__ = hparams

        def __getitem__(self, item):
            return self.__dict__[item]
