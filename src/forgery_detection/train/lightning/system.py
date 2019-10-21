import pytorch_lightning as pl
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from forgery_detection.data.utils import get_data
from forgery_detection.data.utils import ImbalancedDatasetSampler
from forgery_detection.models.binary_classification import Resnet18Binary
from forgery_detection.models.binary_classification import SqueezeBinary
from forgery_detection.models.binary_classification import VGG11Binary


class Supervised(pl.LightningModule):
    MODEL_DICT = {
        "squeeze": SqueezeBinary,
        "vgg11": VGG11Binary,
        "resnet18": Resnet18Binary,
    }

    def __init__(self, hparams: dict):
        super(Supervised, self).__init__()

        self.hparams = self._DictHolder(hparams)

        self.train_data_loader = self._get_dataloader(
            get_data(hparams["train_data_dir"])
        )
        self.val_data_loader = self._get_dataloader(get_data(hparams["val_data_dir"]))

        self.model = self.MODEL_DICT[self.hparams["model"]]()

        self._add_information_to_hparams()

    def forward(self, x):
        return self.model.forward(x)

    def loss(self, logits, labels):
        cross_engropy = F.cross_entropy(logits, labels)
        return cross_engropy

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)

        loss_val = self.loss(y_hat, y)
        train_acc = self._calculate_accuracy(y_hat, y)

        log = {"loss": loss_val, "acc": train_acc}

        return self._construct_lightning_log(log, train=True)

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

        return self._construct_lightning_log(log, train=False)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams["lr"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, verbose=True, patience=self.hparams["scheduler_patience"]
        )
        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        return self.train_data_loader

    @pl.data_loader
    def val_dataloader(self):
        return self.val_data_loader

    def _get_dataloader(self, dataset: Dataset):
        if self.hparams["balance_data"]:
            sampler = ImbalancedDatasetSampler(dataset)
        else:
            sampler = None

        return DataLoader(
            dataset,
            batch_size=self.hparams["batch_size"],
            shuffle=not sampler,
            num_workers=8,
            sampler=sampler,
        )

    @staticmethod
    def _calculate_accuracy(y_hat, y):
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)
        return val_acc

    @staticmethod
    def _construct_lightning_log(log: dict, train=True, prefix: str = "metrics"):
        suffix = "train" if train else "val"
        fixed_log = {
            f"{prefix}/" + metric: {suffix: value} for metric, value in log.items()
        }
        return {"log": fixed_log, **log}

    def _add_information_to_hparams(self):
        self.hparams["val_after_n_train_batches"] = (
            len(self.train_data_loader)
        ) * self.hparams["val_check_interval"]
        self.hparams["val_batches"] = (
            (len(self.val_data_loader))
            * self.hparams["val_check_interval"]
            * self.hparams["val_batch_nb_multiplier"]
        )
        self.hparams["train_samples"] = (
            len(self.train_data_loader) * self.hparams["batch_size"]
        )
        self.hparams["val_samples"] = (
            len(self.val_data_loader) * self.hparams["batch_size"]
        )

    class _DictHolder:
        """This just makes sure that the pytorch_lightning syntax works."""

        def __init__(self, hparams: dict):
            self.__dict__ = hparams

        def __getitem__(self, item):
            return self.__dict__[item]

        def __setitem__(self, key, value):
            self.__dict__[key] = value
