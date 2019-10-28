import pytorch_lightning as pl
import torch
from sklearn.metrics import confusion_matrix
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from forgery_detection.data.utils import get_data
from forgery_detection.data.utils import ImbalancedDatasetSampler
from forgery_detection.models.binary_classification import Resnet18Binary
from forgery_detection.models.binary_classification import SqueezeBinary
from forgery_detection.models.binary_classification import VGG11Binary
from forgery_detection.train.lightning.utils import plot_confusion_matrix
from forgery_detection.train.lightning.utils import plot_to_image


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

        self.hparams.add_data_information_to_hparams(
            len(self.train_data_loader), len(self.val_data_loader)
        )

    def forward(self, x):
        return self.model.forward(x)

    def loss(self, logits, labels):
        cross_engropy = F.cross_entropy(logits, labels)
        return cross_engropy

    def training_step(self, batch, batch_nb):
        x, target = batch
        pred = self.forward(x)

        loss_val = self.loss(pred, target)
        train_acc = self._calculate_accuracy(pred, target)

        log = {"loss": loss_val, "acc": train_acc}

        return self._construct_lightning_log(log, train=True)

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

        return self._construct_lightning_log(log, train=False)

    def on_epoch_end(self):
        # do confusion matrix stuff here
        pass

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

    def _log_confusion_matrix_image(self, pred, target):
        cm_image = self._generate_confusion_matrix_image(target, pred)
        self.logger.experiment.add_image(
            "confusion matrix",
            cm_image,
            dataformats="HWC",
            global_step=self.global_step,
        )

    def _generate_confusion_matrix_image(self, pred, target):
        cm = confusion_matrix(pred.cpu(), torch.argmax(target, dim=1).cpu())
        figure = plot_confusion_matrix(cm, class_names=["fake", "real"])
        cm_image = plot_to_image(figure)
        return cm_image

    def _get_dataloader(self, dataset: Dataset, num_workers=6):
        if self.hparams["balance_data"]:
            sampler = ImbalancedDatasetSampler(dataset)
        else:
            sampler = None

        return DataLoader(
            dataset,
            batch_size=self.hparams["batch_size"],
            shuffle=sampler is None,
            num_workers=num_workers,
            sampler=sampler,
        )

    @staticmethod
    def _calculate_accuracy(y_hat, y):
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = labels_hat.eq(y).float().mean()
        return val_acc

    @staticmethod
    def _construct_lightning_log(log: dict, train=True, prefix: str = "metrics"):
        suffix = "train" if train else "val"
        fixed_log = {
            f"{prefix}/" + metric: {suffix: value} for metric, value in log.items()
        }
        return {"log": fixed_log, **log}

    class _DictHolder(dict):
        """This just makes sure that the pytorch_lightning syntax works."""

        def __init__(self, hparams: dict):
            hparams["cli"] = self._construct_cli_arguments_from_hparams(hparams)
            super().__init__(**hparams)
            self.__dict__: dict = self

        @staticmethod
        def _construct_cli_arguments_from_hparams(hparams: dict):
            cli_arguments = " ".join(
                [f"--{key}={value}" for key, value in hparams.items()]
            )
            return cli_arguments

        def add_data_information_to_hparams(
            self, train_nb_batches: int, val_nb_batches: int
        ):
            self["val_after_n_train_batches"] = (
                train_nb_batches * self["val_check_interval"]
            )
            self["val_batches"] = val_nb_batches * self["val_check_interval"]
            self["train_samples"] = train_nb_batches * self["batch_size"]
            self["val_samples"] = val_nb_batches * self["batch_size"]
