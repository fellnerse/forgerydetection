import pytorch_lightning as pl
import torch
from sklearn.metrics import confusion_matrix
from torch import optim
from torch.nn import functional as F
from torch.utils.data import BatchSampler
from torch.utils.data import RandomSampler
from torch.utils.data.dataset import Dataset

from forgery_detection.data.utils import get_data
from forgery_detection.models.binary_classification import Resnet18Binary
from forgery_detection.models.binary_classification import SqueezeBinary
from forgery_detection.models.binary_classification import VGG11Binary
from forgery_detection.train.lightning.utils import calculate_class_weights
from forgery_detection.train.lightning.utils import class_weights_to_string
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

        self.train_data = get_data(hparams["train_data_dir"])
        self.val_data = get_data(hparams["val_data_dir"])

        self.model = self.MODEL_DICT[self.hparams["model"]]()

        self.hparams.add_data_information_to_hparams(
            len(self.train_data), len(self.val_data)
        )

        if self.hparams["balance_data"]:
            labels, weights = calculate_class_weights(self.val_data)
            self.hparams.add_class_weights(labels, weights)
            self.class_weights = torch.tensor(weights, dtype=torch.float)
        else:
            self.class_weights = None

    def forward(self, x):
        return self.model.forward(x)

    def loss(self, logits, labels):
        cross_engropy = F.cross_entropy(logits, labels, weight=self.class_weights)
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
        return self._get_dataloader(self.train_data)

    @pl.data_loader
    def val_dataloader(self):
        return self._get_dataloader(self.val_data)

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

        sampler = BatchSampler(
            RandomSampler(dataset),
            batch_size=self.hparams["batch_size"],
            drop_last=False,
        )

        class _RepeatSampler(torch.utils.data.Sampler):
            """ Sampler that repeats forever.

            Args:
                sampler (Sampler)
            """

            def __init__(self, sampler):
                super().__init__(sampler)
                self.sampler = sampler

            def __iter__(self):
                while True:
                    yield from iter(self.sampler)

            def __len__(self):
                return len(self.sampler)

        class _DataLoader(torch.utils.data.dataloader.DataLoader):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.iterator = super().__iter__()

            def __len__(self):
                return len(self.batch_sampler.sampler)

            def __iter__(self):
                for i in range(len(self)):
                    yield next(self.iterator)

        loader = _DataLoader(
            dataset=dataset,
            shuffle=False,
            batch_sampler=_RepeatSampler(sampler),
            num_workers=num_workers,
        )

        return loader

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

        def add_data_information_to_hparams(
            self, nb_train_samples: int, nb_val_samples: int
        ):
            self["train_batches"] = (nb_train_samples // self["batch_size"]) * self[
                "val_check_interval"
            ]
            self["val_batches"] = (nb_val_samples // self["batch_size"]) * self[
                "val_check_interval"
            ]
            self["train_samples"] = nb_train_samples
            self["val_samples"] = nb_val_samples

        def add_class_weights(self, labels, weights):
            print("Using class weights:")
            print(class_weights_to_string(labels, weights))
            self["class_weights"] = {
                value[0]: value[1] for value in zip(labels, weights)
            }

        @staticmethod
        def _construct_cli_arguments_from_hparams(hparams: dict):
            cli_arguments = " ".join(
                [f"--{key}={value}" for key, value in hparams.items()]
            )
            return cli_arguments
