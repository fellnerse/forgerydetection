from argparse import Namespace
from copy import deepcopy
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import RandomSampler
from torchvision.datasets import DatasetFolder

from forgery_detection.lightning.confusion_matrix import plot_cm
from forgery_detection.lightning.confusion_matrix import plot_to_image


def get_logger_and_checkpoint_callback(log_dir, val_check_interval):
    """Sets up a logger and a checkpointer.

    The code is mostly copied from pl.trainer.py.
    """

    logger = TestTubeLogger(save_dir=log_dir, name="lightning_logs")
    ckpt_path = "{}/{}/version_{}/{}".format(
        log_dir, logger.experiment.name, logger.experiment.version, "checkpoints"
    )  # todo maybe this is not necessary
    checkpoint_callback = ModelCheckpoint(
        filepath=ckpt_path,
        save_best_only=True,
        monitor="roc_auc",
        mode="max",
        prefix="",
    )
    return checkpoint_callback, logger


def calculate_class_weights(dataset: DatasetFolder) -> Tuple[List[str], List[float]]:
    labels, counts = np.unique(dataset.targets, return_counts=True)
    counts = 1 / counts
    counts /= counts.sum()
    return list(map(lambda idx: dataset.classes[idx], labels)), counts


class DictHolder(dict):
    """This just makes sure that the pytorch_lightning syntax works."""

    def __init__(self, kwargs: Union[dict, Namespace]):

        # if loading from checkpoint hparams will be a namespace
        if isinstance(kwargs, Namespace):
            kwargs = kwargs.__dict__
        if "cli" not in kwargs:
            kwargs["cli"] = self._construct_cli_arguments_from_hparams(kwargs)

        super().__init__(**kwargs)
        self.__dict__: dict = self

    def add_dataset_size(self, nb_samples: int, name: str):
        # todo instead of adding to hparams log it
        self[f"{name}_batches"] = (nb_samples // self["batch_size"]) * self[
            "val_check_interval"
        ]
        self[f"{name}_samples"] = nb_samples

    def add_class_weights(self, labels, weights):
        print("Using class weights:")
        print(self._class_weights_to_string(labels, weights))
        self["class_weights"] = {value[0]: value[1] for value in zip(labels, weights)}

    @staticmethod
    def _construct_cli_arguments_from_hparams(hparams: dict):
        hparams_copy = deepcopy(hparams)
        hparams_copy.pop("train")
        balance_data = hparams_copy.pop("balance_data")
        cli_arguments = " ".join(
            [f"--{key}={value}" for key, value in hparams_copy.items()]
        )
        if balance_data:
            cli_arguments += " --balance_data"
        return cli_arguments

    @staticmethod
    def _class_weights_to_string(labels: np.array, class_weights: np.array) -> str:
        return "\n".join(
            map(
                lambda value: f"{value[0]}:\t{value[1]:.3g}", zip(labels, class_weights)
            )
        )


def _get_fixed_dataloader(dataset: Dataset, batch_size: int, num_workers=6):
    # look https://github.com/williamFalcon/pytorch-lightning/issues/434
    sampler = BatchSampler(
        RandomSampler(dataset), batch_size=batch_size, drop_last=False
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

    class _DataLoader(DataLoader):
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


def log_confusion_matrix(logger, global_step, target: torch.tensor, pred: torch.tensor):
    cm = confusion_matrix(target, pred)
    figure = plot_cm(cm, class_names=["fake", "real"])
    cm_image = plot_to_image(figure)
    plt.close()
    logger.experiment.add_image(
        "metrics/cm", cm_image, dataformats="HWC", global_step=global_step
    )


def log_roc_graph(logger, global_step, target: torch.tensor, pred: torch.tensor):
    fpr, tpr, _ = metrics.roc_curve(target, pred, pos_label=1)
    roc_auc = auc(fpr, tpr)
    figure = plt.figure(figsize=(8, 8))
    lw = 2
    plt.plot(
        fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic curve")
    plt.legend(loc="lower right")
    cm_image = plot_to_image(figure)
    plt.close()
    logger.experiment.add_image(
        "metrics/roc", cm_image, dataformats="HWC", global_step=global_step
    )
    return roc_auc
