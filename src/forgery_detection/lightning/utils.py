import ast
from argparse import Namespace
from copy import deepcopy
from enum import auto
from enum import Enum
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import click
import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import RandomSampler
from torch.utils.data import WeightedRandomSampler
from torchvision.datasets import DatasetFolder
from torchvision.datasets import ImageFolder

from forgery_detection.data.face_forensics.splits import TEST_NAME
from forgery_detection.data.utils import get_data
from forgery_detection.lightning.confusion_matrix import plot_cm
from forgery_detection.lightning.confusion_matrix import plot_to_image

CHECKPOINTS = "checkpoints"
RUNS = "runs"


class SystemMode(Enum):
    TRAIN = auto()
    TEST = auto()
    BENCHMARK = auto()

    def __str__(self):
        return self.name


def get_logger_and_checkpoint_callback(
    log_dir, mode: SystemMode, debug, logger_info=None
):
    """Sets up a logger and a checkpointer.

    The code is mostly copied from pl.trainer.py.
    """
    if debug:
        name = "debug"
        description = ""
    else:
        log_dir = str(Path(log_dir) / RUNS / str(mode))
        if logger_info:
            name = logger_info["name"]
            description = logger_info["description"]
        else:
            # if the user provides a name create its own folder in the default folder
            name = click.prompt("Name of run", type=str, default="default").replace(
                " ", "_"
            )
            description = click.prompt("Description of run", type=str, default="")

    logger = TestTubeLogger(save_dir=log_dir, name=name, description=description)
    logger_dir = get_logger_dir(logger)

    checkpoint_callback = ModelCheckpoint(
        filepath=logger_dir / CHECKPOINTS,
        save_best_only=True,
        monitor="roc_auc",
        mode="max",
        prefix="",
    )
    return checkpoint_callback, logger


def get_logger_dir(logger):
    return (
        Path(logger.save_dir)
        / logger.experiment.name
        / f"version_{logger.experiment.version}"
    )


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
        self[f"{name}_batches"] = (nb_samples // self["batch_size"]) * self[
            "val_check_interval"
        ]
        self[f"{name}_samples"] = nb_samples

    def add_class_weights(self, labels, weights):
        print("Using class weights:")
        print(self._class_weights_to_string(labels, weights))
        self["class_weights"] = {value[0]: value[1] for value in zip(labels, weights)}

    def add_nb_trainable_params(self, model: nn.Module):
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Trainable params: " f"{params}")
        self["nb_trainable_params"] = params

    @staticmethod
    def _construct_cli_arguments_from_hparams(hparams: dict):
        hparams_copy = deepcopy(hparams)
        hparams_copy.pop("mode")

        cli_arguments = ""
        for key, value in hparams_copy.items():
            if isinstance(value, bool):
                if value:
                    cli_arguments += f" --{key}"
            else:
                cli_arguments += f" --{key}={value}"

        return cli_arguments

    @staticmethod
    def _class_weights_to_string(labels: np.array, class_weights: np.array) -> str:
        return "\n".join(
            map(
                lambda value: f"{value[0]}:\t{value[1]:.3g}", zip(labels, class_weights)
            )
        )


def get_fixed_dataloader(
    dataset: Dataset, batch_size: int, num_workers=6, sampler=RandomSampler
):
    # look https://github.com/williamFalcon/pytorch-lightning/issues/434
    sampler = BatchSampler(sampler(dataset), batch_size=batch_size, drop_last=False)

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


def log_confusion_matrix(
    logger, global_step, target: torch.tensor, pred: torch.tensor, class_to_idx
) -> Dict[str, np.float]:
    cm = confusion_matrix(target, pred, labels=list(class_to_idx.values()))
    figure = plot_cm(cm, class_names=class_to_idx.keys())
    cm_image = plot_to_image(figure)
    plt.close()
    logger.experiment.add_image(
        "metrics/cm", cm_image, dataformats="HWC", global_step=global_step
    )

    # use cm to calculate class accuracies
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    class_accuracies_dict = {}
    for key, value in class_to_idx.items():
        class_accuracies_dict[key] = class_accuracies[value]
    return class_accuracies_dict


def log_roc_graph(
    logger, global_step, target: torch.tensor, pred: torch.tensor, pos_label
) -> float:
    fpr, tpr, thresholds = metrics.roc_curve(target, pred, pos_label=pos_label)
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
    plt.title(f"Receiver operating characteristic curve for label {pos_label}")
    plt.legend(loc="lower right")

    ax2 = plt.gca().twinx()
    ax2.plot(fpr, thresholds, markeredgecolor="r", linestyle="dashed", color="r")
    ax2.set_ylabel("Threshold", color="r")
    ax2.set_ylim([thresholds[-1], thresholds[0]])
    try:
        ax2.set_xlim([fpr[0], fpr[-1]])
    except ValueError:
        del ax2

    cm_image = plot_to_image(figure)
    plt.close()
    logger.experiment.add_image(
        "metrics/roc", cm_image, dataformats="HWC", global_step=global_step
    )
    return roc_auc


def get_latest_checkpoint(checkpoint_folder: Path) -> str:
    """Returns the latest checkpoint in given path.

    Raises FileNotFoundError if folder does not contain any .ckpt files."""

    checkpoints = sorted(checkpoint_folder.glob("*.ckpt"))
    if len(checkpoints) == 0:
        raise FileNotFoundError(
            f"Could not find any .ckpt files in {checkpoint_folder}"
        )
    latest_checkpoint = str(checkpoints[-1])
    print(f"Using {latest_checkpoint} to load weights.")
    return latest_checkpoint


class PythonLiteralOptionGPUs(click.Option):
    def type_cast_value(self, ctx, value):
        try:
            gpus = ast.literal_eval(value)
            if not isinstance(gpus, list):
                raise TypeError("gpus needs to be a list (i.e. [], [1], or [1,2].")
            gpus = 0 if len(gpus) == 0 else gpus
            return gpus
        except ValueError:
            raise click.BadParameter(value)


def get_labels_dict(data_dir: str) -> dict:
    dataset = get_data(Path(data_dir) / TEST_NAME)
    idx_to_class = {val: key for key, val in dataset.class_to_idx.items()}
    del dataset
    return idx_to_class


class FiftyFiftySampler(WeightedRandomSampler):
    def __init__(self, dataset: ImageFolder, replacement=True):

        targets = np.array(dataset.targets, dtype=np.int)
        _, class_weights = calculate_class_weights(dataset)
        weights = class_weights[targets]

        super().__init__(weights, num_samples=len(dataset), replacement=replacement)


def multiclass_roc_auc_score(y_target, y_pred, label_binarizer):
    y_target = label_binarizer.transform(y_target)
    y_pred = label_binarizer.transform(y_pred)
    return roc_auc_score(y_target, y_pred)


if __name__ == "__main__":
    data_set = get_data("/mnt/ssd1/sebastian/face_forensics_1000_c40_test/test")
    ffs = FiftyFiftySampler(data_set)
    counter = 0
    for i in ffs:
        if counter > 10:
            break
        print(i, data_set.targets[i])
        counter += 1
