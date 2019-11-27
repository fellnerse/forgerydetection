import ast
from pathlib import Path

import click
import torch

from forgery_detection.data.face_forensics.splits import TEST_NAME
from forgery_detection.data.utils import get_data

VAL_ACC = "val_acc"


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


NAN_TENSOR = torch.Tensor([float("NaN")])
