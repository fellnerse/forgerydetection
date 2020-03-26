import ast
import logging
from pathlib import Path

import click
import torch


VAL_ACC = "val_acc"

logger = logging.getLogger(__file__)


def get_latest_checkpoint(checkpoint_folder: Path) -> str:
    """Returns the latest checkpoint in given path.

    Raises FileNotFoundError if folder does not contain any .ckpt files."""

    checkpoints = sorted(checkpoint_folder.glob("*.ckpt"))
    if len(checkpoints) == 0:
        raise FileNotFoundError(
            f"Could not find any .ckpt files in {checkpoint_folder}"
        )
    latest_checkpoint = str(checkpoints[-1])
    logger.info(f"Using {latest_checkpoint} to load weights.")
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


NAN_TENSOR = torch.Tensor([float("NaN")])
