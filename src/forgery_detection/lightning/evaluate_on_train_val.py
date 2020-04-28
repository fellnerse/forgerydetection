import logging

import click
import numpy as np
import torch

from forgery_detection.lightning.logging.const import AudioMode
from forgery_detection.lightning.logging.utils import PythonLiteralOptionGPUs
from forgery_detection.lightning.utils import get_model_and_trainer

logger = logging.getLogger(__file__)


@click.command()
@click.option(
    "--checkpoint_dir",
    required=True,
    type=click.Path(exists=True),
    help="Folder containing logs and checkpoint.",
)
@click.option("--checkpoint_nr", type=int, default=-1)
@click.option("--audio_file", required=False, type=click.Path(exists=True))
@click.option(
    "--log_dir",
    required=True,
    type=click.Path(exists=True),
    help="Folder used for logging.",
    default="/mnt/raid/sebastian/log",
)
@click.option("--train_percent_check", "-tp", type=float, default=0.2)
@click.option("--val_percent_check", "-vp", type=float, default=1.0)
@click.option("--gpus", cls=PythonLiteralOptionGPUs, default="[0]")
@click.option("--debug", is_flag=True)
def run_train_val_evaluation(
    train_percent_check, val_percent_check, audio_file, **kwargs
):

    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if audio_file:
        kwargs["audio_file"] = audio_file
    kwargs["audio_mode"] = AudioMode.EXACT.name
    kwargs["crop_faces"] = False
    kwargs["sampling_probs"] = "1. 1. 1. 1. 4."

    # train data
    model, trainer = get_model_and_trainer(
        test_percent_check=train_percent_check, **kwargs
    )
    _logger = model.logger

    model.test_dataloader = model.train_dataloader

    trainer.test(model)

    # val data
    model, trainer = get_model_and_trainer(
        test_percent_check=val_percent_check, _logger=_logger, **kwargs
    )
    model.test_dataloader = model.val_dataloader

    trainer.test(model)


if __name__ == "__main__":
    run_train_val_evaluation()
