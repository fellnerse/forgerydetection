import logging

import click
import numpy as np
import torch
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler

from forgery_detection.data.misc.evaluate_outputs_binary import (
    print_evaluation_for_test_folder,
)
from forgery_detection.lightning.logging.const import AudioMode
from forgery_detection.lightning.logging.utils import get_logger_dir
from forgery_detection.lightning.logging.utils import PythonLiteralOptionGPUs
from forgery_detection.lightning.utils import get_model_and_trainer

logger = logging.getLogger(__file__)


def print_google_sheet_ready_output(_logger):
    logger_dir = get_logger_dir(_logger)

    print_evaluation_for_test_folder(logger_dir)


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
@click.option("--randomize_sampling", is_flag=True)
@click.option("--set_default_file_list", is_flag=True)
@click.option("--gpus", cls=PythonLiteralOptionGPUs, default="[0]")
@click.option("--debug", is_flag=True)
def run_train_val_evaluation(
    train_percent_check,
    val_percent_check,
    audio_file,
    set_default_file_list,
    randomize_sampling,
    **kwargs,
):

    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if audio_file:
        kwargs["audio_file"] = audio_file
    kwargs["audio_mode"] = AudioMode.EXACT.name
    # kwargs["crop_faces"] = False
    kwargs["optimizer"] = "sgd"

    if set_default_file_list:
        kwargs[
            "data_dir"
        ] = "/home/sebastian/data/file_lists/c40/trf_-1_-1_full_size_relative_bb_8_sl.json"

    # train data
    model, trainer = get_model_and_trainer(
        test_percent_check=train_percent_check, **kwargs
    )
    _logger = model.logger

    model.test_dataloader = model.train_dataloader

    trainer.test(model)

    # val data
    kwargs["sampling_probs"] = None
    model, trainer = get_model_and_trainer(
        test_percent_check=val_percent_check, _logger=_logger, **kwargs
    )
    model.sampler_cls = SequentialSampler
    if randomize_sampling:
        model.sampler_cls = RandomSampler
    model.test_dataloader = model.val_dataloader

    trainer.test(model)

    print_google_sheet_ready_output(_logger)
    print(f"{kwargs['checkpoint_dir']} | ckpt_nr: {kwargs['checkpoint_nr']}")


if __name__ == "__main__":
    run_train_val_evaluation()
