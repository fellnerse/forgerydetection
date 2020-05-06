import logging

import click
import numpy as np
import torch

from forgery_detection.data.misc.evaluate_outputs_binary import calculate_metrics
from forgery_detection.data.misc.evaluate_outputs_binary import (
    get_output_file_names_ordered,
)
from forgery_detection.lightning.logging.const import AudioMode
from forgery_detection.lightning.logging.utils import get_logger_dir
from forgery_detection.lightning.logging.utils import PythonLiteralOptionGPUs
from forgery_detection.lightning.utils import get_model_and_trainer

logger = logging.getLogger(__file__)


def print_google_sheet_ready_output(_logger):
    logger_dir = get_logger_dir(_logger)

    output_files = get_output_file_names_ordered(logger_dir)
    train_acc, train_class_accs = calculate_metrics(output_files[0])
    val_acc, val_class_accs = calculate_metrics(output_files[1])

    print(
        f"{train_acc:.2%}/{val_acc:.2%};"
        + "".join("{:.2%};".format(x) for x in train_class_accs)
        + "".join("{:.2%};".format(x) for x in val_class_accs)
    )


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
    kwargs["optimizer"] = "sgd"

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

    print_google_sheet_ready_output(_logger)


if __name__ == "__main__":
    run_train_val_evaluation()
