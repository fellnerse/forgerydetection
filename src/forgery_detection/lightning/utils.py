from pathlib import Path

import click
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger

from forgery_detection.lightning.logging.const import CHECKPOINTS
from forgery_detection.lightning.logging.const import SystemMode
from forgery_detection.lightning.logging.const import VAL_ACC
from forgery_detection.lightning.logging.utils import (
    backwards_compatible_get_checkpoint,
)
from forgery_detection.lightning.logging.utils import get_logger_dir
from forgery_detection.lightning.logging.utils import OldModelCheckpoint
from forgery_detection.lightning.system import Supervised


def _get_logger_info(kwargs: dict):
    if "logger" in kwargs and kwargs["logger"]:
        return kwargs["logger"]

    if "logger/name" in kwargs:
        return {
            "name": kwargs["logger/name"],
            "description": kwargs["logger/description"],
        }
    return None


def get_model_and_trainer(_logger=None, test_percent_check=1.0, **kwargs):
    kwargs["mode"] = SystemMode.TEST
    checkpoint_folder = Path(kwargs["checkpoint_dir"])
    model: Supervised = Supervised.load_from_metrics(
        weights_path=backwards_compatible_get_checkpoint(
            checkpoint_folder, kwargs["checkpoint_nr"]
        ),
        tags_csv=Path(kwargs["checkpoint_dir"]) / "meta_tags.csv",
        overwrite_hparams=kwargs,
    )
    if _logger is None:
        logger_info = _get_logger_info(model.hparams)
        kwargs["log_dir"] = kwargs["checkpoint_dir"]
        _, _logger = get_logger_and_checkpoint_callback_for_test(
            kwargs["log_dir"], kwargs["debug"], logger_info=logger_info
        )
    model.logger = _logger
    trainer = Trainer(
        gpus=kwargs["gpus"],
        logger=_logger,
        default_save_path=kwargs["log_dir"],
        distributed_backend="ddp"
        if kwargs["gpus"] and len(kwargs["gpus"]) > 1
        else None,
        weights_summary=None,
        test_percent_check=test_percent_check,
    )
    return model, trainer


def get_logger_and_checkpoint_callback_for_test(log_dir, debug, logger_info=None):
    """Sets up a logger and a checkpointer.

    The code is mostly copied from pl.trainer.py.
    """
    if debug:
        name = "debug"
        description = ""
    else:
        if logger_info:
            name = "test"
            description = logger_info["description"]
        else:
            # if the user provides a name create its own folder in the default folder
            name = click.prompt("Name of run", type=str, default="default").replace(
                " ", "_"
            )
            description = click.prompt("Description of run", type=str, default="")

    logger = TestTubeLogger(save_dir=log_dir, name=name, description=description)
    logger_dir = get_logger_dir(logger)

    checkpoint_callback = OldModelCheckpoint(
        filepath=logger_dir / CHECKPOINTS,
        save_best_only=False,
        monitor=VAL_ACC,
        mode="max",
        prefix="",
    )
    return checkpoint_callback, logger
