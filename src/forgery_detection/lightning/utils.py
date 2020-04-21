from pathlib import Path

from pytorch_lightning import Trainer

from forgery_detection.lightning.logging.const import SystemMode
from forgery_detection.lightning.logging.utils import (
    backwards_compatible_get_checkpoint,
)
from forgery_detection.lightning.logging.utils import get_logger_and_checkpoint_callback
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

        _, _logger = get_logger_and_checkpoint_callback(
            kwargs["log_dir"], kwargs["mode"], kwargs["debug"], logger_info=logger_info
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
