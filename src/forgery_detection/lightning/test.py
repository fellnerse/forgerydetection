from pathlib import Path

import click
from pytorch_lightning import Trainer

from forgery_detection.lightning.logging.utils import CHECKPOINTS
from forgery_detection.lightning.logging.utils import get_logger_and_checkpoint_callback
from forgery_detection.lightning.logging.utils import SystemMode
from forgery_detection.lightning.system import Supervised
from forgery_detection.lightning.utils import get_latest_checkpoint
from forgery_detection.lightning.utils import PythonLiteralOptionGPUs


@click.command()
@click.option(
    "--checkpoint_dir",
    required=True,
    type=click.Path(exists=True),
    help="Folder containing logs and checkpoint.",
)
@click.option(
    "--log_dir",
    required=True,
    type=click.Path(exists=True),
    help="Folder used for logging.",
    default="/log",
)
@click.option("--gpus", cls=PythonLiteralOptionGPUs, default="[3]")
@click.option("--debug", is_flag=True)
def run_lightning_test(*args, **kwargs):
    kwargs["mode"] = SystemMode.TEST
    kwargs["log_roc_values"] = True

    checkpoint_folder = Path(kwargs["checkpoint_dir"]) / CHECKPOINTS

    model: Supervised = Supervised.load_from_metrics(
        weights_path=get_latest_checkpoint(checkpoint_folder),
        tags_csv=Path(kwargs["checkpoint_dir"]) / "meta_tags.csv",
        overwrite_hparams=kwargs,
    )

    logger_info = model.hparams.get("logger", None)

    _, logger = get_logger_and_checkpoint_callback(
        kwargs["log_dir"], kwargs["mode"], kwargs["debug"], logger_info=logger_info
    )

    trainer = Trainer(
        gpus=kwargs["gpus"],
        logger=logger,
        default_save_path=kwargs["log_dir"],
        distributed_backend="ddp"
        if kwargs["gpus"] and len(kwargs["gpus"]) > 1
        else None,
        weights_summary=None,
    )
    trainer.test(model)
