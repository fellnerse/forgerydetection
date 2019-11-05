import click
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from forgery_detection.lightning.system import Supervised
from forgery_detection.lightning.utils import get_logger_and_checkpoint_callback
from forgery_detection.lightning.utils import parse_gpus
from forgery_detection.lightning.utils import PythonLiteralOption


@click.command()
@click.option(
    "--data_dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to data dir containing, at least train and val data.",
)
@click.option(
    "--log_dir",
    required=True,
    type=click.Path(exists=True),
    help="Folder used for logging.",
)
@click.option("--lr", default=10e-5, help="Learning rate used by optimizer")
@click.option("--batch_size", default=128, help="Path to data to validate on")
@click.option(
    "--scheduler_patience", default=10, help="Patience of ReduceLROnPlateau scheduler"
)
@click.option("--gpus", cls=PythonLiteralOption, default="[3]")
@click.option(
    "--model",
    type=click.Choice(Supervised.MODEL_DICT.keys()),
    default="resnet18",
    help="Learning rate used by optimizer",
)
# todo convert to absolute batches
@click.option(
    "--val_check_interval",
    default=1.0,
    help="Run validation step after this percentage of training data. 1.0 corresponds to"
    "running the validation after one complete epoch.",
)
@click.option("--balance_data", is_flag=True)
def run_lightning(*args, **kwargs):
    gpus = parse_gpus(kwargs)

    kwargs["train"] = True
    kwargs["gpus"] = gpus

    model = Supervised(kwargs)

    # Logging and Checkpoints
    checkpoint_callback, logger = get_logger_and_checkpoint_callback(
        kwargs["log_dir"], kwargs["val_check_interval"]
    )

    # early stopping
    early_stopping_callback = EarlyStopping(
        monitor="roc_auc", patience=3, verbose=True, mode="max"
    )

    trainer = Trainer(
        gpus=gpus,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stopping_callback,
        default_save_path=kwargs["log_dir"],
        val_percent_check=kwargs["val_check_interval"],
        val_check_interval=kwargs["val_check_interval"],
        distributed_backend="ddp" if gpus and len(gpus) > 1 else None,
    )
    trainer.fit(model)
