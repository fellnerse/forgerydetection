import click
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from forgery_detection.train.lightning.system import Supervised
from forgery_detection.train.lightning.utils import get_logger_and_checkpoint_callback


@click.command()
@click.option(
    "--train_data_dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to data to train on",
)
@click.option(
    "--val_data_dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to data to validate on",
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
@click.option("--no_gpu", is_flag=True)
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
@click.option("--val_batch_nb_multiplier", default=1.0, help="Do more batches.")
@click.option("--balance_data", is_flag=True)
def run_lightning(*args, **kwargs):

    model = Supervised(kwargs)

    # Logging and Checkpoints
    checkpoint_callback, logger = get_logger_and_checkpoint_callback(
        kwargs["log_dir"], kwargs["val_check_interval"]
    )

    # early stopping
    early_stopping_callback = EarlyStopping(
        monitor="acc", patience=3, verbose=True, mode="max"
    )

    gpus = None if kwargs["no_gpu"] else 1

    trainer = Trainer(
        gpus=gpus,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stopping_callback,
        default_save_path=kwargs["log_dir"],
        val_percent_check=kwargs["val_check_interval"]
        * kwargs["val_batch_nb_multiplier"],
        val_check_interval=kwargs["val_check_interval"],
    )
    trainer.fit(model)
