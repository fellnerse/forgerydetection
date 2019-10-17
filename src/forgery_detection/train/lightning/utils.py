import click
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from forgery_detection.train.lightning.system import Supervised


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
@click.option("--lr", default=10e-5, help="Learning rate used by optimizer")
@click.option("--batch_size", default=128, help="Path to data to validate on")
@click.option(
    "--scheduler_patience", default=10, help="Patience of ReduceLROnPlateau scheduler"
)
@click.option("--no_gpu", is_flag=True)
@click.option(
    "--model",
    type=click.Choice(Supervised.MODEL_DICT.keys()),
    default="squeeze",
    help="Learning rate used by optimizer",
)
@click.option(
    "--data_percentage",
    default=1.0,
    help="How much of the data should be used for training, and validation."
    "Use 1.0 for 100%, 0.5 for 50% etc.",
)
@click.option(
    "--val_check_interval",
    default=1.0,
    help="Run validation step after this percentage of training data. 1.0 corresponds to"
    "running the validation after one complete epoch.",
)
def run_lightning(*args, **kwargs):

    model = Supervised(kwargs)

    log_dir = "/log"
    checkpoint_callback = ModelCheckpoint(
        filepath=log_dir + "/checkpoints",
        save_best_only=True,
        verbose=True,
        monitor="acc",
        mode="max",
        prefix="",
    )
    gpus = None if kwargs["no_gpu"] else 1

    trainer = Trainer(
        gpus=gpus,
        checkpoint_callback=checkpoint_callback,
        default_save_path=log_dir,
        train_percent_check=kwargs["data_percentage"],
        val_percent_check=kwargs["data_percentage"],
        val_check_interval=kwargs["val_check_interval"],
    )
    trainer.fit(model)
