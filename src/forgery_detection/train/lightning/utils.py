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
    "--scheduler_patience", default=2, help="Patience of ReduceLROnPlateau scheduler"
)
@click.option("--no_gpu", is_flag=True)
@click.option(
    "--model",
    type=click.Choice(Supervised.MODEL_DICT.keys()),
    default="squeeze",
    help="Learning rate used by optimizer",
)
def run_lightning(*args, **kwargs):

    model = Supervised(kwargs)

    log_dir = "/log"
    checkpoint_callback = ModelCheckpoint(
        filepath=log_dir + "/checkpoints_testing2",
        save_best_only=True,
        verbose=True,
        monitor="val_acc",
        mode="max",
        prefix="",
    )
    gpus = 0 if kwargs["no_gpu"] else 1

    trainer = Trainer(
        gpus=gpus, checkpoint_callback=checkpoint_callback, default_save_path=log_dir
    )
    trainer.fit(model)
