import click
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from forgery_detection.train.lightning.mnist import SupervisedSystem


@click.command()
@click.option(
    "--train_data_dir",
    default="~/PycharmProjects/data_10/train",
    help="Path to data to train on",
)
@click.option(
    "--val_data_dir",
    default="~/PycharmProjects/data_10/val",
    help="Path to data to validate on",
)
@click.option("--batch_size", default=128, help="Path to data to validate on")
def run_lightning(train_data_dir, val_data_dir, batch_size):

    model = SupervisedSystem(train_data_dir, val_data_dir, batch_size)

    # DEFAULTS used by the Trainer
    log_dir = "/log"
    checkpoint_callback = ModelCheckpoint(
        filepath=log_dir + "/checkpoints",
        save_best_only=True,
        verbose=True,
        monitor="val_acc",
        mode="min",
        prefix="",
    )

    trainer = Trainer(
        gpus=1,
        checkpoint_callback=checkpoint_callback,
        default_save_path=log_dir,
        train_percent_check=0.1,
    )
    trainer.fit(model)
