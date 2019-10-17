from datetime import datetime

import click
import numpy as np
import ray
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from ray import tune
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.schedulers import PopulationBasedTraining

from forgery_detection.train.config import simple_vgg
from forgery_detection.train.lightning.mnist import SupervisedSystem
from forgery_detection.train.utils import process_config
from forgery_detection.train.utils import SimpleTrainable


@click.command()
@click.option(
    "--data_dir", default="~/PycharmProjects/data_10", help="Path to data to train on"
)
def run_pbt(data_dir):
    np.random.seed(42)

    ray.init()

    # todo create some function
    train_spec = simple_vgg.copy()
    hyper_parameter = process_config(data_dir, train_spec)

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="mean_accuracy",
        mode="max",
        perturbation_interval=10,
        hyperparam_mutations={"hyper_parameter": hyper_parameter},
    )
    experiment_name = f"vgg_experiments/{datetime.now()}"
    analysis = tune.run(
        SimpleTrainable,
        scheduler=pbt,
        reuse_actors=True,
        verbose=True,
        loggers=DEFAULT_LOGGERS,
        name=experiment_name,
        local_dir="../log",
        **train_spec,
    )
    print(analysis.trials)


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
        filepath=log_dir,
        save_best_only=True,
        verbose=True,
        monitor="val_loss",
        mode="min",
        prefix="",
    )

    trainer = Trainer(
        gpus=1, checkpoint_callback=checkpoint_callback, default_save_path=log_dir
    )
    trainer.fit(model)


if __name__ == "__main__":
    # run_pbt()
    run_lightning()
