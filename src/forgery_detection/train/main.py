from datetime import datetime

import click
import numpy as np
import ray
import torch
from ray import tune
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.schedulers import PopulationBasedTraining

from forgery_detection.data.face_forensics.utils import get_data_loaders
from forgery_detection.train.config import simple_vgg
from forgery_detection.train.utils import sample
from forgery_detection.train.utils import SimpleTrainable


@click.command()
@click.option(
    "--data_dir", default="~PycharmProjects/data_10", help="Path to data to train on"
)
def main(data_dir):
    np.random.seed(42)

    ray.init()

    # todo create some function
    train_spec = simple_vgg.copy()
    batch_size = train_spec["config"]["settings"]["batch_size"]

    train_loader, test_loader = get_data_loaders(
        batch_size=batch_size, data_dir=data_dir
    )  # todo thin about what actually needs ot be pinned
    # X_id = pin_in_object_store(np.random.random(size=100000000))
    train_spec["config"]["settings"]["train_loader_id"] = ray.put(train_loader)
    train_spec["config"]["settings"]["test_loader_id"] = ray.put(test_loader)

    try:
        use_cuda = (
            train_spec["config"]["settings"]["use_gpu"] and torch.cuda.is_available()
        )
    except KeyError:
        use_cuda = False

    train_spec["config"]["settings"]["device"] = torch.device(
        "cuda" if use_cuda else "cpu"
    )

    # sample one time from hyper parameter for starting values
    hyper_parameter = train_spec["config"]["hyper_parameter"].copy()
    train_spec["config"]["hyper_parameter"] = sample(hyper_parameter)

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="mean_accuracy",
        mode="max",
        perturbation_interval=2,
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
    # print("Best config is:", analysis.get_best_config(metric="mean_accuracy"))


if __name__ == "__main__":
    main()
