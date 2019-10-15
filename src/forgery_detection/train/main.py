from datetime import datetime
from types import LambdaType

import numpy as np
import ray
from ray import tune
from ray.tune import sample_from
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.schedulers import PopulationBasedTraining

from forgery_detection.data.face_forensics.utils import get_data_loaders
from forgery_detection.models.simple_vgg import VGgTrainable
from forgery_detection.train.config import simple_vgg


def sample(hyper_parameter: dict) -> dict:
    sampled_dict = {}
    for key, value in hyper_parameter.items():
        if isinstance(value, dict):
            sampled_dict[key] = sample(value)
        elif isinstance(value, LambdaType):
            sampled_dict[key] = sample_from(value)
        else:
            sampled_dict[key] = value
    return sampled_dict


def main():
    np.random.seed(42)

    ray.init()
    batch_size = 8  # todo take this from the config
    train_loader, test_loader = get_data_loaders(
        batch_size=batch_size
    )  # todo thin about what acutally needs ot be pinned
    # X_id = pin_in_object_store(np.random.random(size=100000000))

    train_spec = simple_vgg.copy()
    train_spec["config"]["settings"]["train_loader_id"] = ray.put(train_loader)
    train_spec["config"]["settings"]["test_loader_id"] = ray.put(test_loader)

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
    analysis = tune.run(  # todo put most of the stuff in config
        VGgTrainable,
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
