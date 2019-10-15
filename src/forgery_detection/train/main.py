import argparse
import random
from datetime import datetime

import numpy as np
import ray
from ray import tune
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.schedulers import PopulationBasedTraining

from forgery_detection.data.face_forensics.utils import get_data_loaders
from forgery_detection.models.simple_vgg import VGgTrainable

# Training settings
parser = argparse.ArgumentParser(description="Small vgg on faceforensic dataset")
parser.add_argument(
    "--use-gpu", action="store_true", default=False, help="enables CUDA training"
)
parser.add_argument(
    "--smoke-test", action="store_true", help="Finish quickly for testing"
)


def main():
    np.random.seed(42)

    args = parser.parse_args()
    ray.init()
    batch_size = 8
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)
    # X_id = pin_in_object_store(np.random.random(size=100000000))

    train_spec = {
        "config": {
            "args": args,
            "epoch_size": 2,
            "test_size": 1,
            "batch_size": batch_size,
            "train_loader_id": ray.put(train_loader),
            "test_loader_id": ray.put(test_loader),
            "lr": tune.uniform(0.001, 0.1),  # this is just for initializing the trials
            "decay": tune.uniform(0.1, 0.9),
        }
    }

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="mean_accuracy",
        mode="max",
        perturbation_interval=2,
        hyperparam_mutations={
            "lr": lambda: random.uniform(0.001, 0.1),
            "decay": lambda: random.uniform(0.1, 0.9),
        },
    )
    experiment_name = f"vgg_experiments/{datetime.now()}"
    analysis = tune.run(
        VGgTrainable,
        scheduler=pbt,
        reuse_actors=True,
        verbose=True,
        stop={
            "mean_accuracy": 1.1,  # 0.95,
            "training_iteration": 20 if args.smoke_test else 20,
        },
        resources_per_trial={"cpu": 2, "gpu": int(args.use_gpu)},
        num_samples=4 if args.smoke_test else 20,
        checkpoint_at_end=True,
        checkpoint_freq=5,
        loggers=DEFAULT_LOGGERS,
        name=experiment_name,
        local_dir="./log",
        **train_spec,
    )
    print(analysis.trials)
    # print("Best config is:", analysis.get_best_config(metric="mean_accuracy"))


if __name__ == "__main__":
    main()
