import argparse
from datetime import datetime

import numpy as np
import ray
from ray import tune
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.schedulers import ASHAScheduler

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
    sched = ASHAScheduler(metric="mean_accuracy")
    experiment_name = f"vgg_experiments/{datetime.now()}"
    analysis = tune.run(
        VGgTrainable,
        scheduler=sched,
        stop={
            "mean_accuracy": 0.95,
            "training_iteration": 5 if args.smoke_test else 20,
        },
        resources_per_trial={"cpu": 2, "gpu": int(args.use_gpu)},
        num_samples=5 if args.smoke_test else 20,
        checkpoint_at_end=True,
        checkpoint_freq=3,
        config={
            "args": args,
            "lr": tune.uniform(0.001, 0.1),
            "momentum": tune.uniform(0.1, 0.9),
            "epoch_size": 32,
            "test_size": 16,
            "batch_size": 4,
        },
        loggers=DEFAULT_LOGGERS,
        name=experiment_name,
        local_dir="./log",
    )

    print("Best config is:", analysis.get_best_config(metric="mean_accuracy"))


if __name__ == "__main__":
    main()
