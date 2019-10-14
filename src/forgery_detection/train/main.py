import argparse

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
    analysis = tune.run(
        VGgTrainable,
        scheduler=sched,
        stop={
            "mean_accuracy": 0.95,
            "training_iteration": 3 if args.smoke_test else 20,
        },
        resources_per_trial={"cpu": 3, "gpu": int(args.use_gpu)},
        num_samples=1 if args.smoke_test else 20,
        checkpoint_at_end=True,
        checkpoint_freq=3,
        config={
            "args": args,
            "lr": tune.uniform(0.001, 0.1),
            "momentum": tune.uniform(0.1, 0.9),
            "epoch_size": 16,
            "test_size": 8,
            "batch_size": 4,
        },
        loggers=DEFAULT_LOGGERS,
        name="vgg_experiments",
        local_dir="./log",
    )

    print(analysis)

    print("Best config is:", analysis.get_best_config(metric="mean_accuracy"))


if __name__ == "__main__":
    main()
