import numpy as np
from torch import nn
from torch.optim import Adam

from forgery_detection.models.simple_vgg import VGG11Binary

simple_vgg = {
    "config": {
        "settings": {
            "use_gpu": True,
            "epoch_size": 512,
            "test_size": 128,
            "batch_size": 16,
        },
        "model": VGG11Binary,
        "optimizer": Adam,
        "loss": nn.CrossEntropyLoss,
        "hyper_parameter": {
            "optimizer": {
                "lr": lambda: np.random.uniform(
                    10e-8, 10e-3
                ),  # this is just for initializing the trials
                "weight_decay": lambda: np.random.uniform(0.1, 0.9),
            }
        },
    },
    "stop": {"mean_accuracy": 1.1, "training_iteration": 400},
    "resources_per_trial": {"cpu": 8, "gpu": 1},
    "num_samples": 4,
    "checkpoint_freq": 5,
    "keep_checkpoints_num": 5,
}
