import random

from torch.nn import functional as F
from torch.optim import Adam

from forgery_detection.models.simple_vgg import VGG11Binary

simple_vgg = {
    "config": {
        "settings": {
            "use_gpu": False,
            "epoch_size": 2,
            "test_size": 1,
            "batch_size": 8,
        },
        "model": VGG11Binary,
        "optimizer": Adam,
        "loss": F.nll_loss,
        "hyper_parameter": {  # todo put this in config as well
            "optimizer": {
                "lr": lambda: random.uniform(
                    0.001, 0.1
                ),  # this is just for initializing the trials
                "weight_decay": lambda: random.uniform(0.1, 0.9),
            }
        },
    },
    "stop": {"mean_accuracy": 1.1, "training_iteration": 40},
    "resources_per_trial": {"cpu": 2, "gpu": 0},
    "num_samples": 4,
    "checkpoint_freq": 5,
    "keep_checkpoints_num": 5,
}
