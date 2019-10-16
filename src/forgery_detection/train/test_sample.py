import numpy as np

from forgery_detection.train.utils import sample


def test_sampling():
    sampled_value = sample(
        {
            "lr": lambda: np.random.uniform(-1, 0),
            "weight_decay": lambda: np.random.uniform(0.1, 0.9),
        }
    )["lr"]
    assert sampled_value < 0
