from enum import auto
from enum import Enum

import torch


class AudioMode(str, Enum):
    EXACT = auto()
    DIFFERENT_VIDEO = auto()
    SAME_VIDEO_MIN_DISTANCE = auto()
    SAME_VIDEO_MAX_DISTANCE = auto()

    FAKE_NOISE = auto()
    FAKE_NOISE_DIFFERENT_VIDEO = auto()

    MANIPULATION_METHOD_DIFFERENT_VIDEO = auto()

    def __str__(self):
        return self.name


NAN_TENSOR = torch.Tensor([float("NaN")])
VAL_ACC = "val_acc"
CHECKPOINTS = "checkpoints"
RUNS = "runs"


class SystemMode(Enum):
    TRAIN = auto()
    TEST = auto()
    BENCHMARK = auto()

    def __str__(self):
        return self.name
