import json
from pathlib import Path

__current_directory = Path(__file__).resolve().parent


def flatten(l: list):
    return {item for sublist in l for item in sublist}


TRAIN, TRAIN_NAME = (
    flatten(json.load((__current_directory / "train.json").open())),
    "train",
)
VAL, VAL_NAME = flatten(json.load((__current_directory / "val.json").open())), "val"
TEST, TEST_NAME = flatten(json.load((__current_directory / "test.json").open())), "test"
