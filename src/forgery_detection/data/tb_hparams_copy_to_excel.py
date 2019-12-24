# flake8: noqa
#%%
from typing import List

import numpy as np


def extract_version(path: str):
    return path.split("/")[1]


def process_numbers(numbers: List[str]):
    return list(map(lambda x: np.format_float_positional(float(x)), numbers))


header = ["name", "lr", "weight_decay", "acc", "loss"]
experiments = []
print(__file__)
with open("./tb_hparams.txt") as f:
    text = f.readlines()

for i in range(len(text) // len(header)):
    start_idx = i * len(header)
    version = extract_version(text[start_idx])
    numbers = process_numbers(text[start_idx + 1 : start_idx + 5])
    experiments.append("\t".join([version] + numbers))
print("\t".join(header))
print("\n".join(experiments))
