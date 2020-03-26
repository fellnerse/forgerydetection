import numpy as np


def normalize_dict(filter_banks: dict):
    b = list(filter_banks.values())
    c = np.concatenate(b, axis=0)
    mean, std = c.mean(axis=0), c.std(axis=0)
    for key, value in filter_banks.items():
        filter_banks[key] = (value - mean) / std
