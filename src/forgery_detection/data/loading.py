from typing import List
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import RandomSampler
from torch.utils.data import WeightedRandomSampler

from forgery_detection.data.set import FileListDataset


def calculate_class_weights(dataset: FileListDataset) -> Tuple[List[str], List[float]]:
    labels, counts = np.unique(dataset.targets, return_counts=True)
    counts = 1 / counts
    counts /= counts.sum()
    return list(map(lambda idx: dataset.classes[idx], labels)), counts


def get_fixed_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers=6,
    sampler=RandomSampler,
    worker_init_fn=None,
):
    # look https://github.com/williamFalcon/pytorch-lightning/issues/434
    sampler = BatchSampler(sampler(dataset), batch_size=batch_size, drop_last=False)

    class _RepeatSampler(torch.utils.data.Sampler):
        """ Sampler that repeats forever.

        Args:
            sampler (Sampler)
        """

        def __init__(self, sampler):
            super().__init__(sampler)
            self.sampler = sampler

        def __iter__(self):
            while True:
                yield from iter(self.sampler)

        def __len__(self):
            return len(self.sampler)

    class _DataLoader(DataLoader):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.iterator = super().__iter__()

        def __len__(self):
            return len(self.batch_sampler.sampler)

        def __iter__(self):
            for i in range(len(self)):
                yield next(self.iterator)

    loader = _DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_sampler=_RepeatSampler(sampler),
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )

    return loader


class FiftyFiftySampler(WeightedRandomSampler):
    def __init__(self, dataset: FileListDataset, replacement=True):

        targets = np.array(dataset.targets, dtype=np.int)[dataset.samples_idx]
        _, class_weights = calculate_class_weights(dataset)
        weights = class_weights[targets]

        super().__init__(weights, num_samples=len(dataset), replacement=replacement)
