from __future__ import annotations

import logging
import random
from typing import Dict
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import WeightedRandomSampler
from torch.utils.data._utils.collate import default_collate

from forgery_detection.lightning.logging.const import AudioMode

logger = logging.getLogger(__file__)

if TYPE_CHECKING:
    from forgery_detection.data.set import FileListDataset


def calculate_class_weights(
    dataset: FileListDataset, predefined_weights=None
) -> Dict[str, float]:
    labels, counts = np.unique(
        np.array(dataset.targets, dtype=np.int)[dataset.samples_idx], return_counts=True
    )
    if predefined_weights is not None:
        counts = predefined_weights / counts
    else:
        counts = 1 / counts
    counts /= counts.sum()

    weight_dict = {class_idx: 0 for class_idx in dataset.class_to_idx.values()}

    for label, count in zip(labels, counts):
        weight_dict[label] = count

    return weight_dict


def get_sequence_collate_fn(sequence_length):
    if sequence_length == 1:
        return default_collate
    else:

        def sequence_collate(batch):
            x, y = default_collate(batch)
            if isinstance(y, list):
                y = torch.stack(y)
                y = y[:, ::sequence_length]
            else:
                y = y[::sequence_length]
            if isinstance(x, list):
                viwed_x = []
                for _x in x:
                    x_shape = list(_x.shape)
                    x_shape = [-1, sequence_length] + x_shape[1:]
                    viwed_x.append(_x.view(x_shape))
                return viwed_x, y
            else:
                x_shape = list(x.shape)
                x_shape = [-1, sequence_length] + x_shape[1:]
                return x.view(x_shape), y

        return sequence_collate


def get_fixed_dataloader(
    dataset: FileListDataset,
    batch_size: int,
    num_workers=6,
    sampler=RandomSampler,
    worker_init_fn=None,
):
    # look https://github.com/williamFalcon/pytorch-lightning/issues/434
    batch_sampler = SequenceBatchSampler(
        sampler(dataset),
        batch_size=batch_size,
        drop_last=True,
        sequence_length=dataset.sequence_length,
        samples_idx=dataset.samples_idx,
        dataset=dataset,
    )

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
        batch_sampler=_RepeatSampler(batch_sampler),
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        collate_fn=get_sequence_collate_fn(sequence_length=dataset.sequence_length),
    )

    return loader


class BalancedSampler(WeightedRandomSampler):
    def __init__(
        self, dataset: FileListDataset, replacement=True, predefined_weights=None
    ):
        class_weight_dict = calculate_class_weights(
            dataset, predefined_weights=predefined_weights
        )
        class_weights = np.array(
            [
                class_weight_dict[label_idx]
                for label_idx in sorted(dataset.class_to_idx.values())
            ]
        )
        targets = np.array(dataset.targets, dtype=np.int)[dataset.samples_idx]
        weights = class_weights[targets]

        super().__init__(weights, num_samples=len(dataset), replacement=replacement)


class SequenceBatchSampler(BatchSampler):
    def __init__(
        self,
        sampler,
        batch_size,
        drop_last,
        sequence_length: int,
        samples_idx,
        dataset: FileListDataset,
    ):
        super().__init__(sampler, batch_size, drop_last)
        self.sequence_length = sequence_length
        self.samples_idx = samples_idx
        self.d = dataset

        self.should_sample_audio = dataset.should_sample_audio

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            idx = self.samples_idx[idx]

            vid_idx = [(x, idx) for x in range(idx + 1 - self.sequence_length, idx + 1)]

            if self.should_sample_audio:
                aud_idx = self._sample_audio(idx)
            else:
                aud_idx = [None] * self.sequence_length

            batch += list(zip(vid_idx, aud_idx))

            if len(batch) == self.batch_size * self.sequence_length:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def _sample_audio(self, idx):
        # 50% matching
        if int(random.random() + (1.0 / 2.0)) or self.d.audio_mode == AudioMode.EXACT:
            # do nothing, because audio should match
            pass

        elif self.d.audio_mode == AudioMode.DIFFERENT_VIDEO:
            idx = np.random.choice(self.samples_idx)

        elif self.d.audio_mode == AudioMode.SAME_VIDEO_MIN_DISTANCE:
            offset = np.random.choice(
                self.d._get_possible_audio_shifts_with_min_distance(idx)
            )
            idx += offset
        elif self.d.audio_mode == AudioMode.SAME_VIDEO_MAX_DISTANCE:
            offset = np.random.choice(
                self.d._get_possible_audio_shifts_with_max_distance(idx)
            )
            idx += offset

        aud_idx = [x for x in range(idx + 1 - self.sequence_length, idx + 1)]
        return aud_idx
