from __future__ import annotations

from typing import List
from typing import Tuple
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import WeightedRandomSampler
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets.folder import default_loader


if TYPE_CHECKING:
    from forgery_detection.data.set import FileListDataset


def calculate_class_weights(dataset: FileListDataset) -> Tuple[List[str], List[float]]:
    labels, counts = np.unique(dataset.targets, return_counts=True)
    counts = 1 / counts
    counts /= counts.sum()
    return list(map(lambda idx: dataset.classes[idx], labels)), counts


def get_sequence_collate_fn(sequence_length):
    if sequence_length == 1:
        return default_collate
    else:

        def sequence_collate(batch):
            x, y = default_collate(batch)
            if isinstance(x, list):
                viwed_x = []
                for _x in x:
                    x_shape = list(_x.shape)
                    x_shape = [-1, sequence_length] + x_shape[1:]
                    viwed_x.append(_x.view(x_shape))
                return viwed_x, y[::sequence_length]
            else:
                x_shape = list(x.shape)
                x_shape = [-1, sequence_length] + x_shape[1:]
                return x.view(x_shape), y[::sequence_length]

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
        drop_last=False,
        sequence_length=dataset.sequence_length,
        samples_idx=dataset.samples_idx,
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
    def __init__(self, dataset: FileListDataset, replacement=True):

        targets = np.array(dataset.targets, dtype=np.int)[dataset.samples_idx]
        _, class_weights = calculate_class_weights(dataset)
        weights = class_weights[targets]

        super().__init__(weights, num_samples=len(dataset), replacement=replacement)


class SequenceBatchSampler(BatchSampler):
    def __init__(
        self, sampler, batch_size, drop_last, sequence_length: int, samples_idx
    ):
        super().__init__(sampler, batch_size, drop_last)
        self.sequence_length = sequence_length
        self.samples_idx = samples_idx

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            idx = self.samples_idx[idx]
            batch += range(idx + 1 - self.sequence_length, idx + 1)
            if len(batch) == self.batch_size * self.sequence_length:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch


class ExtendedDefaultLoader:
    """Class used for loading files from disk.

    Additionally to pytorchts default loader this also can load corresponding audio for
    images (given a np-file containing such additional information).

    """

    def __init__(self, audio_file: str = None):
        self.should_load_audio = audio_file is not None

        if self.should_load_audio:
            try:
                # this weird access is only because of numpy saving a dict behaves
                # strange
                self.audio = np.load(audio_file, allow_pickle=True)[()]
            except FileNotFoundError:
                raise FileNotFoundError(f"Could not find {audio_file}.")

    def load_data(self, path: str):
        if self.should_load_audio:
            try:
                return default_loader(path), self._load_audio(path)
            except (KeyError, IndexError):
                print(path)
                raise
        else:
            return default_loader(path)

    def _load_audio(self, path):
        parts = path.split("/")
        video_name = parts[-2]
        image_name = parts[-1].split(".")[0]

        if "youtube" in path:
            corresponding_audio = self.audio[video_name]
        else:
            video_names = video_name.split("_")
            # for now just use primary video because some of the others are not available
            corresponding_audio = self.audio[video_names[0]]
            # if "Deepfakes" in path:
            #     corresponding_audio = self.audio[video_names[0]]
            # else:
            #     corresponding_audio = self.audio[video_names[1]]
        # remove modulo
        return corresponding_audio[int(image_name) % len(corresponding_audio)]
