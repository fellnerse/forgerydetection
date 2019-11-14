import json
import os
from pathlib import Path
from pprint import pformat
from shutil import copy2
from typing import Callable
from typing import List

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

from forgery_detection.data.face_forensics.splits import TEST_NAME
from forgery_detection.data.face_forensics.splits import TRAIN_NAME
from forgery_detection.data.face_forensics.splits import VAL_NAME


class SafeImageFolder(ImageFolder):
    def __init__(self, root, *args, **kwargs):
        # make the imagefolder follow symlinks during initialization
        _os_walk = os.walk

        def _walk(dir, *args, **kwargs):
            return _os_walk(dir, followlinks=True)

        setattr(os, "walk", _walk)
        super().__init__(root, *args, **kwargs)
        setattr(os, "walk", _os_walk)

    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except OSError as e:
            print("Some error with an image: ", e)
            print("With path:", self.samples[index][0])
            print("called with index:", index)
            self.samples.pop(index)
            return self.__getitem__(index % len(self))


class FileList:
    def __init__(self, root: str, classes: List[str]):
        self.root = root
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = {TRAIN_NAME: [], VAL_NAME: [], TEST_NAME: []}

    def add_data_point(self, path: Path, target_label: str, split: str):
        """Adds datapoint to samples.

        Args:
            path: has to be subpath of self.root. Will be saved relative to it.
            target_label: label of the datapoints. Is converted to idx via
                self.class_to_idx
            split: indicates current split (train, val, test)

        """
        self.samples[split].append(
            (str(path.relative_to(self.root)), self.class_to_idx[target_label])
        )

    def save(self, path):
        """Save self.__dict__ as json."""
        with open(path, "w") as f:
            json.dump(self.__dict__, f)  # carefull with self.root->Path

    @classmethod
    def load(cls, path):
        """Restore instance from json via self.__dict__."""
        with open(path, "r") as f:
            __dict__ = json.load(f)
        file_list = cls.__new__(cls)
        file_list.__dict__.update(__dict__)
        return file_list

    def copy_to(self, new_root: Path):
        curr_root = Path(self.root)
        for data_points in tqdm(self.samples.values(), position=0):
            for data_point_path, _ in tqdm(data_points, position=1):
                target_path = new_root / data_point_path
                target_path.parents[0].mkdir(exist_ok=True, parents=True)
                copy2(curr_root / data_point_path, target_path)

        self.root = str(new_root)
        return self

    def binarize(self):
        num_old_classes = len(self.classes)
        negative = "_".join(self.classes[:-1])
        positive = self.classes[-1]
        self.classes = [negative, positive]
        self.class_to_idx = {negative: 0, positive: 1}

        for data_points in self.samples.values():
            for datapoint in data_points:
                datapoint[1] = 1 if datapoint[1] == num_old_classes - 1 else 0

    def get_dataset(self, split, transform=None) -> Dataset:
        """Get dataset by using this instance."""
        transform = transform or []
        transform = transforms.Compose(
            transform
            + [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return FileListDataset(file_list=self, split=split, transform=transform)

    @classmethod
    def get_dataset_form_file(cls, path, split, transform=None) -> Dataset:
        """Get dataset by loading a FileList and calling get_dataset on it."""
        return cls.load(path).get_dataset(split, transform)

    def __str__(self):
        return pformat(self.class_to_idx, indent=4)


class FileListDataset(VisionDataset):
    """Almost the same as DatasetFolder by pyTorch.

    But this one does not build up a file list by walking a folder. Instead this file
    list has to be provided."""

    def __init__(
        self, file_list: FileList, split: str, transform=None, target_transform=None
    ):
        super().__init__(
            file_list.root, transform=transform, target_transform=target_transform
        )
        self.loader = default_loader

        self.classes = file_list.classes
        self.class_to_idx = file_list.class_to_idx
        self.samples = file_list.samples[split]
        self.targets = [s[1] for s in self.samples]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(self.root + "/" + path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


def crop():
    return [transforms.CenterCrop(299)]


def resized_crop():
    return [transforms.Resize(299), transforms.CenterCrop(299)]


def resized_crop_flip():
    return [
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.RandomHorizontalFlip(),
    ]


def get_data(
    data_dir, custom_transforms: Callable[[], List[Callable]] = crop
) -> ImageFolder:
    """Get initialized ImageFolder with faceforensics data"""
    return SafeImageFolder(
        str(data_dir),
        transform=transforms.Compose(
            custom_transforms()
            + [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )
