import json
import logging
import os
import pickle
from pathlib import Path
from pprint import pformat
from shutil import copy2
from typing import List
from typing import Optional

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from forgery_detection.data.face_forensics.splits import TEST_NAME
from forgery_detection.data.face_forensics.splits import TRAIN_NAME
from forgery_detection.data.face_forensics.splits import VAL_NAME
from forgery_detection.data.set import FileListDataset
from forgery_detection.lightning.logging.const import AudioMode

logger = logging.getLogger(__file__)


class SimpleFileList:
    def __init__(self, root: str):
        self.root = root
        self.files = {}

    def save(self, path):
        """Save self.__dict__ as json."""
        with open(path, "w") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        """Restore instance from json via self.__dict__."""
        with open(path, "r") as f:
            __dict__ = json.load(f)
        file_list = cls.__new__(cls)
        file_list.__dict__.update(__dict__)
        file_list._load_data_in_memory()
        return file_list

    def __call__(self, path, stacked=False):
        parts = path.split("/")
        video_name = "/".join(parts[:-1])
        image_name = parts[-1].split(".")[0]

        corresponding_audio = self.files[video_name]

        try:
            if not stacked:
                return corresponding_audio[int(image_name)]
            else:
                start = int(image_name) - 4
                end = int(image_name) + 5

                if start < 0 or end > len(corresponding_audio):
                    return np.concatenate(
                        (
                            np.zeros((abs(min(start, 0)), 4, 13), dtype=np.float32),
                            corresponding_audio[
                                max(start, 0) : min(end, len(corresponding_audio))
                            ],
                            np.zeros(
                                (abs(min(0, len(corresponding_audio) - end)), 4, 13),
                                dtype=np.float32,
                            ),
                        )
                    )
                else:
                    return corresponding_audio[start:end]

        except IndexError:
            logger.error(
                f"{int(image_name)} is out of bounds for {len(corresponding_audio)}.\n"
                f"path is: {path} "
            )
            raise

    def _load_data_in_memory(self):
        total_not_reshaped = 0
        for key, path in self.files.items():
            with open(os.path.join(self.root, path), "rb") as f:
                features = pickle.load(f)  # 13 x [len(video)*4]

            if features.shape[0] == 13:
                try:
                    features = (
                        np.transpose(features, (1, 0))
                        .reshape((-1, 4, 13))
                        .astype("float32")
                    )  # len(video) x 4 x 13
                except ValueError:
                    logger.error(
                        f"skipping {path}, as the shape is off: {features.shape}"
                    )
            else:
                total_not_reshaped += 1
            self.files[key] = features
        logger.warning(f"not reshaping {total_not_reshaped} items")

    def __repr__(self):
        return f"""SimpleFileList:

root={self.root}
number_of_elements={len(self.files)}"""


class FileList:
    def __init__(self, root: str, classes: List[str], min_sequence_length: int):
        self.root = root
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = {TRAIN_NAME: [], VAL_NAME: [], TEST_NAME: []}
        self.samples_idx = {TRAIN_NAME: [], VAL_NAME: [], TEST_NAME: []}
        self.relative_bbs = {TRAIN_NAME: [], VAL_NAME: [], TEST_NAME: []}

        self.min_sequence_length = min_sequence_length

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

    def add_data_points(
        self,
        path_list: List[Path],
        target_label: str,
        split: str,
        sampled_images_idx: np.array,
    ):
        nb_samples_offset = len(self.samples[split])
        sampled_images_idx = (sampled_images_idx + nb_samples_offset).tolist()
        self.samples_idx[split] += sampled_images_idx

        for path in path_list:
            self.add_data_point(path, target_label, split)

    def save(self, path):
        """Save self.__dict__ as json."""
        with open(path, "w") as f:
            json.dump(self.__dict__, f)  # be careful with self.root->Path

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

    def get_dataset(
        self,
        split,
        image_transforms=None,
        tensor_transforms=None,
        sequence_length: int = 1,
        should_align_faces=False,
        audio_file_list: Optional[SimpleFileList] = None,
        audio_mode: AudioMode = AudioMode.EXACT,
    ) -> Dataset:
        """Get dataset by using this instance."""
        if sequence_length > self.min_sequence_length:
            logger.warning(
                f"{sequence_length}>{self.min_sequence_length}. Trying to load data that"
                f"does not exist might raise an error in the FileListDataset."
            )
        image_transforms = image_transforms or []
        tensor_transforms = tensor_transforms or []
        image_transforms = transforms.Compose(
            image_transforms
            + [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
            + tensor_transforms
        )
        return FileListDataset(
            file_list=self,
            split=split,
            sequence_length=sequence_length,
            should_align_faces=should_align_faces,
            transform=image_transforms,
            audio_file_list=audio_file_list,
            audio_mode=audio_mode,
        )

    @classmethod
    def get_dataset_form_file(
        cls, path, split, transform=None, sequence_length: int = 1
    ) -> Dataset:
        """Get dataset by loading a FileList and calling get_dataset on it."""
        return cls.load(path).get_dataset(
            split=split, image_transforms=transform, sequence_length=sequence_length
        )

    def __str__(self):
        return pformat(self.class_to_idx, indent=4)
