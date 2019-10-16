import shutil
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

DATASET_ROOT = Path.home() / "PycharmProjects" / "data_10"


class SaveImageFolder(ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except OSError as e:
            print("Some error with an image: ", e)
            print("With path:", self.samples[index][0])
            self.samples.pop(index)
            return self.__getitem__(index)


def get_data(data_dir) -> ImageFolder:
    """Get initialized ImageFolder with faceforensics data"""
    return SaveImageFolder(
        str(data_dir),
        transform=transforms.Compose(
            [
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )


def get_data_loaders(batch_size, validation_split=0.1, data_dir=DATASET_ROOT):

    # https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
    dataset = get_data(data_dir)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler
    )

    return train_loader, validation_loader


def _copy_images(source, target, target_val):
    target.mkdir(parents=True, exist_ok=True)
    target_val.mkdir(parents=True, exist_ok=True)
    for folder in source.iterdir():
        if folder.is_dir():
            image_folders = folder / "raw" / "images"
            if image_folders.exists():
                print("processing", folder)
                shutil.copytree(image_folders, target / folder.name)
                _do_val_split(target / folder.name, target_val)
            else:
                print("skipping", folder)


def _do_val_split(source, target):
    folders = list(source.glob("*"))
    # just take the last video as test
    try:
        shutil.move(str(folders[-1]), target / source.name / folders[-1].name)
    except OSError as e:
        print("probably the permissions of the source folder are read only: ", e)


def copy_all_images(source_dir, target_dir_train):
    """Copies images from source dir to target dir.

    source dir has to follow faceforensics folder structure:

    data:
        - manipulated_sequences:
            - a:
                - raw:
                    - images
            - b:
                - raw:
                    - images
            ...
        - original_sequences:
            - a:
                - raw:
                    - images
            - b:
                - raw:
                    - images
            ...

    Will do a train val split, by moving the las video of each method to a val folder.

    resulting structure:


    """
    source_dir = Path(source_dir)
    target_dir_val = Path(target_dir_train) / "val"
    target_dir_train = Path(target_dir_train) / "train"

    if not source_dir.exists():
        raise FileNotFoundError(f"{source_dir} does not exist")

    manipulated_sequences_source = source_dir / "manipulated_sequences"
    manipulated_sequences_target = target_dir_train / "manipulated_sequences"
    _copy_images(
        manipulated_sequences_source,
        manipulated_sequences_target,
        target_dir_val / "manipulated_sequences",
    )

    original_sequences_source = source_dir / "original_sequences"
    original_sequences_target = target_dir_train / "original_sequences"
    _copy_images(
        original_sequences_source,
        original_sequences_target,
        target_dir_val / "original_sequences",
    )


if __name__ == "__main__":
    copy_all_images("/media/sdb1/data_10", "/home/sebastian/Documents/data_10")
