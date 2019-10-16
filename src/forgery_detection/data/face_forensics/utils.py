from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

DATASET_ROOT = Path.home() / "PycharmProjects" / "data_10"


def get_data(data_dir) -> ImageFolder:
    """Get initialized ImageFolder with faceforensics data"""
    return ImageFolder(
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
