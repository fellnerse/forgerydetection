from pathlib import Path
from typing import Callable
from typing import List

from torchvision import transforms
from torchvision.datasets import ImageFolder

from forgery_detection.data.set import SafeImageFolder


def crop(size=299):
    return [transforms.CenterCrop(size)]


def resized_crop(size=299):
    return [transforms.Resize(size), transforms.CenterCrop(size)]


def resized_crop_flip(size=299):
    return [
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.RandomHorizontalFlip(),
    ]


def random_resized_crop(size=112):
    return [transforms.RandomResizedCrop(size=size, scale=(0.75, 1.0))]


def random_horizontal_flip():
    return [transforms.RandomHorizontalFlip()]


def colour_jitter():
    return [transforms.ColorJitter()]


def random_rotation():
    return [transforms.RandomRotation(15)]


def random_greyscale():
    return [transforms.RandomGrayscale()]


def random_erasing():
    return [
        transforms.ToTensor(),
        transforms.RandomErasing(scale=(0.02, 0.11)),
        transforms.ToPILImage(),
    ]


def random_flip_rotation():
    return [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)]


def random_flip_greyscale():
    return [transforms.RandomHorizontalFlip(), transforms.RandomGrayscale()]


def random_rotation_greyscale():
    return [transforms.RandomRotation(15), transforms.RandomGrayscale()]


def random_flip_rotation_greyscale():
    return [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomGrayscale(),
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


def _img_name_to_int(img: Path):
    return int(img.name.split(".")[0])
