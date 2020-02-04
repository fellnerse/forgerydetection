from pathlib import Path
from typing import Callable
from typing import List

import numpy as np
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
    return [transforms.RandomApply([transforms.RandomRotation(15)], 0.1)]


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


def img_name_to_int(img: Path):
    return int(img.with_suffix("").name)


def select_frames(nb_images: int, samples_per_video: int) -> List[int]:
    """Selects frames to take from video.

    Args:
        nb_images: length of video aka. number of frames in video
        samples_per_video: how many frames of this video should be taken. If this value
            is bigger then nb_images or -1, nb_images are taken.

    """
    if samples_per_video == -1 or samples_per_video > nb_images:
        selected_frames = range(nb_images)
    else:
        selected_frames = np.rint(
            np.linspace(1, nb_images, min(samples_per_video, nb_images)) - 1
        ).astype(int)
    return selected_frames
