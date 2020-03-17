from pathlib import Path
from typing import List

import numpy as np
import torch
from torchvision import transforms


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


def rfft(x: torch.tensor):
    """Applies torch.rfft to input (in 3 dimensions).

    Also permutes fourier space in front of c x w x h.
    i.e. input shape: b x c x w x h -> output shape: b x 2 x c x w x h

    Args:
        x: tensor that should fourier transform applied to

    Returns:
        Fourier transform of input

    """
    # insert last dimension infront of c x w x h
    # b x c x w x h x fourier -> b x fourier x c x w x h
    original_permutation = range(len(x.shape))
    permute = [
        *original_permutation[:-3],
        len(original_permutation),
        *original_permutation[-3:],
    ]
    return torch.rfft(x, 3, onesided=False, normalized=False).permute(permute)


def irfft(x: torch.tensor):
    """Applies torch.irfft to input (in 3 dimensions).

    First input is permuted so that fourier space is last dim. Assumes fourier space is
    4th last position.
    i.e.:
        input shape b x 2 x c x w x h -> (intermediate shape b x c x w x h x 2) ->
        output shape b x c x w x h

    Args:
        x: tensor that should be used for inverse fourier transform

    Returns:
        Spatial domain output

    """
    # move 4th last dimension to end
    # b x fourier x c x w x h x fourier -> b x c x w x h x 2
    original_permutation = range(len(x.shape))
    permute = [
        *original_permutation[:-4],
        *original_permutation[-3:],
        original_permutation[-4],
    ]
    return torch.irfft(x.permute(permute), 3, onesided=False, normalized=False)


def windowed_rfft(x: torch.tensor):
    """Applies windowed torch.rfft to input (in 3 dimensions).

    First the input is unfolded to generate 8 x 8 patches. Then the 3d rfft is performed
    on the channel, width and height dimension

    Also permutes fourier space in front of c x 8 x 8.
    i.e. input shape: b x c x w x h -> output shape: b x w // 8 x h // 8 x 2 x c x 8 x 8

    Args:
        x: tensor that should fourier transform applied to

    Returns:
        Fourier windowed transform of input

    """
    # create patches and put channel dimension on 3rd last place
    x = x.unfold(2, 8, 8).unfold(3, 8, 8).permute(0, 2, 3, 1, 4, 5)
    x_rfft = torch.rfft(x, 3, onesided=False, normalized=False).permute(
        0, 1, 2, 6, 3, 4, 5
    )
    return x_rfft


def rfft_transform():
    return [rfft]


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
