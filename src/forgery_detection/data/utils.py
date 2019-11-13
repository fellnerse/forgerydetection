import os
from typing import Callable
from typing import List

from torchvision import transforms
from torchvision.datasets import ImageFolder


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
