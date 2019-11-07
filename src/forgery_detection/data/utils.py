import os
from typing import Callable
from typing import List

import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
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


class FiftyFiftySampler(WeightedRandomSampler):
    def __init__(self, dataset: ImageFolder, replacement=True):

        weights = np.array(dataset.targets, dtype=np.float)

        weights_1 = 1 / weights.sum()
        weights_0 = 1 / (len(weights) - weights.sum())

        weights[weights == 1] = weights_1
        weights[weights == 0] = weights_0

        super().__init__(weights, num_samples=len(dataset), replacement=replacement)


def crop():
    return [transforms.CenterCrop(299)]


def resized_crop():
    return [transforms.Resize(299), transforms.CenterCrop(299)]


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


if __name__ == "__main__":
    data_set = get_data("/mnt/ssd1/sebastian/face_forensics_1000_c40_test/test")
    ffs = FiftyFiftySampler(data_set)
    counter = 0
    for i in ffs:
        if counter > 10:
            break
        print(i, data_set.targets[i])
        counter += 1
