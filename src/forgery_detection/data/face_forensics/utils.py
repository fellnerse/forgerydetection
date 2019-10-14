from pathlib import Path

from torchvision.datasets import ImageFolder

DATASET_ROOT = Path.home() / "PycharmProjects" / "data_10"


def get_data() -> ImageFolder:
    """Get initialized ImageFolder with faceforensics data"""
    return ImageFolder(str(DATASET_ROOT))
