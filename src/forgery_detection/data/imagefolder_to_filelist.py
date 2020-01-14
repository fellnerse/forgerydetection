from pathlib import Path

import numpy as np

from forgery_detection.data.set import FileList

root_dir = Path("/data/hdd/cifar10")

f = FileList(
    root_dir,
    [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ],
    1,
)

for label in (root_dir / "train").iterdir():
    images = list(label.glob("*.png"))
    f.add_data_points(images, label.name, "train", np.arange(0, len(images)))

for label in (root_dir / "test").iterdir():
    images = list(label.glob("*.png"))
    f.add_data_points(images, label.name, "val", np.arange(0, len(images)))
