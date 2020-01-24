from pathlib import Path

import numpy as np

from forgery_detection.data.set import FileList

root_dir = Path("/home/mo/datasets/solo/original")

f = FileList(root_dir, ["celeba"], 1)


images = list((root_dir / "train19k").glob("*.jpg"))
f.add_data_points(images, "celeba", "train", np.arange(0, len(images)))

images = list((root_dir / "val").glob("*.jpg"))
f.add_data_points(images, "celeba", "train", np.arange(0, len(images)))

images = list((root_dir / "test").glob("*.jpg"))
f.add_data_points(images, "celeba", "val", np.arange(0, len(images)))

images = list((root_dir / "test").glob("*.jpg"))
f.add_data_points(images, "celeba", "test", np.arange(0, len(images)))

f.root = str(f.root)
f.save("/data/ssd1/file_lists/celeba/celeba.json")

print(f.get_dataset("train"))
print(f.get_dataset("val"))
print(f.get_dataset("test"))

# for label in (root_dir / "train").iterdir():
#     images = list(label.glob("*.png"))
#     f.add_data_points(images, label.name, "train", np.arange(0, len(images)))
#
# for label in (root_dir / "test").iterdir():
#     images = list(label.glob("*.png"))
#     f.add_data_points(images, label.name, "val", np.arange(0, len(images)))
