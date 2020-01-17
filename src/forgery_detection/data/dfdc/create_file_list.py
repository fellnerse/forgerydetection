import json
from pathlib import Path

import numpy as np

from forgery_detection.data.face_forensics.splits import TEST_NAME
from forgery_detection.data.face_forensics.splits import TRAIN_NAME
from forgery_detection.data.face_forensics.splits import VAL_NAME
from forgery_detection.data.set import FileList

root_dir = Path("/data/hdd/dfdc")

with open(root_dir / "all_metadata.json", "r") as f:
    all_meta_data = json.load(f)

f = FileList(str(root_dir), classes=["FAKE", "REAL"], min_sequence_length=1)

train_data_numbers = list(range(5, 50))
val_data_numbers = list(range(5))

for train_data_number in train_data_numbers:
    block = root_dir / f"extracted_images_{train_data_number}"
    if block.exists():
        for label in block.iterdir():
            images = list(label.glob("*/*.png"))
            f.add_data_points(images, label.name, "train", np.arange(0, len(images)))

for val_data_number in val_data_numbers:
    block = root_dir / f"extracted_images_{val_data_number}"
    if block.exists():
        for label in block.iterdir():
            images = list(label.glob("*/*.png"))
            f.add_data_points(images, label.name, "val", np.arange(0, len(images)))
            f.add_data_points(images, label.name, "test", np.arange(0, len(images)))

f.save("/data/ssd1/file_lists/dfdc/5_45_split.json")

for split in [TRAIN_NAME, VAL_NAME, TEST_NAME]:
    data_set = FileList.get_dataset_form_file(
        "/data/ssd1/file_lists/dfdc/5_45_split.json", split
    )
    print(f"{split}-data-set: {data_set}")
