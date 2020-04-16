import os
from pathlib import Path

import numpy as np

from forgery_detection.data.face_forensics.splits import TEST_NAME
from forgery_detection.data.face_forensics.splits import TRAIN_NAME
from forgery_detection.data.face_forensics.splits import VAL_NAME
from forgery_detection.data.file_lists import FileList

file_list_a = FileList.load(
    "/data/ssd1/file_lists/avspeech/100k_100_samples_consolidated.json"
)
file_list_b = FileList.load(
    "/data/ssd1/file_lists/avspeech/avspeech_moria_20k_100_samples_consolidated.json"
)


a_root = Path(file_list_a.root)
b_root = Path(file_list_b.root)
common_path = os.path.commonpath([b_root, a_root])
a_relative_to_root = os.path.relpath(a_root, common_path)
b_relative_to_root = os.path.relpath(b_root, common_path)

print(common_path, a_relative_to_root, b_relative_to_root)


file_list_a.root = common_path

for split in file_list_a.samples.values():
    for item in split:
        item[0] = a_relative_to_root + "/" + item[0]

print(file_list_a.samples["train"][-1])

for split in file_list_b.samples.values():
    for item in split:
        item[0] = b_relative_to_root + "/" + item[0]

print(file_list_b.samples["train"][-1])
print(file_list_a.samples_idx["train"][-1], len(file_list_a.samples["train"]))

# actually merge the samples

for split_name in [TRAIN_NAME, VAL_NAME, TEST_NAME]:
    a_split = file_list_a.samples[split_name]
    a_split_len = len(a_split)

    b_split = file_list_b.samples[split_name]
    a_split.extend(b_split)

    b_idx = file_list_b.samples_idx[split_name]
    b_idx = (np.array(b_idx) + a_split_len).tolist()

    a_idx = file_list_a.samples_idx[split_name]
    a_idx.extend(b_idx)


print(file_list_b.samples["train"][-1])
print(file_list_a.samples_idx["train"][-1], len(file_list_a.samples["train"]))

# save merged file_list

file_list_a.save("/data/ssd1/file_lists/avspeech/100k_100_samples_consolidated.json")

merged = FileList.load(
    "/data/ssd1/file_lists/avspeech/100k_100_samples_consolidated.json"
)
d = merged.get_dataset("train")
