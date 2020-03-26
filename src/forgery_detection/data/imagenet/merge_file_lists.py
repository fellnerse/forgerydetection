import os
from pathlib import Path

import numpy as np

from forgery_detection.data.face_forensics.splits import TEST_NAME
from forgery_detection.data.face_forensics.splits import TRAIN_NAME
from forgery_detection.data.face_forensics.splits import VAL_NAME
from forgery_detection.data.set import FileList

resampled_file_list = FileList.load(
    "/data/ssd1/file_lists/c40/tracked_resampled_faces_224.json"
)
imagenet = FileList.load("/data/ssd1/file_lists/imagenet/ssd_raw.json")

resampled_root = Path(resampled_file_list.root)
imagenet_root = Path(imagenet.root)
common_path = os.path.commonpath([imagenet_root, resampled_root])
resampled_relative_to_root = os.path.relpath(resampled_root, common_path)
imagenet_relative_to_root = os.path.relpath(imagenet_root, common_path)

print("path stuff", common_path, resampled_relative_to_root, imagenet_relative_to_root)

resampled_file_list.class_to_idx = {
    **imagenet.class_to_idx,
    **dict(
        map(
            lambda x: (x[0], x[1] + len(imagenet.class_to_idx)),
            resampled_file_list.class_to_idx.items(),
        )
    ),
}
resampled_file_list.classes = imagenet.classes + resampled_file_list.classes
print(resampled_file_list.samples["train"][-1])


for split in resampled_file_list.samples.values():
    for item in split:
        item[0] = resampled_relative_to_root + "/" + item[0]
        item[1] += len(imagenet.class_to_idx)

# imagenet.class_to_idx = resampled_file_list.class_to_idx
# imagenet.classes = resampled_file_list.classes
for split in imagenet.samples.values():
    for item in split:
        item[0] = imagenet_relative_to_root + "/" + item[0]

print(imagenet.samples["train"][-1])
print(
    resampled_file_list.samples_idx["train"][-1],
    len(resampled_file_list.samples["train"]),
)

# Merge train split
split_name = TRAIN_NAME

resampled_split = resampled_file_list.samples[split_name]
resampled_split_len = len(resampled_split)

imagenet_split = imagenet.samples[split_name]
resampled_split.extend(imagenet_split)

imagenet_idx = imagenet.samples_idx[split_name]
imagenet_idx = (np.array(imagenet_idx) + resampled_split_len).tolist()

resampled_idx = resampled_file_list.samples_idx[split_name]
resampled_idx.extend(imagenet_idx)

# add resampled_val to resampled_test
resampled_test = resampled_file_list.samples[TEST_NAME]
resampled_test_len = len(resampled_test)

resampled_val = resampled_file_list.samples[VAL_NAME]
resampled_test.extend(resampled_val)

val_idx = resampled_file_list.samples_idx[VAL_NAME]
val_idx = (np.array(val_idx) + resampled_test_len).tolist()

resampled_idx = resampled_file_list.samples_idx[TEST_NAME]
resampled_idx.extend(val_idx)

# copy imagenet val data over to resampled_val
resampled_file_list.samples[VAL_NAME] = imagenet.samples[VAL_NAME]
resampled_file_list.samples_idx[VAL_NAME] = imagenet.samples_idx[VAL_NAME]

# former merge
# for split_name in [TRAIN_NAME, VAL_NAME, TEST_NAME]:
#     resampled_split = resampled_file_list.samples[split_name]
#     resampled_split_len = len(resampled_split)
#
#     imagenet_split = imagenet.samples[split_name]
#     resampled_split.extend(imagenet_split)
#
#     imagenet_idx = imagenet.samples_idx[split_name]
#     imagenet_idx = (np.array(imagenet_idx) + resampled_split_len).tolist()
#
#     resampled_idx = resampled_file_list.samples_idx[split_name]
#     resampled_idx.extend(imagenet_idx)

print(imagenet.samples["train"][-1])
print(
    resampled_file_list.samples_idx["train"][-1],
    len(resampled_file_list.samples["train"]),
)

resampled_file_list.root = common_path
file_path = "/data/ssd1/file_lists/imagenet/merged_224_.json"
resampled_file_list.save(file_path)

merged = FileList.load(file_path)
d = merged.get_dataset("train")
