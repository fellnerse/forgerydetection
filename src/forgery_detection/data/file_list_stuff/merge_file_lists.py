import os
from pathlib import Path

import numpy as np

from forgery_detection.data.face_forensics.splits import TEST_NAME
from forgery_detection.data.face_forensics.splits import TRAIN_NAME
from forgery_detection.data.face_forensics.splits import VAL_NAME
from forgery_detection.data.file_lists import FileList

resampled_file_list = FileList.load(
    "/data/ssd1/file_lists/c40/tracked_resampled_faces.json"
)
celeba = FileList.load(
    "/data/ssd1/file_lists/avspeech/100k_100_samples_consolidated.json"
)

resampled_file_list.root = "/mnt/ssd2/sebastian/set/tracked_resampled_faces_112/"
resampled_root = Path(resampled_file_list.root)
celeba_root = Path(celeba.root)
common_path = os.path.commonpath([celeba_root, resampled_root])
resampled_relative_to_root = os.path.relpath(resampled_root, common_path)
celeba_relative_to_root = os.path.relpath(celeba_root, common_path)

print(common_path, resampled_relative_to_root, celeba_relative_to_root)

# change class idx values for resampled file list
# make youtube one value higher
resampled_file_list.class_to_idx["avspeech"] = 5
resampled_file_list.classes.append("avspeech")
# resampled_file_list.class_to_idx["youtube"] = 5
# resampled_file_list.classes.append("youtube")

resampled_file_list.root = common_path

for split in resampled_file_list.samples.values():
    for item in split:
        # if item[1] == 4:
        #     item[1] = 5

        item[0] = resampled_relative_to_root + "/" + item[0]

#
print(resampled_file_list.samples["train"][-1])

# change class idx values for detection file list
celeba.class_to_idx = resampled_file_list.class_to_idx
celeba.classes = resampled_file_list.classes
for split in celeba.samples.values():
    for item in split:
        # if item[1] == 0:
        #     item[1] = 5
        # elif item[1] == 1:
        #     item[1] = 4
        item[1] = 5
        item[0] = celeba_relative_to_root + "/" + item[0]

print(celeba.samples["train"][-1])
print(
    resampled_file_list.samples_idx["train"][-1],
    len(resampled_file_list.samples["train"]),
)

# actually merge the samples

for split_name in [TRAIN_NAME, VAL_NAME, TEST_NAME]:
    resampled_split = resampled_file_list.samples[split_name]
    resampled_split_len = len(resampled_split)

    detection_split = celeba.samples[split_name]
    resampled_split.extend(detection_split)

    detection_idx = celeba.samples_idx[split_name]
    detection_idx = (np.array(detection_idx) + resampled_split_len).tolist()

    resampled_idx = resampled_file_list.samples_idx[split_name]
    resampled_idx.extend(detection_idx)

#
print(celeba.samples["train"][-1])
print(
    resampled_file_list.samples_idx["train"][-1],
    len(resampled_file_list.samples["train"]),
)

# save merged file_list

resampled_file_list.save(
    "/data/ssd1/file_lists/avspeech/resampled_and_avspeech_100_samples_consolidated.json"
)

merged = FileList.load(
    "/data/ssd1/file_lists/avspeech/resampled_and_avspeech_100_samples_consolidated.json"
)
d = merged.get_dataset("train")
