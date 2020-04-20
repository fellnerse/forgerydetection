import os
from pathlib import Path

from forgery_detection.data.face_forensics.splits import TRAIN_NAME
from forgery_detection.data.file_lists import FileList

resampled_file_list = FileList.load(
    "/data/ssd1/file_lists/c40/tracked_resampled_faces.json"
)
detection_file_list = FileList.load(
    "/data/ssd1/file_lists/c40/detection_challenge_112.json"
)


resampled_root = Path(resampled_file_list.root)
detection_root = Path(detection_file_list.root)
common_path = os.path.commonpath([detection_root, resampled_root])
resampled_relative_to_root = os.path.relpath(resampled_root, common_path)
detection_relative_to_root = os.path.relpath(detection_root, common_path)

print(common_path, resampled_relative_to_root, detection_relative_to_root)

# change class idx values for resampled file list
# make youtube one value higher
resampled_file_list.class_to_idx = {"DeepFakeDetection": 0, "youtube": 1}
resampled_file_list.classes = ["DeepFakeDetection", "youtube"]
resampled_file_list.root = common_path

for split in resampled_file_list.samples.values():
    for item in split:
        if item[1] == 4:
            item[1] = 1
        else:
            item[1] = 0

        item[0] = resampled_relative_to_root + "/" + item[0]

print(resampled_file_list.samples["train"][-1])

# change class idx values for detection file list
detection_file_list.class_to_idx = resampled_file_list.class_to_idx
detection_file_list.classes = resampled_file_list.classes
for split in detection_file_list.samples.values():
    for item in split:
        if item[1] == 0:
            item[1] = 1
        elif item[1] == 1:
            item[1] = 0
        item[0] = detection_relative_to_root + "/" + item[0]

print(detection_file_list.samples["train"][-1])
print(
    resampled_file_list.samples_idx["train"][-1],
    len(resampled_file_list.samples["train"]),
)

# actually merge the samples

resampled_file_list.samples[TRAIN_NAME] = detection_file_list.samples[TRAIN_NAME]
resampled_file_list.samples_idx[TRAIN_NAME] = detection_file_list.samples_idx[
    TRAIN_NAME
]

#
print(detection_file_list.samples["train"][-1])
print(
    resampled_file_list.samples_idx["train"][-1],
    len(resampled_file_list.samples["train"]),
)

# save merged file_list

resampled_file_list.save(
    "/data/ssd1/file_lists/c40/detection_and_resampled_val_112.json"
)

merged = FileList.load("/data/ssd1/file_lists/c40/detection_and_resampled_val_112.json")
d = merged.get_dataset("train")
