# flake8: noqa
# %%
from forgery_detection.data.file_lists import FileList

f = FileList.load(
    "/home/sebastian/data/file_lists/c40/tracked_resampled_faces_yt_only_112_8_sequence_length.json"
)

#%%
import json

with open("/home/sebastian/data/misc/felix_reviewed_ff_videos.json", "r") as f_:
    white_list = json.load(f_)

#%%
from pathlib import Path

ff_syncnet_evaluation_folder = Path(
    "/home/sebastian/data/file_lists/syncnet_evaluation_ff"
)
import shutil

mfcc_file = Path("/home/sebastian/data/audio_features/mfcc_features_file_list.json")

for vid in white_list[:10]:
    print(vid)
    for split in ["train", "val", "test"]:
        print(split)
        samples = f.samples[split]
        matches = list(
            filter(
                lambda x: "youtube" in x[1][0] and vid in x[1][0].split("/")[-2],
                enumerate(samples),
            )
        )
        if len(matches):
            new_filelist = FileList(
                root=f.root,
                classes=f.classes,
                min_sequence_length=f.min_sequence_length,
            )
            new_filelist.samples["test"] = [x[1] for x in matches]

            min_idx = min(matches, key=lambda x: x[0])[0]
            max_idx = max(matches, key=lambda x: x[0])[0]

            new_filelist.samples_idx["test"] = list(
                map(
                    lambda x: x - min_idx,
                    filter(lambda x: min_idx <= x <= max_idx, f.samples_idx[split]),
                )
            )
            curr_folder = ff_syncnet_evaluation_folder / vid
            curr_folder.mkdir(exist_ok=True)

            shutil.copy(str(mfcc_file), str(curr_folder / "audio_file_list.json"))

            new_filelist.save(curr_folder / "image_file_list.json")
