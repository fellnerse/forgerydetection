import json
import shutil

import click
import cv2
import numpy as np
from tqdm import tqdm

from forgery_detection.data.face_forensics import Compression
from forgery_detection.data.face_forensics import DataType
from forgery_detection.data.face_forensics import FaceForensicsDataStructure


@click.command()
@click.option("--resampled_data_dir_root", required=True, type=click.Path(exists=True))
@click.option("--bb_data_dir_root", required=True, type=click.Path(exists=True))
@click.option("--compressions", multiple=True, default=Compression.c40)
def resample_tracked_bbs(resampled_data_dir_root, bb_data_dir_root, compressions):
    tracked_bb_data_structure = FaceForensicsDataStructure(
        bb_data_dir_root,
        compressions=Compression.c40,
        data_types=DataType.face_images_tracked,
    )
    resampled_videos_data_structure = FaceForensicsDataStructure(
        resampled_data_dir_root,
        compressions=Compression.c40,
        data_types=DataType.resampled_videos,
    )
    resampled_videos_face_images_data_structure = FaceForensicsDataStructure(
        resampled_data_dir_root,
        compressions=Compression.c40,
        data_types=DataType.face_images_tracked,
    )

    for resampled_videos, tracked_bbs, resampled_face_images in zip(
        resampled_videos_data_structure.get_subdirs(),
        tracked_bb_data_structure.get_subdirs(),
        resampled_videos_face_images_data_structure.get_subdirs(),
    ):
        print(f"Current method: {resampled_videos.parents[1].name}")

        for video in tqdm(sorted(resampled_videos.iterdir())):
            if resampled_videos.parents[1].name != "Deepfakes":
                continue

            tracked_bb = tracked_bbs / video.with_suffix("").name / "tracked_bb.json"
            resampled_tracked_bb = (
                resampled_face_images / video.with_suffix("").name / "tracked_bb.json"
            )
            resampled_tracked_bb.parent.mkdir(parents=True, exist_ok=True)

            with open(tracked_bb, "r") as f:
                tracked_bb_dict = json.load(f)
                cap = cv2.VideoCapture(str(video))
                property_id = int(cv2.CAP_PROP_FRAME_COUNT)
                resampled_video_length = int(cv2.VideoCapture.get(cap, property_id))

                tracked_bb_length = len(tracked_bb_dict)

                # if there is no difference in frame_count, we do not need to do extr
                # stuff
                if tracked_bb_length == resampled_video_length:
                    shutil.copy(str(tracked_bb), str(resampled_tracked_bb))
                else:

                    # this means there is at least one frame without tracking
                    # information
                    # in this video
                    tracked_bb_values = list(tracked_bb_dict.values())
                    data_points_x = np.linspace(0, 100_000, tracked_bb_length)
                    interpolated_x = np.linspace(0, 100_000, resampled_video_length)
                    resampled_idx = []
                    current_pos = 0
                    for x in interpolated_x:
                        while (
                            x > data_points_x[current_pos]
                            and current_pos < tracked_bb_length
                        ):
                            current_pos += 1
                        resampled_idx.append(current_pos)

                    # now we have the mappings and have to choose the values
                    resampled_bbs = {}
                    for _i, idx in enumerate(resampled_idx):
                        # make sure that we do not have discontinued sequences
                        # aka a two sequences after another without a None in between
                        next_val = tracked_bb_values[idx]
                        if (
                            idx != 0
                            and resampled_bbs[f"{_i-1:04d}"] is not None
                            and resampled_bbs[f"{_i-1:04d}"] != next_val
                        ):
                            resampled_bbs[f"{_i:04d}"] = None
                        else:
                            resampled_bbs[f"{_i:04d}"] = next_val

                    with open(resampled_tracked_bb, "w") as f:
                        json.dump(resampled_bbs, f)


if __name__ == "__main__":
    resample_tracked_bbs()
