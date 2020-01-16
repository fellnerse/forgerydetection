import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import List

import click
import face_recognition
from cv2 import cv2
from joblib import delayed
from joblib import Parallel
from tqdm import tqdm

logger = logging.getLogger(__file__)


def extract_first_face(video, extracted_images_dir, meta_data):
    video_path = (
        extracted_images_dir / meta_data["label"] / video.split("/")[-1].split(".")[0]
    )
    video_path.mkdir(exist_ok=True)
    if len(list(video_path.glob("*"))):
        logger.warning(
            f"Files inside {video_path}: {list(video_path.glob('*'))}. Skipping it."
        )
        return
    # find first face
    capture = cv2.VideoCapture(video)
    frame_num = 0
    while capture.isOpened():
        # Read next frame
        ret = capture.grab()

        if not ret:
            capture.release()
            break

        if frame_num % 10 == 0:
            ret, frame = capture.retrieve()

            if not ret:
                capture.release()
                break

            face_locations = face_recognition.face_locations(frame)

            if face_locations:
                # save it to disk
                top, right, bottom, left = face_locations[0]

                x = left
                y = top
                w = right - x
                h = bottom - y
                cropped_face = frame[y : y + h, x : x + w]  # noqa: E203
                cv2.imwrite(str(video_path / f"{frame_num:04d}.png"), cropped_face)
                break

        frame_num += 1
    capture.release()


@click.command()
@click.option("--folder_numbers", "-n", multiple=True, required=True, type=int)
@click.option("--data_dir", type=click.Path(exists=True))
def extract_images(folder_numbers: List[int], data_dir: click.Path):
    root_dir = Path(data_dir)
    with open(root_dir / "all_metadata.json", "r") as f:
        all_meta_data = json.load(f)

    for folder_number in tqdm(folder_numbers):
        extracted_images_dir = root_dir / f"extracted_images_{folder_number}"
        extracted_images_dir.mkdir(exist_ok=True)
        real_folder = extracted_images_dir / "FAKE"
        real_folder.mkdir(exist_ok=True)
        fake_folder = extracted_images_dir / "REAL"
        fake_folder.mkdir(exist_ok=True)

        all_meta_data_filtered = dict(
            filter(
                lambda item: item[0].split("/")[-2].endswith(f"part_{folder_number}"),
                all_meta_data.items(),
            )
        )

        Parallel(n_jobs=mp.cpu_count())(
            delayed(
                lambda _video, _meta_data: extract_first_face(
                    _video, extracted_images_dir, _meta_data
                )
            )(str(root_dir / video), meta_data)
            for video, meta_data in tqdm(
                all_meta_data_filtered.items(), desc=f"folder_{folder_number}"
            )
        )
        # i = 0
        # for video, meta_data in tqdm(
        #     all_meta_data_filtered.items(), desc=f"folder_{folder_number}"
        # ):
        #     extract_first_face(str(root_dir / video), extracted_images_dir, meta_data)
        #
        #     i += 1
        #     if i == 20:
        #         break


if __name__ == "__main__":
    extract_images()
