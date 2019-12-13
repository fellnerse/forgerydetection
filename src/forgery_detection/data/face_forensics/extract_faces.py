import json

import click
import cv2
from joblib import delayed
from joblib import Parallel
from tqdm import tqdm

from forgery_detection.data.face_forensics import Compression
from forgery_detection.data.face_forensics import DataType
from forgery_detection.data.face_forensics import FaceForensicsDataStructure


def _extract_face(img_path, face, face_images_dir):
    if len(face) == 0:
        return False
    img = cv2.imread(str(img_path))
    x, y, h, w = face
    cropped_face = img[y : y + w, x : x + h]  # noqa E203
    cv2.imwrite(str(face_images_dir / img_path.name), cropped_face)
    return True


def _extract_faces_from_video(video_folder, face_locations_dir) -> bool:
    face_images_dir = face_locations_dir / video_folder.with_suffix("").name
    bb_file = face_images_dir / "tracked_bb.json"

    if not bb_file.exists():
        raise ValueError(f"{bb_file} does not exist")

    # 377
    with open(bb_file, "r") as f:
        tracked_bb = json.load(f)

    cap = cv2.VideoCapture(str(video_folder))

    frame_num = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        try:
            if tracked_bb[f"{frame_num:04d}"]:
                x, y, w, h = tracked_bb[f"{frame_num:04d}"]
                image = image[y : y + h, x : x + w]  # noqa E203
                img_path = str(face_images_dir / f"{frame_num:04d}.png")
                cv2.imwrite(img_path, image)
        except IndexError:
            print(
                f"{frame_num} is out of bounds for {len(tracked_bb)}\n"
                f"Apperently there are not enough bbs for the video."
            )
            break
        frame_num += 1
    cap.release()

    return True


@click.command()
@click.option("--data_dir_root", required=True, type=click.Path(exists=True))
@click.option("--compression_face_locations", default=Compression.c40)
@click.option("--compression_data", default=Compression.c40)
def extract_faces(data_dir_root, compression_face_locations, compression_data):
    data_dir_data_structure = FaceForensicsDataStructure(
        data_dir_root,
        compressions=compression_data,
        data_types=DataType.resampled_videos,
    )

    face_locations_dir_data_structure = FaceForensicsDataStructure(
        data_dir_root,
        compressions=compression_face_locations,
        data_types=DataType.face_images_tracked,
    )

    # iterate over all manipulation methods and original videos
    methods = tqdm(
        zip(
            data_dir_data_structure.get_subdirs(),
            face_locations_dir_data_structure.get_subdirs(),
        ),
        position=0,
        leave=False,
    )
    for data_dir, face_locations_dir in methods:
        methods.set_description(f"Current method: {data_dir.parents[1].name}")

        # extract faces from videos in parallel
        Parallel(n_jobs=12)(
            delayed(
                lambda _video_folder: _extract_faces_from_video(
                    _video_folder, face_locations_dir
                )
            )(video_folder)
            for video_folder in tqdm(
                sorted(data_dir.iterdir()), position=1, leave=False
            )
        )


if __name__ == "__main__":
    extract_faces()
