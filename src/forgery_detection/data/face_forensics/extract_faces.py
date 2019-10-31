import json
from pathlib import Path

import click
import cv2
from joblib import delayed
from joblib import Parallel
from tqdm import tqdm

from forgery_detection.data.face_forensics import FaceForensicsDataStructure


def _extract_face(img_path, face, face_images_dir):
    if len(face) == 0:
        return False
    img = cv2.imread(str(img_path))
    x, y, size = face
    cropped_face = img[y : y + size, x : x + size]  # noqa E203
    cv2.imwrite(str(face_images_dir / img_path.name), cropped_face)
    return True


def _extract_faces_from_video(video_folder, data_face_dir, face_locations_dir) -> bool:
    with open(face_locations_dir / video_folder.name / "faces.json", "r") as f:
        faces = json.load(f)

    face_images_dir = data_face_dir / video_folder.name
    # if the folder already exists just continue
    try:
        face_images_dir.mkdir(exist_ok=False)
    except FileExistsError:
        return False

    # extract all faces and save it
    for img in sorted(video_folder.iterdir()):
        face = faces[img.name]
        _extract_face(img, face, face_images_dir)

    return True


@click.command()
@click.option("--data_dir_root", required=True, type=click.Path(exists=True))
@click.option("--compression_face_locations", default="raw")
@click.option("--compression_data", default="c40")
def extract_faces(data_dir_root, compression_face_locations, compression_data):
    data_dir_data_structure = FaceForensicsDataStructure(
        data_dir_root, compression=compression_data, data_type="images"
    )

    face_locations_dir_data_structure = FaceForensicsDataStructure(
        data_dir_root, compression=compression_face_locations, data_type="faces"
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

        # add a face_images folder next to images and videos
        data_face_dir: Path = data_dir.parent / "face_images"
        data_face_dir.mkdir(exist_ok=True)

        # extract faces from videos in parallel
        Parallel(n_jobs=12)(
            delayed(
                lambda _video_folder: _extract_faces_from_video(
                    _video_folder, data_face_dir, face_locations_dir
                )
            )(video_folder)
            for video_folder in tqdm(
                sorted(data_dir.iterdir()), position=1, leave=False
            )
        )


if __name__ == "__main__":
    extract_faces()
