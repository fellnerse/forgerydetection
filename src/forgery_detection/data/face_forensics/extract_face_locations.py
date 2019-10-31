import json
from pathlib import Path

import click
import cv2
import dlib
from joblib import delayed
from joblib import Parallel
from tqdm import tqdm

from forgery_detection.data.face_forensics import FaceForensicsDataStructure


# https://github.com/ondyari/FaceForensics/
# blob/master/classification/detect_from_video.py
def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def _find_face(_img, detector):
    img = cv2.imread(str(_img))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_img, 1)
    try:
        face = faces[0]
        height, width, _ = img.shape
        x, y, size = get_boundingbox(face, width, height, scale=1.3)
        # cropped_face = img[y : y + size, x : x + size]
        # cv2.imwrite("cropped_face.png", cropped_face)
        return _img.name, [x, y, size]
    except IndexError:
        return _img.name, []


def _extract_face_locations_from_video(video_folder, face_dir) -> bool:

    video_face_dir = face_dir / video_folder.name
    # if the folder already exists just continue
    try:
        video_face_dir.mkdir(exist_ok=False)
    except FileExistsError:
        return False

    # extract all faces and save it
    detector = dlib.get_frontal_face_detector()
    faces = dict(_find_face(img, detector) for img in video_folder.iterdir())

    with open(video_face_dir / "faces.json", "w") as fb:
        json.dump(faces, fb)

    return True


@click.command()
@click.option("--data_dir_root", required=True, type=click.Path(exists=True))
@click.option("--compression", default="raw")
def extract_face_locations(data_dir_root, compression):
    source_dir_data_structure = FaceForensicsDataStructure(
        data_dir_root, compression=compression, data_type="images"
    )
    # iterate over all manipulation methods and original videos
    methods = tqdm(source_dir_data_structure.get_subdirs(), position=0, leave=False)
    for method in methods:
        methods.set_description(f"Current method: {method.parents[1].name}")
        # add a face folder next to images and videos
        face_dir: Path = method.parent / "faces"
        face_dir.mkdir(exist_ok=True)

        # extract face locations from videos in parallel
        Parallel(n_jobs=12)(
            delayed(
                lambda _video_folder: _extract_face_locations_from_video(
                    _video_folder, face_dir
                )
            )(video_folder)
            for video_folder in tqdm(sorted(method.iterdir()), position=1, leave=False)
        )


if __name__ == "__main__":
    extract_face_locations()
