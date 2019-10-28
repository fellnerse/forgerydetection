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


@click.command()
@click.option("--data_dir_root", required=True, type=click.Path(exists=True))
@click.option("--compression", default="raw")
def extract_faces(data_dir_root, compression):
    source_dir_data_structure = FaceForensicsDataStructure(
        data_dir_root, compression=compression, data_type="images"
    )

    detector = dlib.get_frontal_face_detector()
    for subdir in tqdm(
        source_dir_data_structure.get_subdirs(), position=0, leave=False
    ):
        face_dir: Path = subdir.parent / "faces"
        face_dir.mkdir(exist_ok=True)
        # (face_dir / "faces.json").touch()
        faces_dict = {}
        for video_folder in tqdm(list(subdir.iterdir()), position=1, leave=False):
            faces = Parallel(n_jobs=12)(
                delayed(
                    lambda img: _find_face(img, detector, faces_dict, video_folder)
                )(img)
                for img in tqdm(list(video_folder.iterdir()), position=2, leave=False)
            )
            faces_dict[video_folder.name] = {face[0]: face[1] for face in faces}
            # for _img in tqdm(list(video_folder.iterdir()), position=2, leave=False):
            #     _find_face(_img, detector, faces_dict, video_folder)
        with open(face_dir / "faces.json", "w") as fb:
            json.dump(faces_dict, fb)


def _find_face(_img, detector, faces_dict, video_folder):
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
        print("didn't find anything")
        return _img.name, []


if __name__ == "__main__":
    extract_faces()
