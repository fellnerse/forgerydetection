import json
from pathlib import Path
from typing import List

import dlib
import numpy as np
from cv2 import cv2
from tqdm import tqdm


def calculate_relative_bb(
    total_width: int, total_height: int, relative_bb_values: List[int]
):
    x, y, w, h = relative_bb_values

    size_bb = int(max(w, h) * 1.3)
    center_x, center_y = x + int(0.5 * w), y + int(0.5 * h)

    # Check for out of bounds, x-y lower left corner
    x = max(int(center_x - size_bb // 2), 0)
    y = max(int(center_y - size_bb // 2), 0)

    # Check for too big size for given x, y
    size_bb = min(total_width - x, size_bb)
    size_bb = min(total_height - y, size_bb)

    return x, y, size_bb, size_bb


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


def crop_and_show_images():
    show_images = True

    video_path = Path("/Users/sebastian/PycharmProjects/forgerydetection/722_458")
    with open(video_path / "relative_bb.json", "r") as f:
        relative_bb = json.load(f)

    shape_predictor = dlib.shape_predictor(
        "/Users/sebastian/PycharmProjects/forgerydetection/shape_predictor_68_face_landmarks.dat"
    )

    for image in tqdm(sorted(video_path.glob("*.png"))):
        img = cv2.imread(str(image))
        total_width, total_height = len(img[0]), len(img)

        # show video_crop image
        if show_images:
            cv2.imshow(str(image.name), img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        relative_bb_values = relative_bb[str(image.with_suffix("").name)]
        x, y, w, h = calculate_relative_bb(
            total_width, total_height, relative_bb_values
        )

        # show local bb_image
        new_image = img[y : y + w, x : x + h]
        if show_images:
            cv2.imshow(str(image.name), new_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # show aligned global_image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        box = dlib.rectangle(left=x, top=y, right=x + w, bottom=y + h)
        shape = shape_predictor(gray, box)
        shape = shape_to_np(shape)
        left_eye = np.mean(shape[36:42], axis=0)
        right_eye = np.mean(shape[42:48], axis=0)
        dx, dy = right_eye - left_eye
        # Calculate rotation angle with x & y component of the eyes vector
        angle = np.rad2deg(np.arctan2(dy, dx))
        center = np.mean(shape, axis=0)
        R = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
        aligned_image = cv2.warpAffine(img, R, (total_width, total_height))

        if show_images:
            cv2.imshow(str(image.name), aligned_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        aligned_image = aligned_image[y : y + w, x : x + h]

        if show_images:
            cv2.imshow(str(image.name), aligned_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    crop_and_show_images()
