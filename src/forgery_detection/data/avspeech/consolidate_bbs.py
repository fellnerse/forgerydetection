import json
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple

import click
import numpy as np
from cv2 import cv2
from joblib import delayed
from joblib import Parallel
from tqdm import tqdm


def _calculate_tracking_bounding_box(
    face_bb: Dict[str, List[int]], image_size: Tuple[int, int], scale: int = 1.0
):
    height, width = image_size

    bounding_boxes = face_bb.values()
    left = min([bounding_box[0] for bounding_box in bounding_boxes])
    top = min([bounding_box[2] for bounding_box in bounding_boxes])
    right = max([bounding_box[1] for bounding_box in bounding_boxes])
    bottom = max([bounding_box[3] for bounding_box in bounding_boxes])

    x, y, w, h = left, top, right - left, bottom - top

    size_bb = int(max(w, h) * scale)

    center_x, center_y = x + int(0.5 * w), y + int(0.5 * h)

    # Check for out of bounds, x-y lower left corner
    x = max(int(center_x - size_bb // 2), 0)
    y = max(int(center_y - size_bb // 2), 0)

    # Check for too big size for given x, y
    size_bb = min(width - x, size_bb)
    size_bb = min(height - y, size_bb)

    relative_bb = {}
    for key in face_bb.keys():
        _x, _y, w, h = face_bb[key]
        relative_bb[key] = _x - x, _y - y, w, h
        face_bb[key] = [x, y, size_bb, size_bb]

    return relative_bb


def _face_bb_to_tracked_bb(
    face_bb: Dict[str, List[int]], image_size: Tuple[int, int], scale: int = 1.0
):
    current_sequence = {}
    tracked_bb = {}
    relative_bb = {}

    def calculate_tracked_bb_for_sequence(image_name):
        if len(current_sequence) > 0:
            relative_bb.update(
                _calculate_tracking_bounding_box(
                    current_sequence, image_size, scale=scale
                )
            )
            tracked_bb.update(current_sequence)
        if image_name:
            relative_bb[image_name] = None
            tracked_bb[image_name] = None

    for image_name, face_bb_value in face_bb.items():
        if not face_bb_value:
            calculate_tracked_bb_for_sequence(image_name)
            current_sequence = {}
        else:
            current_sequence[image_name] = face_bb_value

    if len(current_sequence) > 0:
        calculate_tracked_bb_for_sequence(None)
    return tracked_bb, relative_bb


def _calculate_items_to_delete(all_bbs_keys):
    actual_frames = np.arange(start=all_bbs_keys[0], stop=all_bbs_keys[-1], step=5)
    deleted_frames = sorted(set(actual_frames) - set(all_bbs_keys))

    if len(deleted_frames) == 0:
        return []

    ranges = []
    current_range = []
    for item in deleted_frames:
        if not current_range or current_range[-1] + 5 == item:
            current_range += [item]
        else:
            ranges += [current_range]
            current_range = []
    if current_range:
        ranges += [current_range]

    def _ranges_to_items():
        items_to_delete = []
        for _range in ranges:
            items_to_delete += list(range(_range[0] - 4, _range[-1] + 5))
        return items_to_delete

    return _ranges_to_items()


def _extract_face(img, face, face_images_dir, frame_number):
    if not face:
        return False
    x, y, w, h = face
    try:
        cropped_face = img[y : y + int(h), x : x + int(w)]  # noqa E203
    except TypeError:
        print(face)
        raise
    cv2.imwrite(str(face_images_dir / f"{frame_number:04d}.png"), cropped_face)
    return True


def _extract_correct_video_crops(
    all_bbs_video, interpolated_bbs_video, output_path, current_video
):
    output_path.mkdir(exist_ok=True)

    # remove entries that are None
    all_bbs_video = dict((k, v) for k, v in all_bbs_video.items() if len(v))
    all_bbs_keys = np.array(list(all_bbs_video.keys()), dtype=int)

    # interpolated_bbs_keys = np.array(list(interpolated_bbs_video.keys()), dtype=int)

    # when there are some keys missing this will be true
    if all_bbs_keys[-1] // 5 != len(all_bbs_keys):

        # delete all items that are in the ranges of not detected bbs
        items_to_delete = _calculate_items_to_delete(all_bbs_keys)
        for item in items_to_delete:
            interpolated_bbs_video[str(item)] = []

    cap = cv2.VideoCapture(str(current_video))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    tracked_bb, relative_bb = _face_bb_to_tracked_bb(
        interpolated_bbs_video, (height, width)
    )

    # read till first frame
    for i in range(0, all_bbs_keys[0]):
        _ = cap.grab()

    # extract all faces and save it
    frame_num = all_bbs_keys[0]
    while cap.isOpened() and frame_num <= all_bbs_keys[-1]:
        _ = cap.grab()
        success, image = cap.retrieve()
        if not success:
            break

        face = tracked_bb[f"{frame_num}"]
        _extract_face(image, face, output_path, frame_num)

        frame_num += 1
    cap.release()


def extract_faces_mp(
    all_bbs, i, interpolated_bbs, output_path, videos_crops, videos_path
):
    current_video_crop = videos_crops[i]
    name = current_video_crop.name
    current_video = (videos_path / name).with_suffix(".mp4")
    all_bbs_video = (all_bbs / name).with_suffix(".json")
    interpolated_bbs_video = (interpolated_bbs / name).with_suffix(".json")
    with open(all_bbs_video, "r") as f:
        all_bbs_video = json.load(f)
    with open(interpolated_bbs_video, "r") as f:
        interpolated_bbs_video = json.load(f)
    _extract_correct_video_crops(
        all_bbs_video=all_bbs_video,
        interpolated_bbs_video=interpolated_bbs_video,
        output_path=output_path / current_video_crop.name,
        current_video=current_video,
    )


@click.command()
@click.option("--videos_path", "-i", default="/mnt/avspeech/downloads/videos")
@click.option(
    "--extracted_path", "-e", default="/mnt/avspeech_extracted/dlib_extracted"
)
@click.option("--output_path", "-e", default="/data/hdd/avspeech_consolidated_bbs/test")
@click.option("--num_threads", type=int, default=1)
@click.option("--start", type=int, default=0)
@click.option("--end", type=int, default=10)  # 7000)
@click.option("--contains_string", type=str, default="000")
def extract_faces_tracked(
    videos_path, extracted_path, output_path, num_threads, start, end, contains_string
):
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    extracted_folder = Path(extracted_path)

    videos_path = Path(videos_path)

    videos_crops_folder = extracted_folder / "video_crops"
    videos_crops = sorted(videos_crops_folder.glob("*" + contains_string + "*"))

    all_bbs = extracted_folder / "full_dlib_bounding_boxes"
    interpolated_bbs = extracted_folder / "dlib_bounding_box_tracking"

    # for i in tqdm(range(start, end)):
    #     extract_faces_mp(
    #         all_bbs, i, interpolated_bbs, output_path, videos_crops, videos_path
    #     )

    Parallel(n_jobs=num_threads)(
        delayed(
            lambda _i: extract_faces_mp(
                all_bbs, _i, interpolated_bbs, output_path, videos_crops, videos_path
            )
        )(i)
        for i in tqdm(range(start, end))
    )


if __name__ == "__main__":
    extract_faces_tracked()
