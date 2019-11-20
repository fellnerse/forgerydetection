from pathlib import Path
from typing import List

import click
import numpy as np

from forgery_detection.data.face_forensics import Compression
from forgery_detection.data.face_forensics import DataType
from forgery_detection.data.face_forensics import FaceForensicsDataStructure
from forgery_detection.data.face_forensics.splits import TEST
from forgery_detection.data.face_forensics.splits import TEST_NAME
from forgery_detection.data.face_forensics.splits import TRAIN
from forgery_detection.data.face_forensics.splits import TRAIN_NAME
from forgery_detection.data.face_forensics.splits import VAL
from forgery_detection.data.face_forensics.splits import VAL_NAME
from forgery_detection.data.utils import FileList
from forgery_detection.utils import cl_logger


def _get_min_sequence_length(source_dir_data_structure):
    min_length = -1

    for source_sub_dir in source_dir_data_structure.get_subdirs():
        for video_folder in sorted(source_sub_dir.iterdir()):
            number_of_frames = len(list(video_folder.glob("*.png")))
            if min_length == -1 or min_length > number_of_frames:
                min_length = number_of_frames

    return min_length


def _select_frames(nb_images: int, samples_per_video: int) -> List[int]:
    """Selects frames to take from video.

    Args:
        nb_images: length of video aka. number of frames in video
        samples_per_video: how many frames of this video should be taken. If this value
            is bigger then nb_images or -1, nb_images are taken.

    """
    if samples_per_video == -1 or samples_per_video > nb_images:
        selected_frames = range(nb_images)
    else:
        selected_frames = np.rint(
            np.linspace(1, nb_images, min(samples_per_video, nb_images)) - 1
        ).astype(int)
    return selected_frames


def _img_name_to_int(img: Path):
    return int(img.name.split(".")[0])


@click.command()
@click.option("--source_dir_root", required=True, type=click.Path(exists=True))
@click.option("--output_file", required=True, type=click.Path())
@click.option("--compressions", "-c", multiple=True, default=[Compression.c40])
@click.option("--data_types", "-d", multiple=True, default=[DataType.face_images])
@click.option(
    "--samples_per_video",
    "-s",
    default=-1,
    help="Number of frames selected per video. For videos with less frames then this"
    "number, only these are selected. If samples_per_video is -1 all frames for each"
    "video is selected.",
)
@click.option(
    "--min_sequence_length",
    default=1,
    help="Indicates how many preceeded consecutive frames make a frame eligible (i.e."
    "if set to 5 frame 0004 is eligible if frames 0000-0003 are present as well.",
)
def create_file_list(
    source_dir_root,
    output_file,
    compressions,
    data_types,
    samples_per_video,
    min_sequence_length,
):
    file_list = FileList(
        root=source_dir_root, classes=FaceForensicsDataStructure.METHODS
    )

    # use faceforensicsdatastructure to iterate elegantly over the correct
    # image folders
    source_dir_data_structure = FaceForensicsDataStructure(
        source_dir_root, compressions=compressions, data_types=data_types
    )

    _min_sequence_length = _get_min_sequence_length(source_dir_data_structure)

    if _min_sequence_length < samples_per_video:
        cl_logger.warning(
            f"There is a sequence that is sequence that has less frames "
            f"then you would like to sample: "
            f"{_min_sequence_length}<{samples_per_video}"
        )

    for split, split_name in [(TRAIN, TRAIN_NAME), (VAL, VAL_NAME), (TEST, TEST_NAME)]:
        for source_sub_dir, target in zip(
            source_dir_data_structure.get_subdirs(), file_list.classes
        ):
            for video_folder in sorted(source_sub_dir.iterdir()):
                if video_folder.name.split("_")[0] in split:

                    images = sorted(video_folder.glob("*.png"))
                    filtered_images = []

                    sequence_start = _img_name_to_int(images[0])
                    last_idx = sequence_start

                    for image in images:
                        image_idx = _img_name_to_int(image)
                        if last_idx + 1 != image_idx:
                            sequence_start = image_idx
                        elif image_idx - sequence_start >= min_sequence_length - 1:
                            filtered_images.append(image)
                        last_idx = image_idx

                    # for the test-set all frames are going to be taken
                    selected_frames = _select_frames(
                        len(filtered_images),
                        -1 if split_name == TEST_NAME else samples_per_video,
                    )

                    for idx in selected_frames:
                        file_list.add_data_point(
                            path=filtered_images[idx],
                            target_label=target,
                            split=split_name,
                        )

    file_list.save(output_file)
    cl_logger.info(f"{output_file} created.")
    for split in [TRAIN_NAME, VAL_NAME, TEST_NAME]:
        data_set = FileList.get_dataset_form_file(output_file, split)
        cl_logger.info(f"{split}-data-set: {data_set}")


if __name__ == "__main__":
    create_file_list()
