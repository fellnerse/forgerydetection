import logging
from pathlib import Path

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
from forgery_detection.data.file_lists import FileList
from forgery_detection.data.utils import img_name_to_int
from forgery_detection.data.utils import select_frames

logger = logging.getLogger(__file__)


def _get_min_sequence_length(source_dir_data_structure):
    min_length = -1

    for source_sub_dir in source_dir_data_structure.get_subdirs():
        for video_folder in sorted(source_sub_dir.iterdir()):
            number_of_frames = len(list(video_folder.glob("*.png")))
            if min_length == -1 or min_length > number_of_frames:
                min_length = number_of_frames

    return min_length


def _create_file_list(
    compressions,
    data_types,
    min_sequence_length,
    output_file,
    samples_per_video,
    source_dir_root,
):
    file_list = FileList(
        root=source_dir_root,
        classes=FaceForensicsDataStructure.METHODS,
        min_sequence_length=min_sequence_length,
    )
    # use faceforensicsdatastructure to iterate elegantly over the correct
    # image folders
    source_dir_data_structure = FaceForensicsDataStructure(
        source_dir_root, compressions=compressions, data_types=data_types
    )

    _min_sequence_length = _get_min_sequence_length(source_dir_data_structure)
    if _min_sequence_length < samples_per_video:
        logger.warning(
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
                    filtered_images_idx = []

                    # find all frames that have at least min_sequence_length-1 preceeding
                    # frames
                    sequence_start = img_name_to_int(images[0])
                    last_idx = sequence_start
                    for list_idx, image in enumerate(images):
                        image_idx = img_name_to_int(image)
                        if last_idx + 1 != image_idx:
                            sequence_start = image_idx
                        elif image_idx - sequence_start >= min_sequence_length - 1:
                            filtered_images_idx.append(list_idx)
                        last_idx = image_idx

                    selected_frames = select_frames(
                        len(filtered_images_idx), samples_per_video
                    )

                    sampled_images_idx = np.asarray(filtered_images_idx)[
                        selected_frames
                    ]
                    file_list.add_data_points(
                        path_list=images,
                        target_label=target,
                        split=split_name,
                        sampled_images_idx=sampled_images_idx,
                    )

    file_list.save(output_file)
    logger.info(f"{output_file} created.")
    return file_list


@click.command()
@click.option("--source_dir_root", required=True, type=click.Path(exists=True))
@click.option(
    "--target_dir_root",
    default=None,
    help="If specified, all files in the filelist are copied over to this location",
)
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
    target_dir_root,
    output_file,
    compressions,
    data_types,
    samples_per_video,
    min_sequence_length,
):

    try:
        # if file exists, we don't have to create it again
        file_list = FileList.load(output_file)
    except FileNotFoundError:
        file_list = _create_file_list(
            compressions,
            data_types,
            min_sequence_length,
            output_file,
            samples_per_video,
            source_dir_root,
        )

    if target_dir_root:
        file_list.copy_to(Path(target_dir_root))
        file_list.save(output_file)

    for split in [TRAIN_NAME, VAL_NAME, TEST_NAME]:
        data_set = FileList.get_dataset_form_file(output_file, split)
        logger.info(f"{split}-data-set: {data_set}")


if __name__ == "__main__":
    create_file_list()
