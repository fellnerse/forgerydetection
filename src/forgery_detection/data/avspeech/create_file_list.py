import logging
from pathlib import Path

import click
import numpy as np

from forgery_detection.data.avspeech import AVSPEECH_NAME
from forgery_detection.data.face_forensics.splits import TEST_NAME
from forgery_detection.data.face_forensics.splits import TRAIN_NAME
from forgery_detection.data.face_forensics.splits import VAL_NAME
from forgery_detection.data.file_lists import FileList
from forgery_detection.data.utils import img_name_to_int
from forgery_detection.data.utils import select_frames

logger = logging.getLogger(__file__)


def _create_file_list(
    min_sequence_length: int,
    output_file: str,
    samples_per_video: int,
    source_dir_root: str,
):
    file_list = FileList(
        root=source_dir_root,
        classes=[AVSPEECH_NAME],
        min_sequence_length=min_sequence_length,
    )

    source_dir_root = Path(source_dir_root)
    # split between train and val
    videos = sorted(source_dir_root.iterdir())
    train = videos[: int(len(videos) * 0.9)]
    val = videos[int(len(videos) * 0.9) :]

    for split, split_name in [(train, TRAIN_NAME), (val, VAL_NAME)]:
        for video_folder in sorted(split):
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

            selected_frames = select_frames(len(filtered_images_idx), samples_per_video)

            sampled_images_idx = np.asarray(filtered_images_idx)[selected_frames]

            file_list.add_data_points(
                path_list=images,
                target_label=AVSPEECH_NAME,
                split=split_name,
                sampled_images_idx=sampled_images_idx,
            )

    file_list.save(output_file)
    logger.info(f"{output_file} created.")
    return file_list


@click.command()
@click.option("--source_dir_root", required=True, type=click.Path(exists=True))
@click.option("--output_file", required=True, type=click.Path())
@click.option(
    "--samples_per_video",
    "-s",
    default=100,
    help="Number of frames selected per video. For videos with less frames then this"
    "number, only these are selected. If samples_per_video is -1 all frames for each"
    "video is selected.",
)
@click.option(
    "--min_sequence_length",
    default=8,
    help="Indicates how many preceeded consecutive frames make a frame eligible (i.e."
    "if set to 5 frame 0004 is eligible if frames 0000-0003 are present as well.",
)
def create_file_list(
    source_dir_root, output_file, samples_per_video, min_sequence_length
):
    try:
        # if file exists, we don't have to create it again
        FileList.load(output_file)
    except FileNotFoundError:
        file_list = _create_file_list(
            min_sequence_length, output_file, samples_per_video, source_dir_root
        )
        file_list.save(output_file)

    for split in [TRAIN_NAME, VAL_NAME, TEST_NAME]:
        data_set = FileList.get_dataset_form_file(output_file, split)
        logger.info(f"{split}-data-set: {data_set}")


if __name__ == "__main__":
    create_file_list()
