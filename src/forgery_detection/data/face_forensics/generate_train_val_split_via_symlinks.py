from pathlib import Path

import click

from forgery_detection.data.face_forensics import FaceForensicsDataStructure
from forgery_detection.data.face_forensics.splits import TEST
from forgery_detection.data.face_forensics.splits import TEST_NAME
from forgery_detection.data.face_forensics.splits import TRAIN
from forgery_detection.data.face_forensics.splits import TRAIN_NAME
from forgery_detection.data.face_forensics.splits import VAL
from forgery_detection.data.face_forensics.splits import VAL_NAME


def _symlink_split(source_dir_method, target_dir, split):
    for video in source_dir_method.iterdir():
        if video.is_dir():
            target_dir.mkdir(parents=True, exist_ok=True)
            if video.name.split("_")[0] in split:
                (target_dir / video.name).symlink_to(video, target_is_directory=True)


@click.command()
@click.option("--source_dir_root", required=True, type=click.Path(exists=True))
@click.option("--target_dir_root", required=True)
@click.option("--compression", default="c40")
def symlink_train_val_test_split(source_dir_root, target_dir_root, compression):

    # use faceforensicsdatastructure to iterate elegantly over the correct image folders
    source_dir_data_structure = FaceForensicsDataStructure(
        source_dir_root, compression=compression, data_type="images"
    )
    # target_dir_root will contain a train, val and test folder
    target_dir_root = Path(target_dir_root)

    # for each split and each subdirectory (aka. method, like deepfakes) we are
    # collecting the correct videos and symlinking them
    for split, split_name in [(TRAIN, TRAIN_NAME), (VAL, VAL_NAME), (TEST, TEST_NAME)]:

        target_dir_split = target_dir_root / split_name
        target_dir_split.mkdir(parents=True)

        target_dir_data_structure = FaceForensicsDataStructure(
            target_dir_split, compression=compression, data_type="images"
        )

        for source_sub_dir, target_sub_dir in zip(
            source_dir_data_structure.get_subdirs(),
            target_dir_data_structure.get_subdirs(),
        ):
            _symlink_split(source_sub_dir, target_sub_dir, split)


if __name__ == "__main__":
    symlink_train_val_test_split()
