import shutil
from pathlib import Path

import click
from joblib import delayed
from joblib import Parallel
from tqdm import tqdm

from forgery_detection.data.face_forensics import FaceForensicsDataStructure
from forgery_detection.data.face_forensics.splits import TEST
from forgery_detection.data.face_forensics.splits import TEST_NAME
from forgery_detection.data.face_forensics.splits import TRAIN
from forgery_detection.data.face_forensics.splits import TRAIN_NAME
from forgery_detection.data.face_forensics.splits import VAL
from forgery_detection.data.face_forensics.splits import VAL_NAME


def _symlink_or_copy_folder(copy, split, target_dir, video):
    if video.is_dir():
        target_dir.mkdir(parents=True, exist_ok=True)
        if video.name.split("_")[0] in split:
            if not copy:
                (target_dir / video.name).symlink_to(video, target_is_directory=True)
            else:
                try:
                    shutil.copytree(
                        str(video), str(target_dir / video.name), symlinks=True
                    )
                except FileExistsError:
                    pass


def _symlink_or_copy_split(source_dir_method, target_dir, split, copy=True):
    Parallel(n_jobs=12)(
        delayed(
            lambda _video_folder: _symlink_or_copy_folder(
                copy, split, target_dir, _video_folder
            )
        )(video_folder)
        for video_folder in tqdm(sorted(source_dir_method.iterdir()))
    )


@click.command()
@click.option("--source_dir_root", required=True, type=click.Path(exists=True))
@click.option("--target_dir_root", required=True)
@click.option("--compression", default="c40")
@click.option("--data_type", default="images")
@click.option("--copy", is_flag=True)
def symlink_or_copy_train_val_test_split(
    source_dir_root, target_dir_root, compression, data_type, copy
):

    # use faceforensicsdatastructure to iterate elegantly over the correct image folders
    source_dir_data_structure = FaceForensicsDataStructure(
        source_dir_root, compressions=compression, data_types=data_type
    )
    # target_dir_root will contain a train, val and test folder
    target_dir_root = Path(target_dir_root)

    # for each split and each subdirectory (aka. method, like deepfakes) we are
    # collecting the correct videos and symlinking them
    for split, split_name in [(TRAIN, TRAIN_NAME), (VAL, VAL_NAME), (TEST, TEST_NAME)]:

        target_dir_split = target_dir_root / split_name
        try:
            target_dir_split.mkdir(parents=True)
        except FileExistsError:
            print(f"Warning: {target_dir_split} already exists!!")

        target_dir_data_structure = FaceForensicsDataStructure(
            target_dir_split, compressions=compression, data_types=data_type
        )

        for source_sub_dir, target_sub_dir in zip(
            source_dir_data_structure.get_subdirs(),
            target_dir_data_structure.get_subdirs(),
        ):
            _symlink_or_copy_split(source_sub_dir, target_sub_dir, split, copy)


if __name__ == "__main__":
    symlink_or_copy_train_val_test_split()
