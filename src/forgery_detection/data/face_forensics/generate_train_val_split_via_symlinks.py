from pathlib import Path

import click

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

    source_dir_root = Path(source_dir_root)
    if not source_dir_root.exists():
        raise FileNotFoundError(f"{source_dir_root} does not exist")

    target_dir_root = Path(target_dir_root)

    sub_dirs = ["original_sequences/youtube"] + [
        "manipulated_sequences/" + manipulated_sequence
        for manipulated_sequence in [
            "Deepfakes",
            "Face2Face",
            "FaceSwap",
            "NeuralTextures",
        ]
    ]
    for split, split_name in [(TRAIN, TRAIN_NAME), (VAL, VAL_NAME), (TEST, TEST_NAME)]:
        for sub_dir in sub_dirs:
            _symlink_split(
                source_dir_root / sub_dir / compression / "images",
                target_dir_root / split_name / sub_dir / compression / "images",
                split,
            )


if __name__ == "__main__":
    symlink_train_val_test_split()
