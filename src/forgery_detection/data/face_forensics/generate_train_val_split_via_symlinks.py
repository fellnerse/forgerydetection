import shutil
from pathlib import Path

import click


def _symlink_folders_and_do_val_split(
    source_dir: Path,
    target_dir: Path,
    target_val_dir: Path,
    compression="raw",
    val_percentage=0.2,
):
    target_dir.mkdir(parents=True, exist_ok=True)
    target_val_dir.mkdir(parents=True, exist_ok=True)

    for manipulation_method in source_dir.iterdir():
        if manipulation_method.is_dir():
            images_folder = manipulation_method / compression / "images"

            if images_folder.exists():
                print("processing", manipulation_method)
                (target_dir / manipulation_method.name).mkdir(parents=True)
                for video_folder in images_folder.iterdir():
                    if video_folder.is_dir():
                        symlink_dir = (
                            target_dir / manipulation_method.name / video_folder.name
                        )
                        symlink_dir.symlink_to(video_folder, target_is_directory=True)
                _move_symlinks(
                    target_dir / manipulation_method.name,
                    target_val_dir,
                    val_percentage,
                )
            else:
                print("skipping", manipulation_method)


def _move_symlinks(source: Path, target: Path, percentage=0.2):
    """Move some percentage of videos from one folder to another folder.

    Internally the alphabetically-sorted-last {percentage} videos are moved over.

    Args:
        source: source folder
        target: target folder
        percentage: len(items in source folder)*{precentage} are moved.

    """

    folders = list(source.glob("*"))
    folders.sort()
    num_val_videos = int(len(folders) * percentage + 0.5)
    (target / source.name).mkdir()
    try:
        for i in range(num_val_videos):
            shutil.move(
                str(folders[-1 - i]), str(target / source.name / folders[-1 - i].name)
            )
    except OSError as e:
        print("probably the permissions of the source folder are read only: ", e)


@click.command()
@click.option("--source_dir", required=True, type=click.Path(exists=True))
@click.option("--target_dir", required=True)
@click.option("--compression", default="c40")
@click.option("--val_percentage", default=0.2)
def symlink_all_folders_and_do_val_split(
    source_dir, target_dir, compression, val_percentage
):
    """Symlinks folders from source dir to target dir.

    source dir has to follow faceforensics folder structure:

    data:
        - manipulated_sequences:
            - a:
                - {compression}:
                    - images
            - b:
                - {compression}:
                    - images
            ...
        - original_sequences:
            - a:
                - {compression}:
                    - images
            - b:
                - {compression}:
                    - images
            ...

    Will do a train val split, by moving the last video of each method to a val folder.

    resulting structure:


    """
    source_dir = Path(source_dir)
    target_dir_val = Path(target_dir) / "val"
    target_dir_train = Path(target_dir) / "train"

    if not source_dir.exists():
        raise FileNotFoundError(f"{source_dir} does not exist")

    manipulated_sequences_source = source_dir / "manipulated_sequences"
    manipulated_sequences_target = target_dir_train / "manipulated_sequences"
    _symlink_folders_and_do_val_split(
        manipulated_sequences_source,
        manipulated_sequences_target,
        target_dir_val / "manipulated_sequences",
        compression=compression,
        val_percentage=val_percentage,
    )

    original_sequences_source = source_dir / "original_sequences"
    original_sequences_target = target_dir_train / "original_sequences"
    _symlink_folders_and_do_val_split(
        original_sequences_source,
        original_sequences_target,
        target_dir_val / "original_sequences",
        compression=compression,
        val_percentage=val_percentage,
    )


if __name__ == "__main__":
    symlink_all_folders_and_do_val_split()
