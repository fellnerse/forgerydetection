import logging
from pathlib import Path

import click

from forgery_detection.data.face_forensics.splits import TRAIN_NAME
from forgery_detection.data.set import FileList
from forgery_detection.data.set import SimpleFileList

logger = logging.getLogger(__file__)


def _create_file_list(output_file: str, source_dir_root: str):
    file_list = SimpleFileList(root=source_dir_root)

    source_dir_root = Path(source_dir_root)

    for feature in source_dir_root.glob("*.pckl"):
        file_list.files[str(feature.with_suffix(".mp4").name)] = str(
            feature.relative_to(file_list.root)
        )

    file_list.save(output_file)
    logger.info(f"{output_file} created.")
    return file_list


@click.command()
@click.option("--source_dir_root", required=True, type=click.Path(exists=True))
@click.option("--output_file", required=True, type=click.Path())
def create_file_list(source_dir_root, output_file):
    try:
        # if file exists, we don't have to create it again
        FileList.load(output_file)
    except FileNotFoundError:
        file_list = _create_file_list(output_file, source_dir_root)
        file_list.save(output_file)

    data_set = SimpleFileList.load(output_file)
    logger.info(f"{TRAIN_NAME}-data-set: {data_set}")


if __name__ == "__main__":
    create_file_list()
