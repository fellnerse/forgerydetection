import multiprocessing as mp
from pathlib import Path

import click
from joblib import delayed
from joblib import Parallel
from PIL.Image import Image
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Compose
from tqdm import tqdm

from forgery_detection.data.face_forensics.splits import TEST_NAME
from forgery_detection.data.face_forensics.splits import TRAIN_NAME
from forgery_detection.data.face_forensics.splits import VAL_NAME
from forgery_detection.data.set import FileList
from forgery_detection.data.utils import resized_crop


def resize(old_root, img_path, new_root, transform):
    img = default_loader(old_root / img_path)
    img: Image = transform(img)
    new_path = new_root / img_path
    new_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(new_path)


@click.command()
@click.option("--size", default=224)
@click.option("--source_file_list", type=click.Path(exists=True))
@click.option("--target_file_list", type=click.Path(exists=False))
@click.option("--target_dataset_folder", type=click.Path(exists=False))
def transform_dataset(size, source_file_list, target_dataset_folder, target_file_list):
    f = FileList.load(source_file_list)
    old_root = Path(f.root)
    new_root = Path(target_dataset_folder)
    new_root.mkdir(exist_ok=False)

    transform = Compose(resized_crop(size))

    for split in [TRAIN_NAME, VAL_NAME, TEST_NAME]:
        Parallel(n_jobs=mp.cpu_count())(
            delayed(lambda sample: resize(old_root, sample[0], new_root, transform))(
                sample_
            )
            for sample_ in tqdm(f.samples[split])
        )

    f.root = str(new_root)
    f.save(target_file_list)


if __name__ == "__main__":
    transform_dataset()
