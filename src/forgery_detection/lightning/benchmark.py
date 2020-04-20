import json
from pathlib import Path

import click
import torch

from forgery_detection.lightning.logging.const import SystemMode
from forgery_detection.lightning.logging.utils import get_checkpoint
from forgery_detection.lightning.logging.utils import PythonLiteralOptionGPUs
from forgery_detection.lightning.system import Supervised


@click.command()
@click.option(
    "--checkpoint_dir",
    required=True,
    type=click.Path(exists=True),
    help="Folder containing logs and checkpoint.",
)
@click.option(
    "--log_dir",
    required=True,
    type=click.Path(exists=True),
    help="Folder used for logging.",
)
@click.option(
    "--benchmark_dir",
    required=True,
    type=click.Path(exists=True),
    help="Folder containing images.",
)
@click.option("--gpus", cls=PythonLiteralOptionGPUs, default="[3]")
def run_benchmark(*args, **kwargs):
    kwargs["mode"] = SystemMode.BENCHMARK

    checkpoint_folder = Path(kwargs["checkpoint_dir"]) / "checkpoints"

    model = Supervised.load_from_metrics(
        weights_path=get_checkpoint(checkpoint_folder),
        tags_csv=Path(kwargs["checkpoint_dir"]) / "meta_tags.csv",
    )
    device = torch.device("cuda", kwargs["gpus"][0])

    predictions_dict = model.benchmark(
        benchmark_dir=kwargs["benchmark_dir"], device=device, threshold=0.05
    )
    with open(checkpoint_folder / "submission.json", "w") as f:
        json.dump(predictions_dict, f)
