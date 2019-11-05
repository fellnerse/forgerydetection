import ast
from pathlib import Path

import click
from pytorch_lightning import Trainer

from forgery_detection.lightning.system import Supervised
from forgery_detection.lightning.utils import get_latest_checkpoint


class PythonLiteralOption(click.Option):
    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except ValueError:
            raise click.BadParameter(value)


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
@click.option("--gpus", cls=PythonLiteralOption, default="[3]")
def run_lightning_test(*args, **kwargs):
    gpus = None if len(kwargs["gpus"]) == 0 else kwargs["gpus"]

    checkpoint_folder = Path(kwargs["checkpoint_dir"]) / "checkpoints"

    model = Supervised.load_from_metrics(
        weights_path=get_latest_checkpoint(checkpoint_folder),
        tags_csv=Path(kwargs["checkpoint_dir"]) / "meta_tags.csv",
    )

    trainer = Trainer(
        gpus=gpus,
        default_save_path=kwargs["log_dir"],
        distributed_backend="ddp" if gpus and len(gpus) > 1 else None,
    )
    trainer.test(model)
