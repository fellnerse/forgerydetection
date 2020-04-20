import logging

import click

from forgery_detection.lightning.logging.utils import PythonLiteralOptionGPUs
from forgery_detection.lightning.utils import get_model_and_trainer

logger = logging.getLogger(__file__)


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
    default="/log",
)
@click.option("--dataset_percent_check -p", type=float, default=0.1)
@click.option("--gpus", cls=PythonLiteralOptionGPUs, default="[0]")
@click.option("--debug", is_flag=True)
def run_train_val_evaluation(**kwargs):

    # train data
    model, trainer = get_model_and_trainer(test_percent_check=0.01, **kwargs)
    _logger = model.logger

    model.test_dataloader = model.train_dataloader

    trainer.test(model)

    # val data
    model, trainer = get_model_and_trainer(
        test_percent_check=0.01, _logger=_logger, **kwargs
    )
    model.test_dataloader = model.val_dataloader

    trainer.test(model)


if __name__ == "__main__":
    run_train_val_evaluation()
