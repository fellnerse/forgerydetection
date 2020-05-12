import logging
from copy import deepcopy
from pathlib import Path

import click
from pytorch_lightning import Trainer
from torch.utils.data import SequentialSampler

from forgery_detection.data.misc.evaluate_outputs_binary import (
    print_evaluation_for_test_folder,
)
from forgery_detection.lightning.logging.const import SystemMode
from forgery_detection.lightning.logging.utils import (
    backwards_compatible_get_checkpoint,
)
from forgery_detection.lightning.logging.utils import get_logger_and_checkpoint_callback
from forgery_detection.lightning.logging.utils import get_logger_dir
from forgery_detection.lightning.logging.utils import PythonLiteralOptionGPUs
from forgery_detection.lightning.system import Supervised
from forgery_detection.lightning.utils import get_model_and_trainer

logger = logging.getLogger(__file__)


def print_google_sheet_ready_output(_logger):
    logger_dir = get_logger_dir(_logger)

    print_evaluation_for_test_folder(logger_dir)


@click.command()
@click.option(
    "--checkpoint_dir",
    required=True,
    type=click.Path(exists=True),
    help="Folder containing logs and checkpoint.",
)
@click.option("--checkpoint_nr", type=int, default=-1)
@click.option(
    "--log_dir",
    required=True,
    type=click.Path(exists=True),
    help="Folder used for logging.",
    default="/mnt/raid/sebastian/log",
)
@click.option("--train_percent_check", "-tp", type=float, default=0.17479461632)
@click.option("--batch_size", default=256, help="Path to data to validate on")
@click.option("--gradient_accumulation_nb", default=1)
@click.option("--num_runs", default=1, help="Indicates how often you want to run this.")
@click.option("--gpus", cls=PythonLiteralOptionGPUs, default="[0]")
@click.option("--debug", is_flag=True)
def run_fine_tuning(train_percent_check, num_runs, gradient_accumulation_nb, **kwargs):

    kwargs["mode"] = SystemMode.TRAIN
    kwargs["optimizer"] = "sgd"
    kwargs["lr"] = 0.0013 / 10.0
    _kwargs = deepcopy(kwargs)

    for i in range(num_runs):
        kwargs = deepcopy(_kwargs)
        checkpoint_folder = Path(kwargs["checkpoint_dir"])

        model: Supervised = Supervised.load_from_metrics(
            weights_path=backwards_compatible_get_checkpoint(
                checkpoint_folder, kwargs["checkpoint_nr"]
            ),
            tags_csv=Path(kwargs["checkpoint_dir"]) / "meta_tags.csv",
            overwrite_hparams=kwargs,
        )

        # Logging and Checkpoints
        checkpoint_callback, _logger = get_logger_and_checkpoint_callback(
            kwargs["log_dir"],
            kwargs["mode"],
            kwargs["debug"],
            logger_info={
                "name": "finetune_" + model.hparams["logger/name"],
                "description": model.hparams["logger/description"],
            },
        )
        model.logger = _logger

        trainer = Trainer(
            gpus=kwargs["gpus"],
            logger=_logger,
            checkpoint_callback=checkpoint_callback,
            default_save_path=kwargs["log_dir"],
            distributed_backend="ddp"
            if kwargs["gpus"] and len(kwargs["gpus"]) > 1
            else None,
            train_percent_check=train_percent_check,
            val_percent_check=0.0001,
            val_check_interval=1.0,
            weights_summary=None,
            max_nb_epochs=1,
            accumulate_grad_batches=gradient_accumulation_nb,
        )
        # finetune
        trainer.fit(model)

        # validate
        kwargs["sampling_probs"] = None
        kwargs["checkpoint_dir"] = Path(checkpoint_callback.filepath).parent
        kwargs["checkpoint_nr"] = 0

        model, trainer = get_model_and_trainer(test_percent_check=1.0, **kwargs)

        model.sampler_cls = SequentialSampler
        model.test_dataloader = model.val_dataloader

        trainer.test(model)

        print("#####################")
        print_google_sheet_ready_output(model.logger)

        print(f"{kwargs['checkpoint_dir']} | ckpt_nr: {kwargs['checkpoint_nr']}")
        print("#####################")
        exit(-1)


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)

    run_fine_tuning()
