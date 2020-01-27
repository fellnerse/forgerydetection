import multiprocessing as mp

import click
from pytorch_lightning import Trainer

from forgery_detection.lightning.logging.utils import get_logger_and_checkpoint_callback
from forgery_detection.lightning.logging.utils import SystemMode
from forgery_detection.lightning.system import Supervised
from forgery_detection.lightning.utils import PythonLiteralOptionGPUs


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--data_dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to file list json.",
)
@click.option(
    "--audio_file",
    required=False,
    type=click.Path(exists=True),
    help="Path to np audio file containing.",
)
@click.option(
    "--log_dir",
    required=True,
    type=click.Path(exists=True),
    help="Folder used for logging.",
    default="/log",
)
@click.option("--lr", default=10e-5, help="Learning rate used by optimizer")
@click.option("--weight_decay", default=0.0, help="Weight-decay used by optimizer")
@click.option("--batch_size", default=256, help="Path to data to validate on")
@click.option(
    "--scheduler_patience", default=10, help="Patience of ReduceLROnPlateau scheduler"
)
@click.option("--gpus", cls=PythonLiteralOptionGPUs, default="[0]")
@click.option(
    "--model",
    type=click.Choice(Supervised.MODEL_DICT.keys()),
    default="resnet18multiclassdropout",
    help="Model that should be trained",
)
@click.option(
    "--resize_transforms",
    default="none",
    help="This resize transform is applied to train/val/test images",
)
@click.option(
    "--augmentation_transforms",
    default="none",
    help="Augmentation only applied to train images. "
    "Can be multiple if split with blank.",
)
# todo convert to absolute batches
@click.option(
    "--val_check_interval",
    default=0.02,
    help="If float, % of tng epoch. If int, check every n batch",
)
@click.option(
    "--dont_balance_data",
    is_flag=True,
    default=False,
    help="Indicates if the data distribution should be balanced/normalized."
    "Each class will be sampled with the same probability",
)
@click.option(
    "--class_weights",
    is_flag=True,
    help="Indicates if class weights should be used during loss calculation."
    "Same values as in --balance_data are used for the classes.",
)
@click.option(
    "--log_roc_values",
    is_flag=True,
    default=False,
    help="Indicates if roc values should be calculated and logged. Can slow down"
    "training immensely.",
)
@click.option(
    "--n_cpu",
    default=2,
    help="Number of cpus used for data loading."
    " -1 corresponds to using all cpus available.",
)
@click.option("--debug", is_flag=True)
def run_lightning(*args, **kwargs):
    kwargs["mode"] = SystemMode.TRAIN

    # Logging and Checkpoints
    checkpoint_callback, logger = get_logger_and_checkpoint_callback(
        kwargs["log_dir"], kwargs["mode"], kwargs["debug"]
    )

    kwargs["logger"] = {"name": logger.name, "description": logger.description}

    if kwargs["n_cpu"] == -1:
        kwargs["n_cpu"] = mp.cpu_count()

    model = Supervised(kwargs)

    # early stopping
    # somehow does not work any more, the logs it receives are from train and not from
    # val
    early_stopping_callback = None
    #     EarlyStopping(
    #     monitor=VAL_ACC, patience=1, verbose=True, mode="max"
    # )

    trainer = Trainer(
        gpus=kwargs["gpus"],
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stopping_callback,
        default_save_path=kwargs["log_dir"],
        val_percent_check=kwargs["val_check_interval"],
        val_check_interval=kwargs["val_check_interval"],
        distributed_backend="ddp"
        if kwargs["gpus"] and len(kwargs["gpus"]) > 1
        else None,
        weights_summary=None,
        max_nb_epochs=100,
    )
    trainer.fit(model)
