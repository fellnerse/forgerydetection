import numpy as np
import torch
from pytorch_lightning import Trainer
from test_tube import HyperOptArgumentParser
from test_tube.hpc import SlurmCluster

from forgery_detection.lightning.logging.utils import SystemMode
from forgery_detection.lightning.system import Supervised

# todo find way of changing this logdir via cli
experiment_name = "50_data_augmentation_5_epochs"

parser = HyperOptArgumentParser(strategy="random_search")
parser.add_argument(
    "--data_dir",
    default="/data/ssd1/file_lists/c40/tracked_resampled_faces.json",
    type=str,
)
parser.add_argument("--audio_file", default=None, type=str)
parser.add_argument("--log_dir", default=f"/log/test_slurm/{experiment_name}", type=str)
parser.add_argument("--batch_size", default=50, type=int)
parser.add_argument("--gpus", default=-1, type=int)
parser.add_argument("--model", default="resnet182d", type=str)

parser.add_argument("--val_check_interval", default=0.02, type=float)
parser.add_argument("--dont_balance_data", default=False)
parser.add_argument("--class_weights", default=False)
parser.add_argument("--log_roc_values", default=False)

parser.add_argument("--debug", default=False, type=bool)

# parser.add_argument("--transforms", default="none", type=str)
parser.opt_list(
    "--transforms",
    default="none",
    type=str,
    tunable=True,
    options=[
        "none",
        "random_resized_crop",
        "random_horizontal_flip",
        "colour_jitter",
        "random_rotation",
        "random_greyscale",
        "random_erasing",
    ],
)

parser.opt_list(
    "--lr",
    default=10e-5,
    type=float,
    help="the learning rate",
    tunable=True,
    # options=np.power(10.0, -np.random.uniform(4, 6, size=10)),
    options=np.power(10.0, -np.linspace(4.0, 5.0, num=10)),
)

parser.opt_list(
    "--weight_decay",
    default=0,
    type=float,
    help="the learning rate",
    tunable=True,
    # options=np.power(10.0, -np.random.uniform(0.0, 10.0, size=10)),
    options=np.power(10.0, -np.linspace(4.0, 10.0, num=10)),
)

hparams = parser.parse_args()


# init cluster
cluster = SlurmCluster(
    hyperparam_optimizer=hparams,
    log_path="/log/test_slurm",
    python_cmd="/home/sebastian/.cache/pypoetry/virtualenvs/forgerydetection-py3.7/"
    "bin/python",
)

cluster.notify_job_status(email="fellnerseb@gmail.com", on_done=True, on_fail=True)

cluster.per_experiment_nb_gpus = 1
cluster.per_experiment_nb_nodes = 1
cluster.per_experiment_nb_cpus = 2

cluster.memory_mb_per_node = 20000

cluster.job_time = "2:00:00"
cluster.minutes_to_checkpoint_before_walltime = 1


def train_fx(trial_hparams, cluster_manager):
    print("cuda_devices available", torch.cuda.device_count())
    trial_hparams = trial_hparams.__dict__
    trial_hparams["mode"] = SystemMode.TRAIN

    my_model = Supervised(trial_hparams)

    trainer = Trainer(
        gpus=trial_hparams["gpus"],
        default_save_path=trial_hparams["log_dir"],
        val_percent_check=trial_hparams["val_check_interval"],
        val_check_interval=trial_hparams["val_check_interval"],
        weights_summary=None,
        early_stop_callback=None,
        max_nb_epochs=5,
    )
    trainer.fit(my_model)


# todo change stuff here as well
cluster.optimize_parallel_cluster_gpu(
    train_fx,
    nb_trials=50,
    job_name=experiment_name,
    job_display_name=experiment_name,
    enable_auto_resubmit=True,
)
