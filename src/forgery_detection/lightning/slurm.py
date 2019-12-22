import numpy as np
import torch
from pytorch_lightning import Trainer
from test_tube import HyperOptArgumentParser
from test_tube.hpc import SlurmCluster

from forgery_detection.lightning.logging.utils import SystemMode
from forgery_detection.lightning.system import Supervised

parser = HyperOptArgumentParser(strategy="random_search")
parser.add_argument(
    "--data_dir",
    default="/data/ssd1/file_lists/c40/tracked_resampled_faces.json",
    type=str,
)
parser.add_argument("--audio_file", default=None, type=str)
# todo find way of changing this logdir via cli
parser.add_argument(
    "--log_dir", default="/log/test_slurm/20_lr_weight_decay_better_ranges", type=str
)
parser.add_argument("--batch_size", default=50, type=int)
parser.add_argument("--gpus", default=-1, type=int)
parser.add_argument("--model", default="resnet182d", type=str)
parser.add_argument("--transforms", default="none", type=str)
parser.add_argument("--val_check_interval", default=0.02, type=float)
parser.add_argument("--dont_balance_data", default=False)
parser.add_argument("--class_weights", default=False)
parser.add_argument("--log_roc_values", default=False)


parser.add_argument("--debug", default=False, type=bool)

parser.opt_list(
    "--lr",
    default=10e-5,
    type=float,
    help="the learning rate",
    tunable=True,
    options=np.power(10.0, -np.random.uniform(3, 7, size=10)),
)

parser.opt_list(
    "--weight_decay",
    default=0,
    type=float,
    help="the learning rate",
    tunable=True,
    options=np.power(10.0, -np.random.uniform(0.0, 10.0, size=10)),
)

hparams = parser.parse_args()


# init cluster
cluster = SlurmCluster(
    hyperparam_optimizer=hparams,
    log_path="/log/test_slurm",
    python_cmd="/home/sebastian/.cache/pypoetry/virtualenvs/forgerydetection-py3.7/"
    "bin/python",
)

# let the cluster know where to email for a change in job status
# (ie: complete, fail, etc...)
cluster.notify_job_status(email="fellnerseb@gmail.com", on_done=True, on_fail=True)

# set the job options. In this instance, we'll run 20 different models
# each with its own set of hyperparameters giving each one 1 GPU (ie: taking up 20 GPUs)
cluster.per_experiment_nb_gpus = 1
cluster.per_experiment_nb_nodes = 1

# we'll request 10GB of memory per node
cluster.memory_mb_per_node = 20000

# set a walltime of 10 minues
cluster.job_time = "30:00"
# cluster.job_time = "5:00"
cluster.minutes_to_checkpoint_before_walltime = 1


def train_fx(trial_hparams, cluster_manager):
    print("cuda_devices available", torch.cuda.device_count())
    trial_hparams = trial_hparams.__dict__
    trial_hparams["mode"] = SystemMode.TRAIN

    my_model = Supervised(trial_hparams)

    # give the trainer the cluster object
    trainer = Trainer(
        gpus=trial_hparams["gpus"],
        default_save_path=trial_hparams["log_dir"],
        val_percent_check=trial_hparams["val_check_interval"],
        val_check_interval=trial_hparams["val_check_interval"],
        weights_summary=None,
        early_stop_callback=None,
        max_nb_epochs=14,
    )
    trainer.fit(my_model)


# run the models on the cluster
cluster.optimize_parallel_cluster_gpu(
    train_fx,
    nb_trials=20,
    job_name="my_grid_search_exp_name",
    job_display_name="my_exp",
    enable_auto_resubmit=True,
)
