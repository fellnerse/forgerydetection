import logging
import pickle
from argparse import Namespace
from functools import partial
from pathlib import Path
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer.trainer_io import load_hparams_from_tags_csv
from sklearn.preprocessing import LabelBinarizer
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

from forgery_detection.data.face_forensics.splits import TEST_NAME
from forgery_detection.data.face_forensics.splits import TRAIN_NAME
from forgery_detection.data.face_forensics.splits import VAL_NAME
from forgery_detection.data.loading import BalancedSampler
from forgery_detection.data.loading import calculate_class_weights
from forgery_detection.data.loading import get_fixed_dataloader
from forgery_detection.data.loading import get_sequence_collate_fn
from forgery_detection.data.loading import SequenceBatchSampler
from forgery_detection.data.set import FileList
from forgery_detection.data.utils import colour_jitter
from forgery_detection.data.utils import crop
from forgery_detection.data.utils import get_data
from forgery_detection.data.utils import random_erasing
from forgery_detection.data.utils import random_flip_greyscale
from forgery_detection.data.utils import random_flip_rotation
from forgery_detection.data.utils import random_flip_rotation_greyscale
from forgery_detection.data.utils import random_greyscale
from forgery_detection.data.utils import random_horizontal_flip
from forgery_detection.data.utils import random_resized_crop
from forgery_detection.data.utils import random_rotation
from forgery_detection.data.utils import random_rotation_greyscale
from forgery_detection.data.utils import resized_crop
from forgery_detection.data.utils import resized_crop_flip
from forgery_detection.lightning.logging.utils import DictHolder
from forgery_detection.lightning.logging.utils import get_logger_dir
from forgery_detection.lightning.logging.utils import log_confusion_matrix
from forgery_detection.lightning.logging.utils import log_dataset_preview
from forgery_detection.lightning.logging.utils import log_hparams
from forgery_detection.lightning.logging.utils import log_roc_graph
from forgery_detection.lightning.logging.utils import multiclass_roc_auc_score
from forgery_detection.lightning.logging.utils import SystemMode
from forgery_detection.lightning.utils import NAN_TENSOR
from forgery_detection.models.audio.multi_class_classification import AudioNet
from forgery_detection.models.audio.multi_class_classification import AudioOnly
from forgery_detection.models.image.multi_class_classification import ResidualResnet
from forgery_detection.models.image.multi_class_classification import Resnet182D
from forgery_detection.models.image.multi_class_classification import Resnet182d1Block
from forgery_detection.models.image.multi_class_classification import (
    Resnet182d1BlockFrozen,
)
from forgery_detection.models.image.multi_class_classification import Resnet182d2Blocks
from forgery_detection.models.image.multi_class_classification import (
    Resnet182d2BlocksFrozen,
)
from forgery_detection.models.image.multi_class_classification import Resnet182dFrozen
from forgery_detection.models.image.multi_class_classification import Resnet18Frozen
from forgery_detection.models.image.multi_class_classification import (
    Resnet18MultiClassDropout,
)
from forgery_detection.models.image.multi_class_classification import (
    Resnet18UntrainedMultiClassDropout,
)
from forgery_detection.models.image.vae import SimpleVAE
from forgery_detection.models.image.vae import SupervisedVae
from forgery_detection.models.utils import LightningModel
from forgery_detection.models.video.multi_class_classification import MC3
from forgery_detection.models.video.multi_class_classification import R2Plus1
from forgery_detection.models.video.multi_class_classification import R2Plus1Frozen
from forgery_detection.models.video.multi_class_classification import R2Plus1Small
from forgery_detection.models.video.multi_class_classification import Resnet183D
from forgery_detection.models.video.multi_class_classification import (
    Resnet183DNoDropout,
)
from forgery_detection.models.video.multi_class_classification import (
    Resnet183DUntrained,
)
from forgery_detection.models.video.multi_class_classification import Resnet18Fully3D
from forgery_detection.models.video.multi_class_classification import (
    Resnet18Fully3DPretrained,
)
from forgery_detection.models.video.vae import VideoAE
from forgery_detection.models.video.vae import VideoVae
from forgery_detection.models.video.vae import VideoVaeDetachedSupervised
from forgery_detection.models.video.vae import VideoVaeSupervised
from forgery_detection.models.video.vae import VideoVaeSupervisedBCE
from forgery_detection.models.video.vae import VideoVaeUpsample
from forgery_detection.models.video.vae import VVVGGLoss

logger = logging.getLogger(__file__)


class Supervised(pl.LightningModule):
    MODEL_DICT = {
        "resnet182d": Resnet182D,
        "resnet182d2blocks": Resnet182d2Blocks,
        "resnet182d1block": Resnet182d1Block,
        "resnet182d1blockfrozen": Resnet182d1BlockFrozen,
        "resnet182d2blocksfrozen": Resnet182d2BlocksFrozen,
        "resnet182dfrozen": Resnet182dFrozen,
        "resnet18frozen": Resnet18Frozen,
        "residualresnet": ResidualResnet,
        "resnet18multiclassdropout": Resnet18MultiClassDropout,
        "resnet18untrainedmulticlassdropout": Resnet18UntrainedMultiClassDropout,
        "resnet183d": Resnet183D,
        "resnet183duntrained": Resnet183DUntrained,
        "resnet183dnodropout": Resnet183DNoDropout,
        "resnet18fully3d": Resnet18Fully3D,
        "resnet18fully3dpretrained": Resnet18Fully3DPretrained,
        "r2plus1": R2Plus1,
        "r2plus1frozen": R2Plus1Frozen,
        "r2plus1small": R2Plus1Small,
        "mc3": MC3,
        "audionet": AudioNet,
        "audioonly": AudioOnly,
        "vae": SimpleVAE,
        "vae_supervised": SupervisedVae,
        "vae_video": VideoVae,
        "vae_video_upsample": VideoVaeUpsample,
        "vae_video_supervised": VideoVaeSupervised,
        "vae_video_detached_supervised": VideoVaeDetachedSupervised,
        "vae_video_supervised_bce": VideoVaeSupervisedBCE,
        "ae_video": VideoAE,
        "vae_vgg": VVVGGLoss,
    }

    CUSTOM_TRANSFORMS = {
        "none": [],
        "crop": crop(),
        "resized_crop": resized_crop(),
        "resized_crop_small": resized_crop(224),
        "resized_crop_128": resized_crop(128),
        "resized_crop_112": resized_crop(112),
        "resized_crop_56": resized_crop(56),
        "resized_crop_28": resized_crop(28),
        "resized_crop_14": resized_crop(14),
        "resized_crop_7": resized_crop(7),
        "resized_crop_flip": resized_crop_flip(),
        "random_resized_crop": random_resized_crop(112),
        "random_horizontal_flip": random_horizontal_flip(),
        "colour_jitter": colour_jitter(),
        "random_rotation": random_rotation(),
        "random_greyscale": random_greyscale(),
        "random_erasing": random_erasing(),
        "random_flip_rotation": random_flip_rotation(),
        "random_flip_greyscale": random_flip_greyscale(),
        "random_rotation_greyscale": random_rotation_greyscale(),
        "random_flip_rotation_greyscale": random_flip_rotation_greyscale(),
    }

    def _get_transforms(self, transforms: str):
        if " " not in transforms:
            return self.CUSTOM_TRANSFORMS[transforms]

        transform_list = transforms.split(" ")
        transforms = []
        for transform in transform_list:
            transforms.extend(self.CUSTOM_TRANSFORMS[transform])
        return transforms

    def __init__(self, kwargs: Union[dict, Namespace]):
        super(Supervised, self).__init__()

        self.hparams = DictHolder(kwargs)

        # load data-sets
        self.file_list = FileList.load(self.hparams["data_dir"])

        self.model: LightningModel = self.MODEL_DICT[self.hparams["model"]](
            num_classes=len(self.file_list.classes)
        )

        if len(self.file_list.classes) != self.model.num_classes:
            logger.error(
                f"Classes of model ({self.model.num_classes}) != classes of dataset"
                f" ({len(self.file_list.classes)})"
            )

        self.sampling_probs = self.hparams["sampling_probs"]
        if self.sampling_probs:
            self.sampling_probs = np.array(self.sampling_probs.split(" "), dtype=float)
        if self.sampling_probs is not None and len(self.file_list.classes) != len(
            self.sampling_probs
        ):
            raise ValueError(
                f"Classes of dataset ({len(self.file_list.classes)}) != classes of "
                f"sampling probs ({len(self.sampling_probs)})!"
            )

        resize_transform = self._get_transforms(self.hparams["resize_transforms"])
        augmentation_transform = self._get_transforms(
            self.hparams["augmentation_transforms"]
        )
        self.train_data = self.file_list.get_dataset(
            TRAIN_NAME,
            resize_transform + augmentation_transform,
            sequence_length=self.model.sequence_length,
            audio_file=self.hparams["audio_file"],
        )
        self.val_data = self.file_list.get_dataset(
            VAL_NAME,
            resize_transform,
            sequence_length=self.model.sequence_length,
            audio_file=self.hparams["audio_file"],
        )
        # handle empty test_data better
        self.test_data = self.file_list.get_dataset(
            TEST_NAME,
            resize_transform,
            sequence_length=self.model.sequence_length,
            audio_file=self.hparams["audio_file"],
        )
        self.hparams.add_dataset_size(len(self.train_data), TRAIN_NAME)
        self.hparams.add_dataset_size(len(self.val_data), VAL_NAME)
        self.hparams.add_dataset_size(len(self.test_data), TEST_NAME)

        # label_binarizer and positive class is needed for roc_auc-multiclass
        self.label_binarizer = LabelBinarizer()
        self.label_binarizer.fit(list(self.file_list.class_to_idx.values()))
        self.positive_class = list(self.file_list.class_to_idx.values())[-1]
        self.hparams["positive_class"] = self.positive_class

        if self.hparams["dont_balance_data"]:
            self.sampler_cls = RandomSampler
        else:
            self.sampler_cls = BalancedSampler

        self.system_mode = self.hparams.pop("mode")

        if self.system_mode is SystemMode.TRAIN:
            self.hparams.add_nb_trainable_params(self.model)
            if self.hparams["class_weights"]:
                labels, weights = calculate_class_weights(self.val_data)
                self.hparams.add_class_weights(labels, weights)
                self.class_weights = torch.tensor(weights, dtype=torch.float)
            else:
                self.class_weights = None

        elif self.system_mode is SystemMode.TEST:
            self.class_weights = None

        elif self.system_mode is SystemMode.BENCHMARK:
            pass

        # hparams logging
        self.decay = 0.95
        self.acc = -1
        self.loss = -1

    def on_sanity_check_start(self):
        log_hparams(
            hparam_dict=self.hparams.to_dict(),
            metric_dict={"metrics/acc": np.nan, "metrics/loss": np.nan},
            _logger=self.logger,
            global_step=0,
        )

        if not self.hparams["debug"] or True:
            log_dataset_preview(self.train_data, "preview/train_data", self.logger)
            log_dataset_preview(self.val_data, "preview/val_data", self.logger)
            try:
                log_dataset_preview(self.test_data, "preview/test_data", self.logger)
            except IndexError as ie:
                logger.warning(f"Not logging preview of test_data: {ie}")

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_nb):
        tensorboard_log, lightning_log = self.model.training_step(batch, batch_nb, self)
        return self._construct_lightning_log(
            tensorboard_log, lightning_log, suffix="train"
        )

    def validation_step(self, batch, batch_nb):
        x, target = batch
        pred = self.forward(x)

        return {"pred": pred, "target": target, "x": x}

    def _log_metrics_for_hparams(self, tensorboard_log: dict):
        acc = tensorboard_log["acc"]
        loss = tensorboard_log["loss"]
        if self.acc < 0:
            self.acc = acc
            self.loss = loss
        else:
            self.acc = self.decay * self.acc + (1 - self.decay) * acc
            self.loss = self.decay * self.loss + (1 - self.decay) * loss

        log_hparams(
            hparam_dict=self.hparams.to_dict(),
            metric_dict={"metrics/acc": self.acc, "metrics/loss": self.loss},
            _logger=self.logger,
            global_step=self.global_step,
        )

    def validation_end(self, outputs):
        tensorboard_log, lightning_log = self.model.aggregate_outputs(outputs, self)

        # self._log_metrics_for_hparams(tensorboard_log)

        return self._construct_lightning_log(
            tensorboard_log, lightning_log, suffix="val"
        )

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_end(self, outputs):
        with open(get_logger_dir(self.logger) / "outputs.pkl", "wb") as f:
            pickle.dump(outputs, f)

        tensorboard_log, lightning_log = self.model.aggregate_outputs(outputs, self)
        logger.info(f"Test accuracy is: {tensorboard_log['acc']}")
        return self._construct_lightning_log(
            tensorboard_log, lightning_log, suffix="test"
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        return optimizer

    @pl.data_loader
    def train_dataloader(self):
        if self.sampling_probs is None:
            sampler = self.sampler_cls
        else:
            sampler = partial(self.sampler_cls, predefined_weights=self.sampling_probs)

        return get_fixed_dataloader(
            self.train_data,
            self.hparams["batch_size"],
            sampler=sampler,
            num_workers=self.hparams["n_cpu"],
        )

    @pl.data_loader
    def val_dataloader(self):
        return get_fixed_dataloader(
            self.val_data,
            self.hparams["batch_size"],
            sampler=self.sampler_cls,
            num_workers=self.hparams["n_cpu"],
            worker_init_fn=lambda worker_id: np.random.seed(worker_id),
        )

    @pl.data_loader
    def test_dataloader(self):
        sampler = SequenceBatchSampler(
            self.sampler_cls(self.test_data, replacement=True),
            batch_size=self.hparams["batch_size"],
            drop_last=False,
            sequence_length=self.test_data.sequence_length,
            samples_idx=self.test_data.samples_idx,
        )
        # we want to make sure test data follows the same distribution like the benchmark
        loader = DataLoader(
            dataset=self.test_data,
            batch_sampler=sampler,
            num_workers=self.hparams["n_cpu"],
            collate_fn=get_sequence_collate_fn(
                sequence_length=self.test_data.sequence_length
            ),
        )
        return loader

    def benchmark(self, benchmark_dir, device, threshold=0.5):
        self.cuda(device)
        self.eval()
        data = get_data(benchmark_dir)
        predictions_dict = {}
        total = 0
        real = 0
        for i in tqdm(range(len(data))):
            img, _ = data[i]
            name = Path(data.samples[i][0]).name
            pred = self(img.unsqueeze(0).cuda(device)).detach().cpu().squeeze()
            pred = F.softmax(pred, dim=0).numpy()
            pred -= threshold
            if pred[1] > 0:
                predictions_dict[name] = "real"
                real += 1
            else:
                predictions_dict[name] = "fake"
            total += 1
        logger.info(f"real %: {real/total}")
        return predictions_dict

    def multiclass_roc_auc_score(self, target: torch.Tensor, pred: torch.Tensor):
        if self.hparams["log_roc_values"]:
            return multiclass_roc_auc_score(
                target.squeeze().detach().cpu(),
                pred.detach().cpu().argmax(dim=1),
                self.label_binarizer,
            )
        else:
            return NAN_TENSOR

    def log_confusion_matrix(self, target: torch.Tensor, pred: torch.Tensor):
        return log_confusion_matrix(
            self.logger,
            self.global_step,
            target,
            torch.argmax(pred, dim=1),
            self.file_list.class_to_idx,
        )

    def log_roc_graph(self, target: torch.Tensor, pred: torch.Tensor):
        if self.hparams["log_roc_values"]:
            log_roc_graph(
                self.logger,
                self.global_step,
                target.squeeze(),
                pred[:, self.positive_class],
                self.positive_class,
            )

    @staticmethod
    def _construct_lightning_log(
        tensorboard_log: dict,
        lightning_log: dict = None,
        suffix: str = "train",
        prefix: str = "metrics",
    ):
        lightning_log = lightning_log or {}
        fixed_log = {}

        for metric, value in tensorboard_log.items():
            if isinstance(value, dict):
                fixed_log[f"{prefix}/{metric}"] = value
            else:
                fixed_log[f"{prefix}/{metric}"] = {suffix: value}
                # dicts need to be removed from log, otherwise lightning tries to call
                # .item() on it -> only add non dict values
        return {"log": fixed_log, **lightning_log}

    @classmethod
    def load_from_metrics(cls, weights_path, tags_csv, overwrite_hparams=None):
        overwrite_hparams = overwrite_hparams or {}

        hparams = load_hparams_from_tags_csv(tags_csv)
        hparams.__dict__["logger"] = eval(hparams.__dict__.get("logger", "None"))

        hparams.__setattr__("on_gpu", False)
        hparams.__dict__.update(overwrite_hparams)

        # load on CPU only to avoid OOM issues
        # then its up to user to put back on GPUs
        checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)

        # load the state_dict on the model automatically
        model = cls(hparams)
        model.load_state_dict(checkpoint["state_dict"])

        # give model a chance to load something
        model.on_load_checkpoint(checkpoint)

        return model
