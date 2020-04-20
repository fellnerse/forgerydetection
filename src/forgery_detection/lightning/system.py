import logging
import pickle
from argparse import Namespace
from functools import partial
from typing import Dict
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.core.saving import load_hparams_from_tags_csv
from torch import optim
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.sampler import SequentialSampler
from torchvision import transforms

from forgery_detection.data.face_forensics.splits import TEST_NAME
from forgery_detection.data.face_forensics.splits import TRAIN_NAME
from forgery_detection.data.face_forensics.splits import VAL_NAME
from forgery_detection.data.file_lists import FileList
from forgery_detection.data.file_lists import SimpleFileList
from forgery_detection.data.loading import BalancedSampler
from forgery_detection.data.loading import calculate_class_weights
from forgery_detection.data.loading import get_fixed_dataloader
from forgery_detection.data.utils import colour_jitter
from forgery_detection.data.utils import crop
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
from forgery_detection.data.utils import rfft_transform
from forgery_detection.lightning.logging.utils import AudioMode
from forgery_detection.lightning.logging.utils import DictHolder
from forgery_detection.lightning.logging.utils import get_logger_dir
from forgery_detection.lightning.logging.utils import log_confusion_matrix
from forgery_detection.lightning.logging.utils import log_dataset_preview
from forgery_detection.lightning.logging.utils import log_hparams
from forgery_detection.lightning.logging.utils import log_roc_graph
from forgery_detection.lightning.logging.utils import SystemMode
from forgery_detection.models.audio.multi_class_classification import AudioNet
from forgery_detection.models.audio.multi_class_classification import AudioNetFrozen
from forgery_detection.models.audio.multi_class_classification import (
    AudioNetLayer2Unfrozen,
)
from forgery_detection.models.audio.multi_class_classification import AudioOnly
from forgery_detection.models.audio.multi_class_classification import FrameNet
from forgery_detection.models.audio.multi_class_classification import PretrainedAudioNet
from forgery_detection.models.audio.multi_class_classification import (
    PretrainSyncAudioNet,
)
from forgery_detection.models.audio.multi_class_classification import SyncAudioNet
from forgery_detection.models.audio.similarity_stuff import PretrainedSimilarityNet
from forgery_detection.models.audio.similarity_stuff import PretrainedSyncNet
from forgery_detection.models.audio.similarity_stuff import SimilarityNet
from forgery_detection.models.audio.similarity_stuff import SimilarityNetClassification
from forgery_detection.models.audio.similarity_stuff import SyncNet
from forgery_detection.models.image.ae import AEFullFaceNet
from forgery_detection.models.image.ae import AEFullVGG
from forgery_detection.models.image.ae import AEL1VGG
from forgery_detection.models.image.ae import BiggerFourierAE
from forgery_detection.models.image.ae import BiggerL1AE
from forgery_detection.models.image.ae import BiggerWeightedFourierAE
from forgery_detection.models.image.ae import BiggerWindowedFourierAE
from forgery_detection.models.image.ae import FourierAE
from forgery_detection.models.image.ae import KrakenAE
from forgery_detection.models.image.ae import LaplacianLossNet
from forgery_detection.models.image.ae import PretrainedBiggerFourierAE
from forgery_detection.models.image.ae import PretrainedLaplacianLossNet
from forgery_detection.models.image.ae import SimpleAE
from forgery_detection.models.image.ae import SimpleAEL1
from forgery_detection.models.image.ae import SimpleAEL1Pretrained
from forgery_detection.models.image.ae import SimpleAEVGG
from forgery_detection.models.image.ae import SqrtNet
from forgery_detection.models.image.ae import StackedAE
from forgery_detection.models.image.ae import StyleNet
from forgery_detection.models.image.ae import SupervisedAEL1
from forgery_detection.models.image.ae import SupervisedAEVgg
from forgery_detection.models.image.ae import SupervisedBiggerAEL1
from forgery_detection.models.image.ae import SupervisedBiggerFourierAE
from forgery_detection.models.image.ae import SupervisedResnetAE
from forgery_detection.models.image.ae import SupervisedTwoHeadedAEVGG
from forgery_detection.models.image.ae import WeightedBiggerFourierAE
from forgery_detection.models.image.aegan import AEGAN
from forgery_detection.models.image.frequency_ae import BiggerFrequencyAE
from forgery_detection.models.image.frequency_ae import BiggerFrequencyAElog
from forgery_detection.models.image.frequency_ae import FrequencyAE
from forgery_detection.models.image.frequency_ae import FrequencyAEcomplex
from forgery_detection.models.image.frequency_ae import FrequencyAEMagnitude
from forgery_detection.models.image.frequency_ae import FrequencyAEtanh
from forgery_detection.models.image.frequency_ae import PretrainedFrequencyNet
from forgery_detection.models.image.frequency_ae import SupervisedBiggerFrequencyAE
from forgery_detection.models.image.imagenet import ImageNetResnet
from forgery_detection.models.image.imagenet import ImageNetResnet152
from forgery_detection.models.image.imagenet import PretrainedFFFCResnet152
from forgery_detection.models.image.imagenet import PretrainedImageNetResnet
from forgery_detection.models.image.imagenet import PretrainFFFCResnet152
from forgery_detection.models.image.multi_class_classification import ResidualResnet
from forgery_detection.models.image.multi_class_classification import Resnet18
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
from forgery_detection.models.image.multi_class_classification import Resnet18SameAsInAE
from forgery_detection.models.image.multi_class_classification import (
    Resnet18UntrainedMultiClassDropout,
)
from forgery_detection.models.image.vae import SimpleVAE
from forgery_detection.models.image.vae import SupervisedVae
from forgery_detection.models.utils import LightningModel
from forgery_detection.models.video.ae import SmallerVideoAE
from forgery_detection.models.video.ae import SupervisedSmallerVideoAE
from forgery_detection.models.video.ae import SupervisedSmallerVideoAEGlobalAvgPooling
from forgery_detection.models.video.ae import SupervisedVideoAE
from forgery_detection.models.video.ae import VideoAE2
from forgery_detection.models.video.multi_class_classification import MC3
from forgery_detection.models.video.multi_class_classification import R2Plus1
from forgery_detection.models.video.multi_class_classification import R2Plus1Frozen
from forgery_detection.models.video.multi_class_classification import R2Plus1Small
from forgery_detection.models.video.multi_class_classification import (
    R2Plus1SmallAudioLikePretrain,
)
from forgery_detection.models.video.multi_class_classification import (
    R2Plus1SmallAudiolikePretrained,
)
from forgery_detection.models.video.multi_class_classification import R2Plus1Smallest
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
from forgery_detection.models.video.scramble import ScrambleNet
from forgery_detection.models.video.vae import VideoAE
from forgery_detection.models.video.vae import VideoVae
from forgery_detection.models.video.vae import VideoVaeDetachedSupervised
from forgery_detection.models.video.vae import VideoVaeSupervised
from forgery_detection.models.video.vae import VideoVaeSupervisedBCE
from forgery_detection.models.video.vae import VideoVaeUpsample

logger = logging.getLogger(__file__)


class Supervised(pl.LightningModule):
    MODEL_DICT = {
        "resnet18": Resnet18,
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
        "resnet18_imagenet": ImageNetResnet,
        "pretrained_resnet18_imagenet": PretrainedImageNetResnet,
        "pretrain_ff_fc_resnet152": PretrainFFFCResnet152,
        "resnet152_imagenet": ImageNetResnet152,
        "resnet152_imagenet_pretrained": PretrainedFFFCResnet152,
        "r2plus1": R2Plus1,
        "r2plus1frozen": R2Plus1Frozen,
        "r2plus1small": R2Plus1Small,
        "r2plus1small_audiolike_pretrain": R2Plus1SmallAudioLikePretrain,
        "r2plus1small_audiolike": R2Plus1SmallAudiolikePretrained,
        "r2plus1smallest": R2Plus1Smallest,
        "mc3": MC3,
        "audionet": AudioNet,
        "audionet_frozen": AudioNetFrozen,
        "audionet_pretrained": PretrainedAudioNet,
        "audionet_layer2unfrozen": AudioNetLayer2Unfrozen,
        "audioonly": AudioOnly,
        "vae": SimpleVAE,
        "ae": SimpleAE,
        "ae_vgg": SimpleAEVGG,
        "ae_full_vgg": AEFullVGG,
        "ae_full_facenet": AEFullFaceNet,
        "ae_l1": SimpleAEL1,
        "ae_l1_pretrained": SimpleAEL1Pretrained,
        "ae_l1_vgg": AEL1VGG,
        "ae_laplacian": LaplacianLossNet,
        "ae_laplacian_pretrained": PretrainedLaplacianLossNet,
        "ae_supervised": SupervisedAEL1,
        "ae_vgg_supervised": SupervisedAEVgg,
        "ae_vgg_supervised_two_headed": SupervisedTwoHeadedAEVGG,
        "vae_supervised": SupervisedVae,
        "vae_video": VideoVae,
        "vae_video_upsample": VideoVaeUpsample,
        "vae_video_supervised": VideoVaeSupervised,
        "vae_video_detached_supervised": VideoVaeDetachedSupervised,
        "vae_video_supervised_bce": VideoVaeSupervisedBCE,
        "ae_video": VideoAE,
        "ae_video2": VideoAE2,
        "ae_video_supervised": SupervisedVideoAE,
        "ae_video2_smaller": SmallerVideoAE,
        "ae_video2_smaller_supervised": SupervisedSmallerVideoAE,
        "ae_video2_smaller_supervised_avg_pooling": SupervisedSmallerVideoAEGlobalAvgPooling,
        "ae_stacked": StackedAE,
        "style_net": StyleNet,
        "sqrt_net": SqrtNet,
        "scramble_net": ScrambleNet,
        "ae_gan": AEGAN,
        "kraken_ae": KrakenAE,
        "frequency_ae": FrequencyAE,
        "pretrained_frequency_ae": PretrainedFrequencyNet,
        "frequency_ae_tanh": FrequencyAEtanh,
        "frequency_ae_magnitude": FrequencyAEMagnitude,
        "frequency_ae_complex": FrequencyAEcomplex,
        "bigger_frequency_ae": BiggerFrequencyAE,
        "supervised_bigger_frequency_ae": SupervisedBiggerFrequencyAE,
        "bigger_frequency_ae_log": BiggerFrequencyAElog,
        "fourier_ae": FourierAE,
        "bigger_fourier_ae": BiggerFourierAE,
        "pretrained_bigger_fourier_ae": PretrainedBiggerFourierAE,
        "weighted_bigger_fourier_ae": WeightedBiggerFourierAE,
        "bigger_weighted_fourier_ae": BiggerWeightedFourierAE,
        "supervised_bigger_fourier_ae": SupervisedBiggerFourierAE,
        "supervised_bigger_l1_ae": SupervisedBiggerAEL1,
        "bigger_l1_ae": BiggerL1AE,
        "bigger_windowed_fourier_ae": BiggerWindowedFourierAE,
        "supervised_resnet_ae": SupervisedResnetAE,
        "resnet18_same_as_in_ae": Resnet18SameAsInAE,
        "frame_net": FrameNet,
        "similarity_net": SimilarityNet,
        "pretrained_similarity_net": PretrainedSimilarityNet,
        "similarity_net_classification": SimilarityNetClassification,
        "syncnet": SyncNet,
        "pretrained_syncnet": PretrainedSyncNet,
        "pretrain_sync_audio_net": PretrainSyncAudioNet,
        "sync_audio_net": SyncAudioNet,
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
        "rfft": rfft_transform(),
        "imagenet_val": [transforms.Resize(256), transforms.CenterCrop(224)],
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

        self.resize_transform = self._get_transforms(self.hparams["resize_transforms"])
        image_augmentation_transforms = self._get_transforms(
            self.hparams["image_augmentation_transforms"]
        )
        self.tensor_augmentation_transforms = self._get_transforms(
            self.hparams["tensor_augmentation_transforms"]
        )

        if self.hparams["audio_file"]:
            self.audio_file_list = SimpleFileList.load(self.hparams["audio_file"])
        else:
            self.audio_file_list = None

        self.audio_mode = AudioMode[self.hparams["audio_mode"]]

        self.train_data = self.file_list.get_dataset(
            TRAIN_NAME,
            image_transforms=self.resize_transform + image_augmentation_transforms,
            tensor_transforms=self.tensor_augmentation_transforms,
            sequence_length=self.model.sequence_length,
            audio_file_list=self.audio_file_list,
            audio_mode=self.audio_mode
            # should_align_faces=True,
        )
        self.val_data = self.file_list.get_dataset(
            VAL_NAME,
            image_transforms=self.resize_transform,
            tensor_transforms=self.tensor_augmentation_transforms,
            sequence_length=self.model.sequence_length,
            audio_file_list=self.audio_file_list,
            audio_mode=self.audio_mode
            # should_align_faces=True,
        )
        # handle empty test_data better
        self.test_data = self.file_list.get_dataset(
            TEST_NAME,
            image_transforms=self.resize_transform,
            tensor_transforms=self.tensor_augmentation_transforms,
            sequence_length=self.model.sequence_length,
            audio_file_list=self.audio_file_list,
            audio_mode=self.audio_mode
            # should_align_faces=True,
        )
        self.hparams.add_dataset_size(len(self.train_data), TRAIN_NAME)
        self.hparams.add_dataset_size(len(self.val_data), VAL_NAME)
        self.hparams.add_dataset_size(len(self.test_data), TEST_NAME)

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

    def validation_step(self, batch, batch_nb, dataloader_id=-1):
        # x, target = batch
        # batch = x, (target - 1) % 5
        x, target = batch
        pred = self.forward(x)

        return {
            "pred": pred,
            "target": target,
        }  # todo this is needed for autoencoders "x": x}

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

    def validation_epoch_end(self, outputs):
        tensorboard_log, lightning_log = self.model.aggregate_outputs(outputs, self)

        # self._log_metrics_for_hparams(tensorboard_log)

        return self._construct_lightning_log(
            tensorboard_log, lightning_log, suffix="val"
        )

    def test_step(self, batch, batch_nb):
        with torch.no_grad():
            val_out = self.validation_step(batch, batch_nb)
            # for key, value in val_out.items():
            #     val_out[key] = value.cpu()
            return val_out

    def test_epoch_end(self, outputs):
        with torch.no_grad():
            with open(get_logger_dir(self.logger) / "outputs.pkl", "wb") as f:
                pickle.dump(outputs, f)

            tensorboard_log, lightning_log = self.model.aggregate_outputs(outputs, self)
            logger.info(f"Test accuracy is: {tensorboard_log}")
            return self._construct_lightning_log(
                tensorboard_log, lightning_log, suffix="test"
            )

    def configure_optimizers(self):
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
            momentum=0.9,
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
        # static_batch_data = self.file_list.get_dataset(
        #     VAL_NAME,
        #     image_transforms=self.resize_transform,
        #     tensor_transforms=self.tensor_augmentation_transforms,
        #     sequence_length=self.model.sequence_length,
        #     audio_file=self.hparams["audio_file"],
        # )
        # static_batch_idx = static_batch_data.samples_idx[:: len(static_batch_data) // 3]
        # # this is a really shitty hack but needed for compatibility
        # # if len(static_batch_data) is divisable by 3 the resulting length is only 3
        # if len(static_batch_idx) == 3:
        #     static_batch_idx += [
        #         static_batch_data.samples_idx[len(static_batch_data) // 2]
        #     ]
        # static_batch_data.samples_idx = static_batch_idx
        # static_batch_loader = get_fixed_dataloader(
        #     static_batch_data,
        #     4,
        #     sampler=SequentialSampler,  # use sequence sampler
        #     num_workers=self.hparams["n_cpu"],
        #     worker_init_fn=lambda worker_id: np.random.seed(worker_id),
        # )
        if self.sampling_probs is None:
            sampler = self.sampler_cls
        else:
            sampler = partial(self.sampler_cls, predefined_weights=self.sampling_probs)
        return [
            get_fixed_dataloader(
                self.val_data,
                self.hparams["batch_size"],
                sampler=sampler,
                num_workers=self.hparams["n_cpu"],
                worker_init_fn=lambda worker_id: np.random.seed(worker_id),
            ),
            # use static batch for autoencoders
            # get_fixed_dataloader(
            #     self.test_data,
            #     self.hparams["batch_size"],
            #     sampler=self.sampler_cls,
            #     num_workers=self.hparams["n_cpu"],
            #     worker_init_fn=lambda worker_id: np.random.seed(worker_id),
            # ),
        ]

    @pl.data_loader
    def test_dataloader(self):
        # self.file_list = FileList.load(
        #     "/data/ssd1/file_lists/c40/tracked_resampled_faces_224.json"
        # )
        static_batch_data = self.file_list.get_dataset(
            VAL_NAME,  # TEST_NAME,
            image_transforms=self.resize_transform,
            tensor_transforms=self.tensor_augmentation_transforms,
            sequence_length=self.model.sequence_length,
            audio_file_list=self.hparams["audio_file"],
        )
        # static_batch_idx = static_batch_data.samples_idx[:: len(static_batch_data) // 3]
        static_batch_idx = static_batch_data.samples_idx[::1]
        # this is a really shitty hack but needed for compatibility
        # if len(static_batch_data) is divisable by 3 the resulting length is only 3
        if len(static_batch_idx) == 3:
            static_batch_idx += [
                static_batch_data.samples_idx[len(static_batch_data) // 2]
            ]
        static_batch_data.samples_idx = static_batch_idx
        static_batch_loader = get_fixed_dataloader(
            static_batch_data,
            16,  # 4,
            sampler=SequentialSampler,  # use sequence sampler
            num_workers=self.hparams["n_cpu"],
            worker_init_fn=lambda worker_id: np.random.seed(worker_id),
        )
        return static_batch_loader
        loader = get_fixed_dataloader(
            self.test_data,
            self.hparams["batch_size"],
            sampler=self.sampler_cls,
            num_workers=self.hparams["n_cpu"],
            worker_init_fn=lambda worker_id: np.random.seed(worker_id),
        )
        return loader

    def log_confusion_matrix(self, target: torch.Tensor, pred: torch.Tensor):
        return log_confusion_matrix(
            self.logger,
            self.global_step,
            target,
            torch.argmax(pred[:, : self.model.num_classes], dim=1),
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

    def dictify_list_with_class_names(
        self, class_acc: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        class_accuracies_dict = {}
        for key, value in self.file_list.class_to_idx.items():
            try:
                class_accuracies_dict[str(key)] = class_acc[value]
            except IndexError:
                # can happen when there were no samples in batch
                # will just not logg anything for that step then
                pass
        return class_accuracies_dict

    def _construct_lightning_log(
        self,
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

        if (
            str(hparams.sampling_probs) == "nan"
            or str(hparams.sampling_probs) == "None"
        ):
            hparams.__dict__["sampling_probs"] = None

        if str(hparams.audio_file) == "nan":
            hparams.__dict__["audio_file"] = None

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
