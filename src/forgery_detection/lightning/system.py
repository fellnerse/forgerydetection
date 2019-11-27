import logging
import pickle
from argparse import Namespace
from pathlib import Path
from typing import Union

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
from forgery_detection.data.loading import calculate_class_weights
from forgery_detection.data.loading import FiftyFiftySampler
from forgery_detection.data.loading import get_fixed_dataloader
from forgery_detection.data.set import FileList
from forgery_detection.data.utils import crop
from forgery_detection.data.utils import get_data
from forgery_detection.data.utils import resized_crop
from forgery_detection.data.utils import resized_crop_flip
from forgery_detection.lightning.logging.utils import DictHolder
from forgery_detection.lightning.logging.utils import get_logger_dir
from forgery_detection.lightning.logging.utils import log_confusion_matrix
from forgery_detection.lightning.logging.utils import log_dataset_preview
from forgery_detection.lightning.logging.utils import log_roc_graph
from forgery_detection.lightning.logging.utils import multiclass_roc_auc_score
from forgery_detection.lightning.logging.utils import SystemMode
from forgery_detection.lightning.utils import NAN_TENSOR
from forgery_detection.models.image.multi_class_classification import (
    Resnet18MultiClassDropout,
)
from forgery_detection.models.image.multi_class_classification import Resnet18MultiHead
from forgery_detection.models.utils import SequenceClassificationModel
from forgery_detection.models.video.multi_class_classification import Resnet183D
from forgery_detection.models.video.multi_class_classification import (
    Resnet183DNoDropout,
)

logger = logging.getLogger(__file__)


class Supervised(pl.LightningModule):
    MODEL_DICT = {
        "resnet18multiclassdropout": Resnet18MultiClassDropout,
        "resnet183d": Resnet183D,
        "resnet183dnodropout": Resnet183DNoDropout,
        "resnet18heads": Resnet18MultiHead,
    }

    CUSTOM_TRANSFORMS = {
        "crop": crop,
        "resized_crop": resized_crop,
        "resized_crop_flip": resized_crop_flip,
    }

    def __init__(self, kwargs: Union[dict, Namespace]):
        super(Supervised, self).__init__()

        self.hparams = DictHolder(kwargs)
        self.model: SequenceClassificationModel = self.MODEL_DICT[
            self.hparams["model"]
        ]()

        # load data-sets
        self.file_list = FileList.load(self.hparams["data_dir"])
        if len(self.file_list.classes) != self.model.num_classes:
            raise ValueError(
                f"Classes of model ({self.model.num_classes}) != classes of dataset"
                f" ({len(self.file_list.cl)})"
            )

        transform = self.CUSTOM_TRANSFORMS[self.hparams["transforms"]]()
        self.train_data = self.file_list.get_dataset(
            TRAIN_NAME, transform, sequence_length=self.model.sequence_length
        )
        self.val_data = self.file_list.get_dataset(
            VAL_NAME, transform, sequence_length=self.model.sequence_length
        )
        self.test_data = self.file_list.get_dataset(
            TEST_NAME, transform, sequence_length=self.model.sequence_length
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
            self.sampler_cls = FiftyFiftySampler
        else:
            self.sampler_cls = RandomSampler

        system_mode = self.hparams.pop("mode")

        if system_mode is SystemMode.TRAIN:
            self.hparams.add_nb_trainable_params(self.model)
            if self.hparams["class_weights"]:
                labels, weights = calculate_class_weights(self.val_data)
                self.hparams.add_class_weights(labels, weights)
                self.class_weights = torch.tensor(weights, dtype=torch.float)
            else:
                self.class_weights = None

        elif system_mode is SystemMode.TEST:
            self.class_weights = None

        elif system_mode is SystemMode.BENCHMARK:
            pass

    def on_sanity_check_start(self):
        log_dataset_preview(self.train_data, "preview/train_data", self.logger)
        log_dataset_preview(self.val_data, "preview/val_data", self.logger)
        log_dataset_preview(self.test_data, "preview/test_data", self.logger)

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

        return {"pred": pred, "target": target}

    def validation_end(self, outputs):
        tensorboard_log, lightning_log = self.model.aggregate_outputs(outputs, self)
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
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams["lr"]
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, verbose=True, patience=self.hparams["scheduler_patience"]
        )
        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        return get_fixed_dataloader(
            self.train_data,
            self.hparams["batch_size"],
            sampler=self.sampler_cls,
            num_workers=12,
        )

    @pl.data_loader
    def val_dataloader(self):
        return get_fixed_dataloader(
            self.val_data,
            self.hparams["batch_size"],
            sampler=self.sampler_cls,
            num_workers=12,
        )

    @pl.data_loader
    def test_dataloader(self):
        # we want to make sure test data follows the same distribution like the benchmark
        loader = DataLoader(
            dataset=self.test_data,
            batch_size=self.hparams["batch_size"],
            shuffle=False,
            num_workers=12,
            sampler=self.sampler_cls(self.test_data, replacement=True),
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
