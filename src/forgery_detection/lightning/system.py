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
from forgery_detection.data.utils import crop
from forgery_detection.data.utils import FileList
from forgery_detection.data.utils import get_data
from forgery_detection.data.utils import resized_crop
from forgery_detection.data.utils import resized_crop_flip
from forgery_detection.lightning.utils import calculate_class_weights
from forgery_detection.lightning.utils import DictHolder
from forgery_detection.lightning.utils import FiftyFiftySampler
from forgery_detection.lightning.utils import get_fixed_dataloader
from forgery_detection.lightning.utils import get_logger_dir
from forgery_detection.lightning.utils import log_confusion_matrix
from forgery_detection.lightning.utils import log_roc_graph
from forgery_detection.lightning.utils import multiclass_roc_auc_score
from forgery_detection.lightning.utils import SystemMode
from forgery_detection.models.binary_classification import Resnet18Binary
from forgery_detection.models.binary_classification import Resnet18BinaryDropout
from forgery_detection.models.binary_classification import Resnet18BinaryDropoutFrozen
from forgery_detection.models.binary_classification import Resnet18BinaryFrozen
from forgery_detection.models.binary_classification import SqueezeBinary
from forgery_detection.models.binary_classification import VGG11Binary
from forgery_detection.models.multi_class_classification import Resnet18MultiClass
from forgery_detection.models.multi_class_classification import Resnet18MultiClassFrozen


class Supervised(pl.LightningModule):
    MODEL_DICT = {
        "squeeze": SqueezeBinary,
        "vgg11": VGG11Binary,
        "resnet18": Resnet18Binary,
        "resnet18dropout": Resnet18BinaryDropout,
        "resnet18dropoutfrozen": Resnet18BinaryDropoutFrozen,
        "resnet18frozen": Resnet18BinaryFrozen,
        "resnet18multiclass": Resnet18MultiClass,
        "resnet18multiclassfrozen": Resnet18MultiClassFrozen,
    }

    CUSTOM_TRANSFORMS = {
        "crop": crop,
        "resized_crop": resized_crop,
        "resized_crop_flip": resized_crop_flip,
    }

    def __init__(self, kwargs: Union[dict, Namespace]):
        super(Supervised, self).__init__()

        self.hparams = DictHolder(kwargs)
        self.model = self.MODEL_DICT[self.hparams["model"]]()

        # load data-sets
        self.file_list = FileList.load(self.hparams["data_dir"])
        transform = self.CUSTOM_TRANSFORMS[self.hparams["transforms"]]()
        self.train_data = self.file_list.get_dataset(TRAIN_NAME, transform)
        self.val_data = self.file_list.get_dataset(VAL_NAME, transform)
        self.test_data = self.file_list.get_dataset(TEST_NAME, transform)

        # label_binarizer and positive class is needed for roc_auc-multiclass
        self.label_binarizer = LabelBinarizer()
        self.label_binarizer.fit(list(self.file_list.class_to_idx.values()))
        self.positive_class = list(self.file_list.class_to_idx.values())[-1]
        self.hparams["positive_class"] = self.positive_class

        if self.hparams["balance_data"]:
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

    def forward(self, x):
        return self.model.forward(x)

    def loss(self, logits, labels):
        try:
            # classweights  -> multiply each output with weight of target
            #               -> sum result up and divide by sum of these weights
            cross_engropy = F.cross_entropy(logits, labels, weight=self.class_weights)
        except RuntimeError:
            print(logits, labels, self.class_weights)
            device_index = logits.device.index
            print(f"switching device for class_weights to {device_index}")
            self.class_weights = self.class_weights.cuda(device_index)
            cross_engropy = F.cross_entropy(logits, labels, weight=self.class_weights)
        return cross_engropy

    def training_step(self, batch, batch_nb):
        x, target = batch
        pred = self.forward(x)

        loss_val = self.loss(pred, target)

        pred = F.softmax(pred, dim=1)
        train_acc = self._calculate_accuracy(pred, target)

        roc_auc = multiclass_roc_auc_score(
            target.squeeze().detach().cpu(),
            pred.detach().cpu().argmax(dim=1),
            self.label_binarizer,
        )

        log = {"loss": loss_val, "acc": train_acc, "roc_auc": roc_auc}

        return self._construct_lightning_log(log, suffix="train")

    def validation_step(self, batch, batch_nb):
        x, target = batch
        pred = self.forward(x)

        return {"pred": pred.squeeze(), "target": target.squeeze()}

    def validation_end(self, outputs):
        log = self._aggregate_outputs(outputs)
        return self._construct_lightning_log(log, suffix="val")

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_end(self, outputs):
        with open(get_logger_dir(self.logger) / "outputs.pkl", "wb") as f:
            pickle.dump(outputs, f)
        log = self._aggregate_outputs(outputs)
        return self._construct_lightning_log(log, suffix="test")

    def _aggregate_outputs(self, outputs):
        # aggregate values from validation step
        pred = torch.cat([x["pred"] for x in outputs], 0)
        target = torch.cat([x["target"] for x in outputs], 0)

        test_loss_mean = self.loss(pred, target)
        pred = pred.cpu()
        target = target.cpu()
        pred = F.softmax(pred, dim=1)
        test_acc_mean = self._calculate_accuracy(pred, target)

        # confusion matrix
        class_accuracies = log_confusion_matrix(
            self.logger,
            self.global_step,
            target,
            torch.argmax(pred, dim=1),
            self.file_list.class_to_idx,
        )

        # roc_auc_score
        log_roc_graph(
            self.logger,
            self.global_step,
            target.squeeze(),
            pred[:, self.positive_class],
            self.positive_class,
        )

        roc_auc = multiclass_roc_auc_score(
            target.squeeze().detach().cpu(),
            pred.detach().cpu().argmax(dim=1),
            self.label_binarizer,
        )

        log = {
            "loss": test_loss_mean,
            "acc": test_acc_mean,
            "roc_auc": roc_auc,
            "class_acc": class_accuracies,
        }
        return log

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
        self.hparams.add_dataset_size(len(self.train_data), TRAIN_NAME)
        return get_fixed_dataloader(
            self.train_data, self.hparams["batch_size"], sampler=self.sampler_cls
        )

    @pl.data_loader
    def val_dataloader(self):
        self.hparams.add_dataset_size(len(self.val_data), VAL_NAME)
        return get_fixed_dataloader(
            self.val_data, self.hparams["batch_size"], sampler=self.sampler_cls
        )

    @pl.data_loader
    def test_dataloader(self):
        self.hparams.add_dataset_size(len(self.test_data), TEST_NAME)
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
        print(f"real %: {real/total}")
        return predictions_dict

    @staticmethod
    def _calculate_accuracy(y_hat, y):
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = labels_hat.eq(y).float().mean()
        return val_acc

    @staticmethod
    def _construct_lightning_log(
        log: dict, suffix: str = "train", prefix: str = "metrics"
    ):
        fixed_log = {}
        safe_log = {}
        for metric, value in log.items():
            if isinstance(value, dict):
                fixed_log[f"{prefix}/{metric}"] = value
            else:
                fixed_log[f"{prefix}/{metric}"] = {suffix: value}
                # dicts need to be removed from log, otherwise lightning tries to call
                # .item() on it -> only add non dict values
                safe_log[metric] = value
        return {"log": fixed_log, **safe_log}

    @classmethod
    def load_from_metrics(cls, weights_path, tags_csv, overwrite_hparams=None):
        overwrite_hparams = overwrite_hparams or {}

        hparams = load_hparams_from_tags_csv(tags_csv)
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
