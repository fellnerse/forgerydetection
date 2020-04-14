import abc
import logging
from abc import ABC
from functools import reduce

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.utils import make_grid

from forgery_detection.lightning.utils import VAL_ACC

logger = logging.getLogger(__file__)

RECON_X = "recon_x"
PRED = "pred"
TARGET = "target"
X = "x"
LOSS = "loss"
RECONSTRUCTION_LOSS = "reconstruction_loss"
CLASSIFICATION_LOSS = "classification_loss"
KLD_LOSS = "kld_loss"
ACC = "acc"
CLASS_ACC = "class_acc"
MU = "mu"
LOG_VAR = "log_var"
BATCH_SIZE = "batch_size"


class LightningModel(nn.Module, ABC):
    def __init__(self, num_classes, sequence_length, contains_dropout):
        super().__init__()
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.contains_dropout = contains_dropout

    @abc.abstractmethod
    def loss(self, logits, labels):
        raise NotImplementedError()

    @abc.abstractmethod
    def training_step(self, batch, batch_nb, system):
        raise NotImplementedError()

    @abc.abstractmethod
    def aggregate_outputs(self, outputs, system):
        raise NotImplementedError()

    def calculate_accuracy(self, pred, target):
        labels_hat = torch.argmax(pred, dim=1)
        acc = labels_hat.eq(target).float().mean()
        return acc

    @staticmethod
    def _set_requires_grad_for_module(module, requires_grad=False):
        for param in module.parameters():
            param.requires_grad = requires_grad


class SequenceClassificationModel(LightningModel, ABC):
    def __init__(self, num_classes, sequence_length, contains_dropout):
        super().__init__(
            num_classes, sequence_length, contains_dropout=contains_dropout
        )

    @abc.abstractmethod
    def forward(self, x):
        raise NotImplementedError()

    def loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def training_step(self, batch, batch_nb, system):
        x, target = batch

        # if the model uses dropout we want to calculate the metrics on predictions done
        # in eval mode before training no the samples
        if self.contains_dropout:
            with torch.no_grad():
                self.eval()
                pred_ = self.forward(x)
                self.train()

        pred = self.forward(x)
        loss = self.loss(pred, target)
        lightning_log = {"loss": loss}

        with torch.no_grad():
            train_acc = self.calculate_accuracy(pred, target)
            tensorboard_log = {"loss": {"train": loss}, "acc": {"train": train_acc}}

            if self.contains_dropout:
                pred = pred_
                loss_eval = self.loss(pred, target)
                acc_eval = self.calculate_accuracy(pred, target)
                tensorboard_log["loss"]["train_eval"] = loss_eval
                tensorboard_log["acc"]["train_eval"] = acc_eval

            tensorboard_log["roc_auc"] = system.multiclass_roc_auc_score(target, pred)
        return tensorboard_log, lightning_log

    def aggregate_outputs(self, outputs, system):
        # if there are more then one dataloader we ignore the additional data
        if len(system.val_dataloader()) > 1:
            outputs = outputs[0]

        with torch.no_grad():
            pred = torch.cat([x["pred"] for x in outputs], 0)
            target = torch.cat([x["target"] for x in outputs], 0)

            loss_mean = self.loss(pred, target)
            pred = pred.cpu()
            target = target.cpu()
            pred = F.softmax(pred, dim=1)
            acc_mean = self.calculate_accuracy(pred, target)

            # confusion matrix
            class_accuracies = system.log_confusion_matrix(target, pred)

            # roc_auc_score
            system.log_roc_graph(target, pred)

            roc_auc = system.multiclass_roc_auc_score(target, pred)

            tensorboard_log = {
                "loss": loss_mean,
                "acc": acc_mean,
                "roc_auc": roc_auc,
                "class_acc": class_accuracies,
            }
            lightning_log = {VAL_ACC: acc_mean}

        return tensorboard_log, lightning_log


class GeneralAE(LightningModel, ABC):
    def __init__(
        self, num_classes, sequence_length, contains_dropout, log_images_every=1
    ):
        super().__init__(num_classes, sequence_length, contains_dropout)
        self.log_image_count = log_images_every
        self.log_images_every = log_images_every

    def training_step(self, batch, batch_nb, system):
        x, target = batch
        forward_dict = self.forward(x)

        forward_dict[TARGET] = target
        forward_dict[X] = x

        metric_dict = self._calculate_metrics(**forward_dict)

        if hasattr(system, "_log_training") and system._log_training:
            system._log_training = False
            self._log_reconstructed_images(
                system, x, forward_dict[RECON_X], suffix="train"
            )

        lightning_log = {"loss": metric_dict[LOSS]}
        tensorboard_log = self._metric_dict_to_tensorboard_log(
            metric_dict, suffix="train"
        )
        return tensorboard_log, lightning_log

    def aggregate_outputs(self, output, system):
        """Aggregates outputs from val_step"""

        should_log_images = self._should_log_images()

        # make sure this can see how many data loaders we have
        random_batch, static_batch = output

        # process the random batch
        output_dict = self._transform_output_dict(random_batch)
        with torch.no_grad():
            # calculated metrics based on outputs
            metric_dict = self._calculate_metrics(**output_dict)

            if should_log_images:
                # log reconstructed images
                self._log_reconstructed_images(
                    system,
                    output_dict[X],
                    output_dict[RECON_X],
                    suffix="val/random_batch",
                )
                # the next training step should log reconstructed images as well
                system._log_training = True

            # confusion matrix
            class_accuracies = system.log_confusion_matrix(
                output_dict[TARGET].cpu(), output_dict[PRED].cpu()
            )

            metric_dict[CLASS_ACC] = class_accuracies

            tensorboard_log = metric_dict
            lightning_log = {VAL_ACC: metric_dict[ACC]}

        # process the static batch
        # for the static batch we log only the images
        if should_log_images:
            output_dict = self._transform_output_dict(static_batch)
            self._log_reconstructed_images(
                system, output_dict[X], output_dict[RECON_X], suffix="val/static_batch"
            )

        return tensorboard_log, lightning_log

    @abc.abstractmethod
    def encode(self, x):
        raise NotImplementedError()

    @abc.abstractmethod
    def decode(self, z):
        raise NotImplementedError()

    @abc.abstractmethod
    def forward(self, x) -> dict:
        raise NotImplementedError()

    @staticmethod
    def reconstruction_loss(recon_x, x) -> dict:
        raise NotImplementedError()

    def loss(self, logits, labels):
        """classification loss"""
        raise NotImplementedError()

    def _should_log_images(self):
        if self.log_image_count - self.log_images_every == 0:
            self.log_image_count = 1
            return True
        else:
            self.log_image_count += 1
            return False

    @staticmethod
    def _transform_output_dict(output: dict):
        network_output = [x[PRED] for x in output]
        x_recon = torch.cat([x[RECON_X] for x in network_output], 0)
        pred = torch.cat([x[PRED] for x in network_output], 0)

        target = torch.cat([x[TARGET] for x in output], 0)
        bs = output[0][TARGET].shape[0]
        x = torch.cat([x[X] for x in output], 0)
        return {RECON_X: x_recon, PRED: pred, TARGET: target, X: x, BATCH_SIZE: bs}

    def _log_reconstructed_images(self, system, x, x_recon, suffix="train"):
        x_12 = x[:4].view(-1, 3, 112, 112)
        x_12_recon = x_recon[:4].contiguous().view(-1, 3, 112, 112)
        x_12 = torch.cat(
            (x_12, x_12_recon), dim=2
        )  # this needs to stack the images differently
        datapoints = make_grid(
            x_12, nrow=self.sequence_length, range=(-1, 1), normalize=True
        )
        system.logger.experiment.add_image(
            f"reconstruction/{suffix}",
            datapoints,
            dataformats="CHW",
            global_step=system.global_step,
        )

    @staticmethod
    def _metric_dict_to_tensorboard_log(metric_dict: dict, suffix="train"):
        _metric_dict = {}
        for key, value in metric_dict.items():
            _metric_dict[key] = {suffix: value}
        return _metric_dict

    def _calculate_metrics(self, batch_size=1, **kwargs):
        reconstruction_loss_dict = self._calculate_batched_reconstruction_loss(
            batch_size, kwargs
        )
        reconstruction_loss = reduce(torch.add, reconstruction_loss_dict.values())

        classification_loss = self.loss(kwargs[PRED], kwargs[TARGET])

        if not torch.isnan(classification_loss):
            loss = reconstruction_loss + classification_loss
        else:
            loss = reconstruction_loss

        pred = F.softmax(kwargs[PRED], dim=1)

        with torch.no_grad():
            acc_mean = self.calculate_accuracy(pred, kwargs[TARGET])

        return {
            RECONSTRUCTION_LOSS: reconstruction_loss,
            ACC: acc_mean,
            CLASSIFICATION_LOSS: classification_loss,
            LOSS: loss,
            **reconstruction_loss_dict,
        }

    def _calculate_batched_reconstruction_loss(self, batch_size, kwargs):
        reconstruction_loss = {}
        recon_x, x = kwargs[RECON_X], kwargs[X]
        recon_x, x = (
            recon_x.reshape(batch_size, -1, *recon_x.shape[-3:]),
            x.reshape(batch_size, -1, *x.shape[-3:]),
        )
        for _recon_x, _x in zip(recon_x, x):
            _reconstruction_loss = self.reconstruction_loss(_recon_x, _x)
            for loss, value in _reconstruction_loss.items():
                try:
                    reconstruction_loss[loss] += value / recon_x.shape[0]
                except KeyError:
                    reconstruction_loss[loss] = value / recon_x.shape[0]
        return reconstruction_loss


class GeneralVAE(GeneralAE, ABC):
    def __init__(self, num_classes, sequence_length, contains_dropout):
        super().__init__(num_classes, sequence_length, contains_dropout)

    def _calculated_kld(self, mu, logvar, mu_scaling) -> torch.Tensor:
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld /= mu_scaling
        return kld

    def _calculate_metrics(self, **kwargs):
        # here we only need to calculate the kld, the rest is done in AE
        metric_dict = super()._calculate_metrics(**kwargs)
        kld = self._calculated_kld(kwargs[MU], kwargs[LOG_VAR], len(kwargs[X].view(-1)))
        metric_dict[LOSS] += kld
        metric_dict[KLD_LOSS] = kld
        return metric_dict

    @staticmethod
    def _transform_output_dict(output: dict):
        network_output = [x[PRED] for x in output]
        x_recon = torch.cat([x[RECON_X] for x in network_output], 0)
        pred = torch.cat([x[PRED] for x in network_output], 0)
        mu = torch.cat([x[MU] for x in network_output], 0)
        log_var = torch.cat([x[LOG_VAR] for x in network_output], 0)

        target = torch.cat([x[TARGET] for x in output], 0)
        x = torch.cat([x[X] for x in output], 0)
        return {
            RECON_X: x_recon,
            PRED: pred,
            MU: mu,
            LOG_VAR: log_var,
            TARGET: target,
            X: x,
        }

    def _reparametrize(self, mu: Variable, logvar: Variable) -> Variable:

        if self.training:
            # multiply log variance with 0.5, then in-place exponent
            # yielding the standard deviation

            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            sample_z = eps.mul(std).add_(mu)

            return sample_z

        else:
            return mu
