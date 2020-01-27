import abc
from abc import ABC
from typing import Tuple

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.utils import make_grid

from forgery_detection.lightning.utils import VAL_ACC


class LightningModel(nn.Module):
    def __init__(self, num_classes, sequence_length, contains_dropout):
        super().__init__()
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.contains_dropout = contains_dropout

    def loss(self, logits, labels):
        raise NotImplementedError()

    def training_step(self, batch, batch_nb, system):
        raise NotImplementedError()

    def aggregate_outputs(self, outputs, system):
        raise NotImplementedError()

    def calculate_accuracy(self, pred, target):
        raise NotImplementedError()

    @staticmethod
    def _set_requires_grad_for_module(module, requires_grad=False):
        for param in module.parameters():
            param.requires_grad = requires_grad


class SequenceClassificationModel(LightningModel):
    def __init__(self, num_classes, sequence_length, contains_dropout):
        super().__init__(
            num_classes, sequence_length, contains_dropout=contains_dropout
        )

    def forward(self, x):
        raise NotImplementedError()

    def loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def calculate_accuracy(self, pred, target):
        labels_hat = torch.argmax(pred, dim=1)
        acc = labels_hat.eq(target).float().mean()
        return acc

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


class GeneralVAE(LightningModel, ABC):
    def __init__(self, num_classes, sequence_length, contains_dropout):
        super().__init__(num_classes, sequence_length, contains_dropout)

    @abc.abstractmethod
    def encode(self, x):
        raise NotImplementedError()

    @abc.abstractmethod
    def decode(self, z):
        raise NotImplementedError()

    def training_step(self, batch, batch_nb, system):
        x, target = batch
        recon_x, pred, mu, logvar = self.forward(x)

        BCE, KLD, acc_mean, classification_loss, loss = self._calculate_metrics(
            logvar, mu, pred, recon_x, target, x
        )

        lightning_log = {"loss": loss}
        tensorboard_log = {
            "loss": {"train": loss},
            "reconstruction_loss": {"train": BCE},
            "kld_loss": {"train": KLD},
            "classification_loss": {"train": classification_loss},
            "acc": {"train": acc_mean},
        }
        return tensorboard_log, lightning_log

    def aggregate_outputs(self, outputs, system):
        network_output = [x["pred"] for x in outputs]
        x_recon = torch.cat([x[0] for x in network_output], 0)
        pred = torch.cat([x[1] for x in network_output], 0)
        mu = torch.cat([x[2] for x in network_output], 0)
        logvar = torch.cat([x[3] for x in network_output], 0)

        target = torch.cat([x["target"] for x in outputs], 0)
        x = torch.cat([x["x"] for x in outputs], 0)

        BCE, KLD, acc_mean, classification_loss, loss = self._calculate_metrics(
            logvar, mu, pred, x_recon, target, x
        )

        # confusion matrix
        class_accuracies = system.log_confusion_matrix(target.cpu(), pred.cpu())

        tensorboard_log = {
            "loss": loss,
            "reconstruction_loss": BCE,
            "kld_loss": KLD,
            "classification_loss": classification_loss,
            "acc": acc_mean,
            "class_acc": class_accuracies,
        }
        lightning_log = {VAL_ACC: acc_mean}

        # log 10 images
        x_12 = x[:3].view(-1, 3, 112, 112)
        x_12_recon = x_recon[:3].view(-1, 3, 112, 112)
        x_12 = torch.cat((x_12, x_12_recon), dim=0)
        datapoints = make_grid(x_12, nrow=12, range=(-1, 1), normalize=True)
        system.logger.experiment.add_image(
            "reconstruction",
            datapoints,
            dataformats="CHW",
            global_step=system.global_step,
        )

        return tensorboard_log, lightning_log

    def vae_loss(self, recon_x, x, mu, logvar) -> Tuple[torch.Tensor, torch.Tensor]:
        BCE = self.reconstruction_loss(recon_x, x)
        # BCE = F.binary_cross_entropy(recon_x, x)

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= len(x.view(-1))

        return BCE, KLD

    @staticmethod
    def reconstruction_loss(recon_x, x):
        return F.l1_loss(recon_x, x)

    def loss(self, logits, labels):
        raise NotImplementedError()

    def _calculate_metrics(self, logvar, mu, pred, recon_x, target, x):
        BCE, KLD = self.vae_loss(recon_x, x, mu, logvar)
        classifiaction_loss = self.loss(pred, target)
        loss = BCE + KLD + classifiaction_loss
        pred = F.softmax(pred, dim=1)
        acc_mean = self.calculate_accuracy(pred, target)
        return BCE, KLD, acc_mean, classifiaction_loss, loss

    def calculate_accuracy(self, pred, target):
        labels_hat = torch.argmax(pred, dim=1)
        acc = labels_hat.eq(target).float().mean()
        return acc

    def reparametrize(self, mu: Variable, logvar: Variable) -> Variable:

        if self.training:
            # multiply log variance with 0.5, then in-place exponent
            # yielding the standard deviation

            std = logvar.mul(0.5).exp_()  # type: Variable
            eps = Variable(std.data.new(std.size()).normal_())
            sample_z = eps.mul(std).add_(mu)

            return sample_z

        else:
            return mu

    @abc.abstractmethod
    def forward(
        self, x
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError()
