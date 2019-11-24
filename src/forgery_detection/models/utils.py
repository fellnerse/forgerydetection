import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


class SequenceClassificationModel(nn.Module):
    def __init__(self, num_classes, sequence_length, contains_dropout=False):
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


class PretrainedResnet18(SequenceClassificationModel):
    def __init__(self, num_classes, sequence_length, contains_dropout=False):
        super().__init__(
            num_classes, sequence_length, contains_dropout=contains_dropout
        )

        self.resnet = resnet18(pretrained=True, num_classes=1000)

        self.resnet.layer4 = nn.Identity()
        self.resnet.fc = nn.Linear(256, num_classes)
        self.eval()

    def forward(self, x):
        return self.resnet.forward(x)

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
        lightning_log = {"acc": acc_mean}

        return tensorboard_log, lightning_log
