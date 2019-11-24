import torch
from torch import nn
from torch.nn import functional as F

from forgery_detection.models.utils import PretrainedResnet18


class Resnet18MultiClassDropout(PretrainedResnet18):
    def __init__(self):
        super().__init__(num_classes=5, sequence_length=1, contains_dropout=True)

        self.resnet.layer1 = nn.Sequential(nn.Dropout2d(0.1), self.resnet.layer1)
        self.resnet.layer2 = nn.Sequential(nn.Dropout2d(0.2), self.resnet.layer2)
        self.resnet.layer3 = nn.Sequential(nn.Dropout2d(0.3), self.resnet.layer3)
        self.resnet.fc = nn.Sequential(nn.Dropout(0.5), self.resnet.fc)


class Resnet18MultiHead(Resnet18MultiClassDropout):
    def __init__(self):
        super().__init__()
        self.resnet.layer4_0 = nn.Linear(256, 2)
        self.resnet.layer4_1 = nn.Linear(256, 2)
        self.resnet.layer4_2 = nn.Linear(256, 2)
        self.resnet.layer4_3 = nn.Linear(256, 2)
        self.resnet.layer4_4 = nn.Linear(256, 2)

        self.resnet.layer4_out = nn.Linear(5 * 2, 256)

    def forward(self, x):
        x = self.resnet.conv1(x).squeeze(2)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        x_0 = self.resnet.layer4_0(x)
        x_1 = self.resnet.layer4_1(x)
        x_2 = self.resnet.layer4_2(x)
        x_3 = self.resnet.layer4_3(x)
        x_4 = self.resnet.layer4_4(x)

        x_out = self.resnet.layer4_out(torch.cat([x_0, x_1, x_2, x_3, x_4], dim=1))
        x = self.resnet.relu(x_out)

        x = self.resnet.fc(x)

        return x, [x_0, x_1, x_2, x_3, x_4]

    def loss(self, logits, labels):
        loss = F.cross_entropy(logits[0], labels)

        for idx, x in enumerate(logits[1]):
            loss += F.cross_entropy(x, labels.eq(idx).long()) / 5

        return loss

    def calculate_accuracy(self, pred, target):
        labels_hat = torch.argmax(pred[0], dim=1)
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

            tensorboard_log["roc_auc"] = system.multiclass_roc_auc_score(
                target, pred[0]
            )
        return tensorboard_log, lightning_log

    def aggregate_outputs(self, outputs, system):
        pred = (
            torch.cat([x["pred"][0] for x in outputs], 0).cpu(),
            [
                torch.cat([x[y] for x in [x["pred"][1] for x in outputs]]).cpu()
                for y in range(5)
            ],
        )
        target = torch.cat([x["target"] for x in outputs], 0).cpu()

        loss_mean = self.loss(pred, target)
        acc_mean = self.calculate_accuracy(pred, target)

        # confusion matrix
        class_accuracies = system.log_confusion_matrix(target, pred[0])

        # roc_auc_score
        system.log_roc_graph(target, pred[0])

        roc_auc = system.multiclass_roc_auc_score(target, pred[0])

        tensorboard_log = {
            "loss": loss_mean,
            "acc": acc_mean,
            "roc_auc": roc_auc,
            "class_acc": class_accuracies,
        }
        lightning_log = {"acc": acc_mean}

        return tensorboard_log, lightning_log
