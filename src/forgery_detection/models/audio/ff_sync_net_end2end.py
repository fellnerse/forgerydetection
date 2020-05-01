import torch
from torch import nn
from torch.nn import functional as F

from forgery_detection.lightning.logging.confusion_matrix import confusion_matrix
from forgery_detection.models.audio.ff_sync_net import FFSyncNet
from forgery_detection.models.mixins import BinaryEvaluationMixin
from forgery_detection.models.mixins import PretrainedNet
from forgery_detection.models.utils import SequenceClassificationModel


class FFSyncNetEnd2End(BinaryEvaluationMixin, SequenceClassificationModel):
    def __init__(self, num_classes=5, sequence_length=8):
        super().__init__(
            num_classes=num_classes,
            sequence_length=sequence_length,
            contains_dropout=False,
        )

        self.ff_sync_net = FFSyncNet()

        self.out = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(1024 * 2, 50),
            # nn.BatchNorm1d(50),
            # nn.Dropout(p=0.5),
            nn.LeakyReLU(0.02),
            nn.Linear(50, 2),
        )

    def forward(self, x):
        embeddings = self.ff_sync_net(x)
        cat = torch.cat(embeddings, dim=1)
        out = self.out(cat)
        return out, embeddings

    def training_step(self, batch, batch_nb, system):
        x, (target, audio_shift) = batch

        label = target // 4

        # if the model uses dropout we want to calculate the metrics on predictions done
        # in eval mode before training no the samples
        if self.contains_dropout:
            with torch.no_grad():
                self.eval()
                pred_ = self.forward(x)
                self.train()

        pred, embeddings = self.forward(x)
        classification_loss = self.loss(pred, label)
        contrastive_loss = self.ff_sync_net.loss(embeddings, label)
        lightning_log = {"loss": classification_loss + contrastive_loss}

        with torch.no_grad():
            train_acc = self.calculate_accuracy(pred, label)
            tensorboard_log = {
                "loss": {"train": classification_loss + contrastive_loss},
                "classification_loss": classification_loss,
                "constrastive_loss": contrastive_loss,
                "acc": {"train": train_acc},
                "vid_std": torch.std(embeddings[0]),
                "aud_std": torch.std(embeddings[1]),
            }

            class_loss = self.ff_sync_net.loss_per_class(
                embeddings[0], embeddings[1], target
            )
            tensorboard_log["class_loss_train"] = {
                str(idx): val for idx, val in enumerate(class_loss)
            }
            tensorboard_log["class_loss_diff_train"] = {
                str(idx): val - class_loss[4] for idx, val in enumerate(class_loss[:4])
            }

            if self.contains_dropout:
                pred = pred_
                loss_eval = self.loss(pred, label)
                acc_eval = self.calculate_accuracy(pred, label)
                tensorboard_log["loss"]["train_eval"] = loss_eval
                tensorboard_log["acc"]["train_eval"] = acc_eval

        return tensorboard_log, lightning_log

    def aggregate_outputs(self, outputs, system):
        if len(system.val_dataloader()) > 1:
            outputs = outputs[0]

        with torch.no_grad():
            pred = torch.cat([x["pred"][0] for x in outputs], 0)
            label = torch.cat([x["target"][0] for x in outputs], 0)
            target = label // 4

            video_logits = torch.cat([x["pred"][1][0] for x in outputs], 0)
            audio_logtis = torch.cat([x["pred"][1][1] for x in outputs], 0)
            class_loss = self.ff_sync_net.loss_per_class(
                video_logits, audio_logtis, label
            )

            loss_mean_classification = self.loss(pred, target)
            contrastive_loss = self.ff_sync_net.loss(
                (video_logits, audio_logtis), label
            )
            pred = pred.cpu()
            target = target.cpu()
            pred = F.softmax(pred, dim=1)
            acc_mean = self.calculate_accuracy(pred, target)

            # confusion matrix
            cm = confusion_matrix(label, torch.argmax(pred, dim=1), num_classes=5)
            cm = cm[:, :2]  # this is only binary classification
            cm[0] = torch.sum(cm[:-1], dim=0)
            cm[1] = cm[-1]
            accs = cm.diag() / torch.sum(cm[:2, :2], dim=1)
            class_accuracies = system.log_confusion_matrix(label, pred)
            class_accuracies[list(class_accuracies.keys())[0]] = accs[0]
            class_accuracies[list(class_accuracies.keys())[1]] = accs[1]

            tensorboard_log = {
                "loss": loss_mean_classification + contrastive_loss,
                "acc": acc_mean,
                "classification_loss": loss_mean_classification,
                "constrastive_loss": contrastive_loss,
                "class_loss_val": {str(idx): val for idx, val in enumerate(class_loss)},
                "class_loss_diff_val": {
                    str(idx): val - class_loss[4]
                    for idx, val in enumerate(class_loss[:4])
                },
                "vid_std": torch.std(video_logits),
                "aud_std": torch.std(audio_logtis),
            }
        # if system.global_step > 0:
        self.log_class_loss = True

        return tensorboard_log, {}


class FFSyncNetEnd2EndPretrained(
    PretrainedNet(
        "/home/sebastian/log/runs/TRAIN/ff_syncnet_end2end/version_0/checkpoints/_ckpt_epoch_1.ckpt"
    ),
    FFSyncNetEnd2End,
):
    pass


class FFSyncNetEnd2EndSmall(FFSyncNetEnd2End):
    def __init__(self, num_classes=2, sequence_length=8, pretrained=True):
        super().__init__(num_classes=num_classes, sequence_length=sequence_length)

        self.ff_sync_net = FFSyncNet(pretrained=pretrained)

        self.ff_sync_net.video_mlp = nn.Sequential(
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 1024)
        )
        #
        self.out = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(1024 * 2, 50),
            # nn.BatchNorm1d(50),
            # nn.Dropout(p=0.5),
            # nn.ReLU(),
            nn.LeakyReLU(0.02),
            nn.Linear(50, 2),
        )


class FFSyncNetEnd2EndSmallUntrained(FFSyncNetEnd2EndSmall):
    def __init__(self, num_classes=2, sequence_length=8):
        super().__init__(
            num_classes=num_classes, sequence_length=sequence_length, pretrained=False
        )
