import torch
from torch import nn
from torch.nn import functional as F

from forgery_detection.models.audio.similarity_stuff import PretrainedSyncNet
from forgery_detection.models.utils import SequenceClassificationModel


class SyncAudioNetRegularized(SequenceClassificationModel):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__(
            num_classes=num_classes, sequence_length=8, contains_dropout=False
        )
        self.r2plus1 = torch.hub.load(
            "moabitcoin/ig65m-pytorch",
            "r2plus1d_34_8_kinetics",
            num_classes=400,
            pretrained=True,
        )
        self.r2plus1.layer3 = nn.Identity()
        self.r2plus1.layer4 = nn.Identity()
        self.r2plus1.fc = nn.Identity()

        self.sync_net = PretrainedSyncNet()
        self._set_requires_grad_for_module(self.sync_net, requires_grad=False)

        self.relu = nn.ReLU()
        self.out = nn.Sequential(
            nn.Linear(128 + 1024, 50), nn.ReLU(), nn.Linear(50, self.num_classes)
        )

    def forward(self, x):
        # def forward(self, video, audio):
        video, audio = x  # bs x 8 x 3 x 112 x 112 , bs x 8 x 29
        # video = x  # bs x 8 x 3 x 112 x 112 , bs x 8 x 29

        video = video.transpose(1, 2)
        video = self.r2plus1(video)

        # syncnet only uses 5 frames
        audio = audio[:, 2:-1]
        audio = (audio.reshape((audio.shape[0], -1, 13)).unsqueeze(1)).transpose(-2, -1)
        audio = self.sync_net.audio_extractor(audio)

        flat = torch.cat((video, audio), dim=1)
        out = self.out(self.relu(flat))
        return out, (video, audio)

    def weight_loss(self):
        vid_weights = self.out[0].weight[:, :128].std()
        aud_weights = self.out[0].weight[:, 128:].std()
        return torch.norm(vid_weights - aud_weights, 2) * 1e3

    def training_step(self, batch, batch_nb, system):
        x, (target, _) = batch

        pred, embeddings = self.forward(x)
        classification_loss = self.loss(pred, target)
        weight_loss = self.weight_loss()
        lightning_log = {"loss": classification_loss + weight_loss}

        with torch.no_grad():
            train_acc = self.calculate_accuracy(pred, target)
            tensorboard_log = {
                "loss": {"train": classification_loss + weight_loss},
                "classification_loss": classification_loss,
                "weight_loss": weight_loss,
                "acc": {"train": train_acc},
                "vid_std": torch.std(self.out[0].weight[:, :128]),
                "aud_std": torch.std(self.out[0].weight[:, 128:]),
            }

        return tensorboard_log, lightning_log

    def aggregate_outputs(self, outputs, system):
        if len(system.val_dataloader()) > 1:
            outputs = outputs[0]

        with torch.no_grad():
            pred = torch.cat([x["pred"][0] for x in outputs], 0)
            target = torch.cat([x["target"][0] for x in outputs], 0)

            loss_mean_classification = self.loss(pred, target)
            pred = pred.cpu()
            target = target.cpu()
            pred = F.softmax(pred, dim=1)
            acc_mean = self.calculate_accuracy(pred, target)

            # confusion matrix
            class_accuracies = system.log_confusion_matrix(target, pred)

            weight_loss = self.weight_loss()

            tensorboard_log = {
                "loss": loss_mean_classification + weight_loss,
                "acc": acc_mean,
                "class_acc": class_accuracies,
                "classification_loss": loss_mean_classification,
                "weight_loss": weight_loss,
                "vid_std": torch.std(self.out[0].weight[:, :128]),
                "aud_std": torch.std(self.out[0].weight[:, 128:]),
            }
        # if system.global_step > 0:
        self.log_class_loss = True

        return tensorboard_log, {}
