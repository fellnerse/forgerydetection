from collections import OrderedDict

import pytorch_lightning as pl
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from forgery_detection.data.face_forensics.utils import get_data
from forgery_detection.models.simple_vgg import VGG11Binary


class SupervisedSystem(pl.LightningModule):
    def __init__(self, train_data_dir, val_data_dir, batch_size=128):
        super(SupervisedSystem, self).__init__()
        self.batch_size = batch_size
        self.train_data = get_data(train_data_dir)
        self.val_data = get_data(val_data_dir)

        # self.model = SqueezeBinary()
        self.model = VGG11Binary()

    def forward(self, x):
        return self.model.forward(x)

    def loss(self, logits, labels):
        cross_engropy = F.cross_entropy(logits, labels)
        return cross_engropy

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss_val = self.loss(y_hat, y)
        train_acc = self.calculate_accuracy(y_hat, y)

        tensorboard_dict = {"train_loss": loss_val, "train_acc": train_acc}
        output = OrderedDict({"loss": loss_val, "log": tensorboard_dict})

        return output

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)

        loss_val = self.loss(y_hat, y)
        val_acc = self.calculate_accuracy(y_hat, y)

        output = {"val_loss": loss_val, "val_acc": val_acc}
        output["log"] = output

        return output

    def validation_end(self, outputs):
        # OPTIONAL
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_acc_mean = torch.stack([x["val_acc"] for x in outputs]).mean()
        tensorboard_dict = {"val_loss": val_loss_mean, "val_acc": val_acc_mean}
        result = {"log": tensorboard_dict, "val_loss": val_loss_mean}
        return result

    def calculate_accuracy(self, y_hat, y):
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)
        return val_acc

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=10e-6)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, verbose=True, patience=2
        )
        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=8
        )

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=True, num_workers=8
        )
