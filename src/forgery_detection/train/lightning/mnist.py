import pytorch_lightning as pl
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader

from forgery_detection.data.face_forensics.utils import get_data
from forgery_detection.models.simple_vgg import VGG11Binary


class CoolSystem(pl.LightningModule):
    def __init__(self, train_data_dir, val_data_dir, batch_size=128):
        super(CoolSystem, self).__init__()
        self.batch_size = batch_size
        self.train_data = get_data(train_data_dir)
        self.val_data = get_data(val_data_dir)

        # self.model = SqueezeBinary()
        self.model = VGG11Binary()

    def forward(self, x):
        return self.model.forward(x)

    def make_one_hot(self, labels, C=2):
        one_hot = torch.cuda.FloatTensor(
            labels.size(0), C, labels.size(2), labels.size(3)
        ).zero_()
        target = one_hot.scatter_(1, labels.data, 1)

        target = Variable(target)

        return target

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)

        correct = torch.zeros(1)
        total = 0
        _, predicted = torch.max(y_hat.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

        acc = correct / total

        return {
            "val_loss": F.cross_entropy(y_hat, y),
            "acc": acc,
            "log": {"val_acc": acc},
        }

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        acc = torch.stack([x["acc"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss, "acc": acc}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=0.00001)

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
