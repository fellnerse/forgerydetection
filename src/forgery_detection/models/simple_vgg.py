import os

import torch
import torch.nn.functional as F
from ray.tune import Trainable
from torch import nn
from torch import optim
from torchvision import models
from tqdm import tqdm

from forgery_detection.data.face_forensics.utils import get_data_loaders


class VGG11Binary(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg11_bn = models.vgg11_bn(pretrained=True, num_classes=1000)

        self.vgg11_bn.classifier = nn.Sequential(
            *list(self.vgg11_bn.classifier)[:-1],
            nn.Linear(in_features=4096, out_features=2, bias=True)
        )

    def forward(self, x):
        return self.vgg11_bn.forward(x)


class VGgTrainable(Trainable):
    def _setup(self, config):
        self.epoch_size = config.get("epoch_size")
        self.test_size = config.get("test_size")
        self.batch_size = config.get("batch_size")
        use_cuda = config.get("use_gpu") and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.train_loader, self.test_loader = get_data_loaders(
            batch_size=self.batch_size
        )
        self.model = VGG11Binary().to(self.device)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.get("lr", 0.01),
            momentum=config.get("momentum", 0.9),
        )

    def _train(self):
        # train
        self.model.train()
        for batch_idx, (data, target) in enumerate(
            tqdm(self.train_loader, total=self.epoch_size / self.batch_size)
        ):
            if batch_idx * len(data) > self.epoch_size:
                break
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()

        # test
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(
                tqdm(self.test_loader, total=self.test_size / self.batch_size)
            ):
                if batch_idx * len(data) > self.test_size:
                    break
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        acc = correct / total
        return {"mean_accuracy": acc}

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))
