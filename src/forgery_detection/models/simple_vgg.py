import os

import ray
import torch
import torch.nn.functional as F
from ray.tune import Trainable
from torch import nn
from torchvision import models


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
        self.settings = config["settings"]
        try:
            use_cuda = self.settings["use_gpu"] and torch.cuda.is_available()
        except KeyError:
            use_cuda = False
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.model = config["model"]().to(self.device)
        self.hyper_parameter = config["hyper_parameter"]
        self._initialize_optimizer(
            config["optimizer"], self.hyper_parameter[("optimizer")]
        )

        self.train_loader, self.test_loader = ray.get(
            [self.settings["train_loader_id"], self.settings["test_loader_id"]]
        )

    def _train(self):
        # train
        total_loss = torch.zeros(1)
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if batch_idx * len(data) > self.settings["epoch_size"]:
                break
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)  # todo this seems to be wrong
            loss.backward()
            self.optimizer.step()

            total_loss -= loss.item()

        # test
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                if batch_idx * len(data) > self.settings["test_size"]:
                    break
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        acc = correct / total
        return {
            "mean_accuracy": acc,
            "total_loss": total_loss.mean().item(),
            "hyper_parameter": self.hyper_parameter,
        }

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))

    def _initialize_optimizer(self, optimizer_cls, optimizer_config: dict):
        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_config)

    def reset_config(self, new_config):
        self.hyper_parameter = new_config["hyper_parameter"]
        self._initialize_optimizer(
            new_config["optimizer"], self.hyper_parameter["optimizer"]
        )
        return True
