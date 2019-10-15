import os

import ray
import torch
import torch.nn.functional as F
from ray.tune import Trainable
from torch import nn
from torch import optim
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
        self.epoch_size = config.get("epoch_size")
        self.test_size = config.get("test_size")
        self.batch_size = config.get("batch_size")
        use_cuda = config.get("use_gpu") and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.train_loader, self.test_loader = ray.get(
            [config.get("train_loader_id"), config.get("test_loader_id")]
        )
        self.model = VGG11Binary().to(self.device)
        self.reset_config(config)

    def _train(self):
        # train
        total_loss = 0
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if batch_idx * len(data) > self.epoch_size:
                break
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()

            # total_loss += loss.item()

        # test
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                if batch_idx * len(data) > self.test_size:
                    break
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        acc = correct / total
        return {
            "mean_accuracy": acc,
            "total_loss": total_loss,
            "lr": self.lr,
            "decay": self.decay,
        }

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))

    def reset_config(self, new_config):
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=new_config.get("lr", 0.01),
            momentum=new_config.get("decay", 0.01),
        )
        # optim.Adam(
        #     self.model.parameters(),
        #     lr=new_config.get("lr", 0.01),
        #     weight_decay=new_config.get("decay"),
        # )
        self.lr = new_config.get("lr", 0.01)
        self.decay = new_config.get("decay")
        return True
