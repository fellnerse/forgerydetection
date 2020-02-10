from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from forgery_detection.lightning.utils import NAN_TENSOR
from forgery_detection.models.video.vgg import Vgg16


class VGGLossMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = Vgg16(requires_grad=False)
        self.vgg.eval()
        self._set_requires_grad_for_module(self.vgg, requires_grad=False)

    def vgg_loss(self, recon_x, x):
        features_y = self.vgg(recon_x.view(-1, *recon_x.shape[-3:]))
        features_x = self.vgg(x.view(-1, *x.shape[-3:]))
        return F.mse_loss(features_y, features_x)


class L1LossMixin(nn.Module):
    def l1_loss(self, recon_x, x):
        return F.l1_loss(recon_x, x)


def PretrainedNet(path_to_model: str):
    class PretrainedNetMixin(nn.Module):
        __path_to_model = path_to_model

        def __init__(self, *args, **kwargs):
            super().__init__()
            state_dict = torch.load(self.__path_to_model)["state_dict"]

            mapped_state_dict = OrderedDict()
            for key, value in state_dict.items():
                mapped_state_dict[key.replace("model.", "")] = value

            self.load_state_dict(mapped_state_dict)

    return PretrainedNetMixin


def SupervisedNet(input_units: int, num_classes: int):
    class SupervisedNetMixin(nn.Module):
        __input_units = input_units
        __num_classes = num_classes

        def __init__(self, *args, **kwargs):
            super().__init__()
            self.classifier = nn.Sequential(
                nn.Linear(self.__input_units, 50),
                nn.ReLU(),
                nn.Linear(50, self.__num_classes),
            )

        def loss(self, logits, labels):
            # for now just remove it here
            logits = logits[labels != 5]
            labels = labels[labels != 5]
            if logits.shape[0] == 0:
                return NAN_TENSOR.cuda(device=logits.device)
            return F.cross_entropy(logits, labels)

        def calculate_accuracy(self, pred, target):
            pred = pred[target != 5]
            target = target[target != 5]
            if pred.shape[0] == 0:
                return NAN_TENSOR
            labels_hat = torch.argmax(pred, dim=1)
            acc = labels_hat.eq(target).float().mean()
            return acc

    return SupervisedNetMixin
