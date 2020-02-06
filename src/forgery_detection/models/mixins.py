from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from forgery_detection.models.video.vgg import Vgg16


class VGGLossMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = Vgg16(requires_grad=False)
        self._set_requires_grad_for_module(self.vgg)

    def vgg_loss(self, recon_x, x):
        features_y = self.vgg(recon_x.view(-1, 3, 112, 112))
        features_x = self.vgg(x.view(-1, 3, 112, 112))

        return F.mse_loss(features_y, features_x)


class L1LossMixin(nn.Module):
    def l1_loss(self, recon_x, x):
        return F.l1_loss(recon_x, x)


def PretrainedNet(path_to_model: str):
    class PretrainedNetMixin(nn.Module):
        _path_to_model = path_to_model

        def __init__(self, *args, **kwargs):
            super().__init__()
            state_dict = torch.load(self._path_to_model)["state_dict"]

            mapped_state_dict = OrderedDict()
            for key, value in state_dict.items():
                mapped_state_dict[key.replace("model.", "")] = value

            self.load_state_dict(mapped_state_dict)

    return PretrainedNetMixin
