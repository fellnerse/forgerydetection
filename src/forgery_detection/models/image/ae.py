import logging

import torch
import torch.nn.functional as F
from torch import nn

from forgery_detection.models.image.utils import ConvBlock
from forgery_detection.models.utils import GeneralAE
from forgery_detection.models.utils import PRED
from forgery_detection.models.utils import RECON_X
from forgery_detection.models.video.vgg import Vgg16

logger = logging.getLogger(__file__)


class SimpleAE(GeneralAE):
    def __init__(self, *args, **kwargs):
        super(SimpleAE, self).__init__(
            num_classes=5, sequence_length=1, contains_dropout=False
        )

        # Encoder
        self.block1 = ConvBlock(3, 64, (3, 3), 1, 1)  # 64
        self.block2 = ConvBlock(64, 128, (3, 3), 1, 1)  # 32
        self.block3 = ConvBlock(128, 256, (3, 3), 1, 1)  # 16
        self.block4 = ConvBlock(256, 16, (3, 3), 1, 1)  # 8

        # Decoder
        self.fct_decode = nn.Sequential(
            nn.Conv2d(16, 64, (3, 3), padding=1),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 16
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 32
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 64
            nn.Conv2d(64, 16, (3, 3), padding=1),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 128
        )

        self.final_decod_mean = nn.Conv2d(16, 3, (3, 3), padding=1)

    def encode(self, x):
        """return mu_z and logvar_z"""

        x = F.elu(self.block1(x))
        x = F.elu(self.block2(x))
        x = F.elu(self.block3(x))
        x = F.elu(self.block4(x))

        return x

    def decode(self, z):

        z = self.fct_decode(z)
        z = self.final_decod_mean(z)
        z = torch.tanh(z)

        return z

    def forward(self, x):
        x = self.encode(x)

        return {
            RECON_X: self.decode(x),
            PRED: torch.ones((x.shape[0], self.num_classes), device=x.device),
        }

    def reconstruction_loss(self, recon_x, x):
        return F.binary_cross_entropy_with_logits(recon_x, torch.sigmoid(x))

    def loss(self, logits, labels):
        return torch.zeros((1,), device=logits.device)


class SimpleAEVGG(SimpleAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vgg = Vgg16(requires_grad=False)
        self._set_requires_grad_for_module(self.vgg)

    def reconstruction_loss(self, recon_x, x):
        features_y = self.vgg(recon_x.view(-1, 3, 112, 112))
        features_x = self.vgg(x.view(-1, 3, 112, 112))

        return F.mse_loss(features_y, features_x)


class SimpleAEL1(SimpleAE):
    def reconstruction_loss(self, recon_x, x):

        return F.l1_loss(recon_x, x)
