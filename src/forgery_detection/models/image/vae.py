# https://github.com/atinghosh/VAE-pytorch/blob/master/VAE_celeba.py
import logging

import torch
import torch.nn.functional as F
from torch import nn

from forgery_detection.models.image.utils import ConvBlock
from forgery_detection.models.utils import GeneralVAE
from forgery_detection.models.utils import LOG_VAR
from forgery_detection.models.utils import MU
from forgery_detection.models.utils import PRED
from forgery_detection.models.utils import RECON_X

logger = logging.getLogger(__file__)


class SimpleVAE(GeneralVAE):
    def __init__(self, *args, **kwargs):
        super(SimpleVAE, self).__init__(
            num_classes=5, sequence_length=1, contains_dropout=False
        )

        # Encoder
        self.block1 = ConvBlock(3, 64, (3, 3), 1, 1)  # 64
        self.block2 = ConvBlock(64, 128, (3, 3), 1, 1)  # 32
        self.block3 = ConvBlock(128, 256, (3, 3), 1, 1)  # 16
        self.block4 = ConvBlock(256, 32, (3, 3), 1, 1)  # 8

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

        return (
            x[:, :16, :, :],
            x[:, 16:, :, :],
        )  # output shape - batch_size x 16 x 8 x 8

    def decode(self, z):

        z = self.fct_decode(z)
        z = self.final_decod_mean(z)
        z = torch.tanh(z)

        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self._reparametrize(mu, logvar)

        return {
            RECON_X: self.decode(z),
            PRED: torch.ones((x.shape[0], self.num_classes), device=x.device),
            MU: mu,
            LOG_VAR: logvar,
        }

    def reconstruction_loss(self, recon_x, x):
        return F.binary_cross_entropy_with_logits(recon_x, torch.sigmoid(x))

    def loss(self, logits, labels):
        return torch.zeros((1,), device=logits.device)


class SupervisedVae(SimpleVAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.classifier = nn.Linear(1024, self.num_classes)

    def loss(self, logits, labels):
        return torch.zeros((1,), device=logits.device)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self._reparametrize(mu, logvar)
        return self.decode(z), self.classifier(torch.flatten(z, 1)), mu, logvar
