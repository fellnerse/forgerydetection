# https://github.com/atinghosh/VAE-pytorch/blob/master/VAE_celeba.py
import logging
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from forgery_detection.models.utils import GeneralVAE


logger = logging.getLogger(__file__)


class SimpleVAE(GeneralVAE):
    class ConvBlock(nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding,
            stride,
            pool_kernel_size=(2, 2),
        ):
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size, padding, stride
            )
            self.conv2 = nn.Conv2d(
                out_channels, out_channels, kernel_size, padding, stride
            )
            self.pool = nn.MaxPool2d(pool_kernel_size)

        def forward(self, x):
            x = F.elu(self.conv1(x))
            x = F.elu(self.conv2(x))
            x = self.pool(x)

            return x

    def __init__(self, *args, **kwargs):
        super(SimpleVAE, self).__init__(
            num_classes=5, sequence_length=1, contains_dropout=False
        )

        # Encoder
        self.block1 = self.ConvBlock(3, 64, (3, 3), 1, 1)  # 64
        self.block2 = self.ConvBlock(64, 128, (3, 3), 1, 1)  # 32
        self.block3 = self.ConvBlock(128, 256, (3, 3), 1, 1)  # 16
        self.block4 = self.ConvBlock(256, 32, (3, 3), 1, 1)  # 8

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

    def forward(
        self, x
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)

        return (
            self.decode(z),
            torch.ones((x.shape[0], self.num_classes), device=x.device),
            mu,
            logvar,
        )


class SupervisedVae(SimpleVAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.classifier = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        # todo should classify only on mu
        return self.decode(z), self.classifier(torch.flatten(z, 1)), mu, logvar
