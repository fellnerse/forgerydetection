import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.video.resnet import Conv3DNoTemporal

from forgery_detection.models.utils import GeneralAE
from forgery_detection.models.utils import PRED
from forgery_detection.models.utils import RECON_X
from forgery_detection.models.video.vae import Encoder
from forgery_detection.models.video.vae import StemSample
from forgery_detection.models.video.vae import Transpose
from forgery_detection.models.video.vae import UpBlockSample


class VideoAE2(GeneralAE):
    def __init__(
        self,
        num_classes=5,
        sequence_length=8,
        upblock=UpBlockSample,
        stem=StemSample,
        activation=nn.ReLU,
        *args,
        **kwargs
    ):
        super(GeneralAE, self).__init__(
            num_classes=num_classes,
            sequence_length=sequence_length,
            contains_dropout=False,
        )

        self.encoder = Encoder(self.sequence_length)
        self.encoder.mc3.reduce.add_module(
            "ae_conv3d", Conv3DNoTemporal(16, 8, stride=2)
        )
        self.encoder.mc3.reduce.add_module("ae_bn3d", nn.BatchNorm3d(8))
        self.encoder.mc3.reduce.add_module("ae_elu", nn.ELU())

        # output: 256 -> mu and sigma are 128

        # input 8 x 8 x 2 x 2
        self.decoder = nn.Sequential(
            stem(8, 128, activation=activation),
            upblock(128, 128, num_conv=3, activation=activation),
            upblock(128, 128, num_conv=3, activation=activation),
            upblock(128, 128, num_conv=3, activation=activation),
            upblock(128, 128, num_conv=3, activation=activation),
            nn.Conv3d(
                128, 3, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
            ),
            Transpose(2, 1),
        )

    def encode(self, x):
        """return mu_z and logvar_z"""
        x = self.encoder(x)
        return x  # output shape - batch_size x 128

    def decode(self, z):
        z = self.decoder(z)
        return z

    def forward(self, x):
        x = self.encode(x)
        return {
            RECON_X: self.decode(x),
            PRED: torch.ones((x.shape[0], self.num_classes), device=x.device),
        }

    def reconstruction_loss(self, recon_x, x):
        return F.l1_loss(recon_x, x)

    def loss(self, logits, labels):
        return torch.zeros((1,), device=logits.device)
