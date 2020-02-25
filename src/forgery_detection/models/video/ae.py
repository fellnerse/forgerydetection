import logging

import torch
from torch import nn
from torchvision.models.video.resnet import Conv3DNoTemporal

from forgery_detection.models.mixins import L1LossMixin
from forgery_detection.models.mixins import PretrainedNet
from forgery_detection.models.mixins import SupervisedNet
from forgery_detection.models.mixins import VGGLossMixin
from forgery_detection.models.utils import GeneralAE
from forgery_detection.models.utils import PRED
from forgery_detection.models.utils import RECON_X
from forgery_detection.models.video.vae import Encoder
from forgery_detection.models.video.vae import StemSample
from forgery_detection.models.video.vae import Transpose
from forgery_detection.models.video.vae import UpBlockSample

logger = logging.getLogger(__file__)


class VideoAE2(GeneralAE, VGGLossMixin, L1LossMixin):
    def __init__(
        self,
        num_classes=5,
        sequence_length=8,
        upblock=UpBlockSample,
        stem=StemSample,
        activation=nn.ReLU,
        *args,
        **kwargs,
    ):
        super().__init__(
            num_classes=num_classes,
            sequence_length=sequence_length,
            contains_dropout=False,
            log_images_every=10,
        )

        self.encoder = Encoder(self.sequence_length)
        self.encoder.mc3.reduce = nn.Sequential(
            Conv3DNoTemporal(256, 8, stride=2), nn.BatchNorm3d(8), nn.ELU()
        )
        # self.encoder.mc3.reduce.add_module(
        #     "ae_conv3d", Conv3DNoTemporal(16, 8, stride=2)
        # )
        # self.encoder.mc3.reduce.add_module("ae_bn3d", nn.BatchNorm3d(8))
        # self.encoder.mc3.reduce.add_module("ae_elu", nn.ELU())

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
        return {
            "l1_loss": self.l1_loss(recon_x, x),
            "perceptual_loss": self.full_loss(recon_x, x, slices=2) / 4,
        }

    def loss(self, logits, labels):
        return torch.zeros((1,), device=logits.device)


class PretrainedVideoAE(
    PretrainedNet(
        "/mnt/raid5/sebastian/model_checkpoints/avspeech_ff_100/video/ae/"
        "l1+vgg/bigger_latent_space_further.ckpt"
    ),
    VideoAE2,
):
    pass


class SupervisedVideoAE(
    SupervisedNet(input_units=8 * 8 * 7 * 7, num_classes=5), PretrainedVideoAE
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        h = self.encode(x)
        return {RECON_X: self.decode(h), PRED: self.classifier(h.flatten(1))}

    def loss(self, logits, labels):
        return super().loss(logits, labels)


class SmallerVideoAE(PretrainedVideoAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder.mc3.reduce = nn.Sequential(
            Conv3DNoTemporal(256, 16, stride=1), nn.BatchNorm3d(16), nn.ELU()
        )
        self.decoder.__delitem__(0)
        self.decoder.__setitem__(
            0,
            nn.Sequential(
                Conv3DNoTemporal(16, 128, stride=1), nn.BatchNorm3d(128), nn.ELU()
            ),
        )
