import logging

import torch
from torch import nn
from torchvision.utils import make_grid

from forgery_detection.models.image.ae import SimpleAE
from forgery_detection.models.image.utils import ConvBlock
from forgery_detection.models.mixins import L1LossMixin
from forgery_detection.models.mixins import PretrainedNet
from forgery_detection.models.utils import PRED
from forgery_detection.models.utils import RECON_X

logger = logging.getLogger(__file__)


class FrequencyAE(SimpleAE, L1LossMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, log_images_every=10, **kwargs)
        self.block1 = ConvBlock(6, 64, (3, 3), 1, 1)
        self.final_decod_mean = nn.Conv2d(16, 6, (3, 3), padding=1)

    def decode(self, z):

        z = self.fct_decode(z)
        z = self.final_decod_mean(z)
        # z = z.reshape((z.shape[0], 2, 3, *z.shape[-2:]))
        # z = z.permute((0, 2, 3, 4, 1))

        return z

    def forward(self, f):
        # transform into fourier space
        # f = torch.rfft(x, 3, onesided=False, normalized=False)
        # put fourier dim in front of image
        # f = f.permute((0, 4, 1, 2, 3))
        # f = f.reshape((x.shape[0], -1, *x.shape[-2:]))
        # apply tanh
        f = torch.tanh(f)
        # encode
        f = self.encode(f)
        # decode
        d = self.decode(f)
        # d = self._inverse_tanh(d) # todo do i need this here? does it make sense?
        # maybe the network should be able to learn this

        # d = torch.irfft(d, 3, onesided=False, normalized=False) # I should do this in logging
        # d = torch.tanh(d)
        return {
            RECON_X: d,
            PRED: torch.ones((f.shape[0], self.num_classes), device=f.device),
            # PRED: torch.ones((x.shape[0], self.num_classes), device=x.device),
        }

    def reconstruction_loss(self, recon_x, x):
        return {"l1_loss": self.l1_loss(recon_x, x)}

    @staticmethod
    def _inverse_tanh(x):
        return torch.log((x + 1) / (1 - x)) / 2

    def _log_reconstructed_images(self, system, x, x_recon, suffix="train"):
        x_12 = x[:4].view(-1, 2, 3, 112, 112).permute((0, 2, 3, 4, 1))
        x_12_recon = (
            x_recon[:4].contiguous().view(-1, 2, 3, 112, 112).permute((0, 2, 3, 4, 1))
        )

        x_12 = torch.irfft(x_12, 3, onesided=False, normalized=False)
        x_12_recon = torch.irfft(x_12_recon, 3, onesided=False, normalized=False).clamp(
            -1, 1
        )

        x_12 = torch.cat(
            (x_12, x_12_recon), dim=2
        )  # this needs to stack the images differently
        datapoints = make_grid(
            x_12, nrow=self.sequence_length, range=(-1, 1), normalize=True
        )
        system.logger.experiment.add_image(
            f"reconstruction/{suffix}",
            datapoints,
            dataformats="CHW",
            global_step=system.global_step,
        )

    # todo implement loss function here


class PretrainedFrequencyNet(
    PretrainedNet(
        "/home/sebastian/log/runs/TRAIN/frequency_ae/version_0/checkpoints/_ckpt_epoch_5.ckpt"
    ),
    FrequencyAE,
):
    pass
