import logging

import torch
from torch import nn
from torchvision.utils import make_grid

from forgery_detection.data.utils import irfft
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

        return z

    def forward(self, f):
        # input is b x 2 x c x w x h -> combine fourier and colour channels
        f = f.reshape((f.shape[0], -1, *f.shape[-2:]))
        f = torch.tanh(f)
        # encode
        f = self.encode(f)
        # decode
        d = self.decode(f)

        # add the fourier channel
        d = d.reshape((d.shape[0], 2, 3, *d.shape[-2:]))
        return {
            RECON_X: d,
            PRED: torch.ones((f.shape[0], self.num_classes), device=f.device),
        }

    def reconstruction_loss(self, recon_x, x):
        # todo experiment what is better:
        # l1 loss on raw data, or on tanhd data
        return {"l1_loss": self.l1_loss(recon_x, x)}
        # return {"l1_loss": self.l1_loss(torch.tanh(recon_x), torch.tanh(x))}

    @staticmethod
    def _inverse_tanh(x):
        return torch.log((x + 1) / (1 - x)) / 2

    def _log_reconstructed_images(self, system, x, x_recon, suffix="train"):
        x_12 = x[:4].view(-1, *x.shape[-4:])
        x_12_recon = x_recon[:4].contiguous().view(-1, *x.shape[-4:])

        x_12 = irfft(x_12)
        x_12_recon = irfft(x_12_recon)

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
