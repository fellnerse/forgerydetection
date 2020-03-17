import logging

import torch
from torch import nn
from torchvision.utils import make_grid

from forgery_detection.data.utils import irfft
from forgery_detection.models.image.ae import BiggerAE
from forgery_detection.models.image.ae import SimpleAE
from forgery_detection.models.image.utils import ConvBlock
from forgery_detection.models.mixins import FourierLoggingMixin
from forgery_detection.models.mixins import L1LossMixin
from forgery_detection.models.mixins import PretrainedNet
from forgery_detection.models.mixins import SupervisedNet
from forgery_detection.models.utils import PRED
from forgery_detection.models.utils import RECON_X

logger = logging.getLogger(__file__)


class FrequencyAE(FourierLoggingMixin, SimpleAE, L1LossMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block1 = ConvBlock(6, 64, (3, 3), 1, 1)
        self.final_decod_mean = nn.Conv2d(16, 6, (3, 3), padding=1)

        self.log_image_count = 10
        self.log_images_every = 10

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
        return {"l1_loss": self.l1_loss(recon_x, x)}

    def _log_reconstructed_images(self, system, x, x_recon, suffix="train"):
        x = x[:4]
        x_recon = x_recon[:4].contiguous()
        self.circles = self.circles.to(x.device)
        x_frequencies = torch.cat([irfft(x * mask) for mask in self.circles])
        x_recon_frequencies = torch.cat(
            [irfft(x_recon * mask) for mask in self.circles]
        )

        x = torch.cat((x_frequencies, x_recon_frequencies), dim=2)
        datapoints = make_grid(
            x, nrow=self.sequence_length * 4, range=(-1, 1), normalize=True
        )
        system.logger.experiment.add_image(
            f"reconstruction/{suffix}",
            datapoints,
            dataformats="CHW",
            global_step=system.global_step,
        )


class PretrainedFrequencyNet(
    PretrainedNet(
        "/home/sebastian/log/runs/TRAIN/frequency_ae/version_0/checkpoints/_ckpt_epoch_5.ckpt"
    ),
    FrequencyAE,
):
    pass


class FrequencyAEtanh(FrequencyAE):
    def reconstruction_loss(self, recon_x, x):
        return {"l1_loss": self.l1_loss(torch.tanh(recon_x), torch.tanh(x))}


class FrequencyAEMagnitude(FrequencyAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block1 = ConvBlock(3, 64, (3, 3), 1, 1)
        self.final_decod_mean = nn.Conv2d(16, 3, (3, 3), padding=1)

    def forward(self, f):
        magnitude = f[:, 0]
        shift = f[:, 1]
        mag = torch.tanh(magnitude)
        # encode
        mag = self.encode(mag)
        # decode
        d = self.decode(mag)
        # add the fourier channel
        d = torch.stack((d, shift), dim=1)
        return {
            RECON_X: d,
            PRED: torch.ones((f.shape[0], self.num_classes), device=f.device),
        }

    def reconstruction_loss(self, recon_x, x):
        return {"l1_loss": self.l1_loss(recon_x[:, 0], x[:, 0])}


class FrequencyAEcomplex(FrequencyAE):
    def reconstruction_loss(self, recon_x, x):
        loss = torch.mean(torch.sqrt(torch.sum((recon_x - x) ** 2, dim=-1)))
        return {"complex_loss": loss}


class BiggerFrequencyAE(FourierLoggingMixin, BiggerAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block1 = ConvBlock(6, 64, (3, 3), 1, 1)
        self.final_decod_mean = nn.Conv2d(16, 6, (3, 3), padding=1)

        self.log_image_count = 10
        self.log_images_every = 10

    def decode(self, z):

        z = self.fct_decode(z)
        z = self.final_decod_mean(z)

        return z

    def forward(self, f):
        # input is b x 2 x c x w x h -> combine fourier and colour channels
        f = f.reshape((f.shape[0], -1, *f.shape[-2:]))
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
        loss = torch.mean(torch.sqrt(torch.sum((recon_x - x) ** 2, dim=-1)))
        return {"complex_loss": loss}

    def _log_reconstructed_images(self, system, x, x_recon, suffix="train"):
        x = x[:4]
        x_recon = x_recon[:4].contiguous()
        self.circles = self.circles.to(x.device)
        x_frequencies = torch.cat([irfft(x * mask) for mask in self.circles])
        x_recon_frequencies = torch.cat(
            [irfft(x_recon * mask) for mask in self.circles]
        )

        x = torch.cat((x_frequencies, x_recon_frequencies), dim=2)
        datapoints = make_grid(
            x, nrow=self.sequence_length * 4, range=(-1, 1), normalize=True
        )
        system.logger.experiment.add_image(
            f"reconstruction/{suffix}",
            datapoints,
            dataformats="CHW",
            global_step=system.global_step,
        )


class BiggerFrequencyAElog(BiggerFrequencyAE):
    def forward(self, f):
        # input is b x 2 x c x w x h -> combine fourier and colour channels
        f = f.reshape((f.shape[0], -1, *f.shape[-2:]))
        f = self.logify(f)
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

    def logify(self, x: torch.Tensor):
        return x.sign() * torch.log(x.abs() + 1)


class SupervisedBiggerFrequencyAE(
    SupervisedNet(16 * 7 * 7, num_classes=5), BiggerFrequencyAE
):
    def forward(self, f):
        # input is b x 2 x c x w x h -> combine fourier and colour channels
        f = f.reshape((f.shape[0], -1, *f.shape[-2:]))
        # encode
        h = self.encode(f)
        # decode
        d = self.decode(h)

        # add the fourier channel
        d = d.reshape((d.shape[0], 2, 3, *d.shape[-2:]))
        return {RECON_X: d, PRED: self.classifier(h.flatten(1))}

    def loss(self, logits, labels):
        return super().loss(logits, labels)
