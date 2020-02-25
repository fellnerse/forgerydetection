import logging

import torch
import torch.nn.functional as F
from torch import nn

from forgery_detection.models.image.ae import SimpleAE
from forgery_detection.models.image.utils import ConvBlock
from forgery_detection.models.utils import BATCH_SIZE
from forgery_detection.models.utils import LOSS
from forgery_detection.models.utils import PRED
from forgery_detection.models.utils import RECON_X
from forgery_detection.models.utils import TARGET
from forgery_detection.models.utils import X

logger = logging.getLogger(__file__)

DFD = "dfd"
DFG = "dfg"
DRD = "drd"

DGD = "dgd"
DGG = "dgg"


class Flatten(nn.Module):
    def forward(self, x):
        x = torch.flatten(x, 1)
        return x


class AEGAN(SimpleAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.disc = nn.Sequential(
            ConvBlock(3, 64, (3, 3), 1, 1),
            nn.ELU(),
            ConvBlock(64, 128, (3, 3), 1, 1),
            nn.ELU(),
            ConvBlock(128, 256, (3, 3), 1, 1),
            nn.ELU(),
            ConvBlock(256, 16, (3, 3), 1, 1),
            nn.ELU(),
            ConvBlock(16, 16, (3, 3), 1, 1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        ae_dict = super().forward(x)

        # disc loss
        # 1. discriminator on "real image"
        real_image_disc_output = self.disc(
            x
        )  # this needs to be lossed with bce real label
        # 2. discriminator on "fake image"
        fake_image_disc_output_for_disc = self.disc(
            ae_dict[RECON_X].detach()
        )  # this needs to be lossed with bce fake label

        D_G_z1 = fake_image_disc_output_for_disc.mean()  # this should be around 0.5

        # gen loss
        fake_image_disc_output_for_gen = self.disc(
            ae_dict[RECON_X]
        )  # this needs to be lossed with bce real label
        D_G_z2 = (
            fake_image_disc_output_for_gen.mean()
        )  # this should be around 0.5 as well

        ae_dict[DRD] = real_image_disc_output
        ae_dict[DFD] = fake_image_disc_output_for_disc
        ae_dict[DFG] = fake_image_disc_output_for_gen

        ae_dict[DGD] = D_G_z1
        ae_dict[DGG] = D_G_z2

        return ae_dict

    @staticmethod
    def _transform_output_dict(output: dict):
        network_output = [x[PRED] for x in output]
        bs = network_output[0][PRED].shape[0]
        x_recon = torch.cat([x[RECON_X] for x in network_output], 0)
        pred = torch.cat([x[PRED] for x in network_output], 0)

        drd = torch.cat([x[DRD] for x in network_output], 0)
        dfd = torch.cat([x[DFD] for x in network_output], 0)
        dfg = torch.cat([x[DFG] for x in network_output], 0)
        dgd = torch.stack([x[DGD] for x in network_output], 0)
        dgg = torch.stack([x[DGG] for x in network_output], 0)

        target = torch.cat([x[TARGET] for x in output], 0)
        x = torch.cat([x[X] for x in output], 0)
        return {
            RECON_X: x_recon,
            PRED: pred,
            TARGET: target,
            X: x,
            BATCH_SIZE: bs,
            DRD: drd,
            DFD: dfd,
            DFG: dfg,
            DGD: dgd,
            DGG: dgg,
        }

    def _calculate_metrics(self, batch_size=1, **kwargs):
        metrics_dict = super()._calculate_metrics(**kwargs)

        label_real, label_fake = (
            torch.ones(kwargs[DRD].shape[0], 1),
            torch.zeros(kwargs[DRD].shape[0], 1),
        )
        label_real = label_real.to(kwargs[DRD].device)
        label_fake = label_fake.to(kwargs[DRD].device)

        desc_loss = (
            F.binary_cross_entropy(kwargs[DRD], label_real)
            + F.binary_cross_entropy(kwargs[DFD], label_fake)
        ) / 2
        gen_loss = F.binary_cross_entropy(kwargs[DFG], label_real)
        metrics_dict[LOSS] += desc_loss + gen_loss

        metrics_dict["desc_loss"] = desc_loss
        metrics_dict["gen_loss"] = gen_loss

        metrics_dict[DGD] = torch.mean(kwargs[DGD])
        metrics_dict[DGG] = torch.mean(kwargs[DGG])

        return metrics_dict

    def reconstruction_loss(self, recon_x, x):
        return {"l1_loss": F.l1_loss(recon_x, x)}
