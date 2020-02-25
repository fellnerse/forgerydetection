from abc import ABC
from abc import abstractmethod
from collections import OrderedDict
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.video import mc3_18
from torchvision.models.video.resnet import Conv3DNoTemporal

from forgery_detection.lightning.utils import NAN_TENSOR
from forgery_detection.models.utils import GeneralVAE
from forgery_detection.models.utils import LOG_VAR
from forgery_detection.models.utils import MU
from forgery_detection.models.utils import PRED
from forgery_detection.models.utils import RECON_X


class UpBlockTranspose(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        kernel_size=(1, 3, 3),
        stride=(1, 1, 1),
        padding=(0, 1, 1),
        activation=nn.ReLU,
        num_conv=5,
    ):
        super(UpBlockTranspose, self).__init__()
        self.up = nn.ConvTranspose3d(in_size, out_size, (1, 2, 2), stride=(1, 2, 2))
        self.conv = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv3d(out_size, out_size, kernel_size, stride, padding),
                    activation(),
                )
                for _ in range(num_conv)
            ]
        )
        self.activation = activation

    def forward(self, x):
        up = self.up(x)
        out = self.conv(up)
        return out


class UpBlockSample(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        kernel_size=(1, 3, 3),
        stride=(1, 1, 1),
        padding=(0, 1, 1),
        activation=nn.ReLU,
        num_conv=5,
    ):
        super(UpBlockSample, self).__init__()
        self.up = nn.Upsample(scale_factor=(1, 2, 2), mode="nearest")
        self.conv = nn.Sequential(
            nn.Conv3d(in_size, out_size, kernel_size, stride, padding),
            nn.BatchNorm3d(out_size),
            activation(),
            *[
                nn.Sequential(
                    nn.Conv3d(out_size, out_size, kernel_size, stride, padding),
                    nn.BatchNorm3d(out_size),
                    activation(),
                )
                for _ in range(num_conv - 1)
            ]
        )
        self.activation = activation

    def forward(self, x):
        up = self.up(x)
        out = self.conv(up)
        return out


class Stem(nn.Module, ABC):
    @abstractmethod
    def __init__(self, in_size, out_size, activation):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.activation = activation
        self.stem: Optional[nn.Module] = None

    def forward(self, x):
        return self.stem(x)


class StemTranspose(Stem):
    def __init__(self, in_size, out_size, activation=nn.ReLU):
        super().__init__(
            in_size, out_size, activation
        )  # todo support variable sequence length
        self.stem = nn.Sequential(
            nn.ConvTranspose3d(in_size, out_size, (1, 3, 3), stride=(3, 4, 4)),
            nn.BatchNorm3d(out_size),
            activation(),
        )


class StemSample(Stem):
    def __init__(self, in_size, out_size, activation=nn.ReLU):
        super().__init__(in_size, out_size, activation)
        self.stem = nn.Sequential(
            nn.Upsample(size=(8, 7, 7), mode="nearest"),
            nn.Conv3d(
                in_size,
                out_size // 2,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
            ),
            nn.BatchNorm3d(out_size // 2),
            activation(),
            nn.Conv3d(
                out_size // 2,
                out_size,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
            ),
            nn.BatchNorm3d(out_size),
            activation(),
            nn.Upsample(size=(8, 7, 7), mode="nearest"),
            nn.Conv3d(
                out_size,
                out_size,
                kernel_size=(1, 3, 3),
                stride=(1, 1, 1),
                padding=(0, 1, 1),
            ),
            nn.BatchNorm3d(out_size),
            activation(),
            nn.Conv3d(
                out_size,
                out_size,
                kernel_size=(1, 3, 3),
                stride=(1, 1, 1),
                padding=(0, 1, 1),
            ),
            nn.BatchNorm3d(out_size),
            activation(),
        )


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, input):
        return input.transpose(self.dim0, self.dim1)


class Encoder(nn.Module):
    def __init__(self, sequence_length):
        super().__init__()
        self.sequence_length = sequence_length
        self.mc3 = mc3_18(pretrained=True)
        self.mc3.reduce = nn.Sequential(
            Conv3DNoTemporal(256, 64, stride=2),
            nn.BatchNorm3d(64),
            nn.ELU(),
            Conv3DNoTemporal(64, 32, stride=2),
            nn.BatchNorm3d(32),
            nn.ELU(),
            Conv3DNoTemporal(32, 16, stride=2),
            nn.BatchNorm3d(16),
            nn.ELU(),
        )

        # remove unnecessary stuff
        self.mc3.layer4 = nn.Identity()
        self.mc3.fc = nn.Identity()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.mc3.stem(x)

        x = self.mc3.layer1(x)
        x = self.mc3.layer2(x)
        x = self.mc3.layer3(x)

        x = self.mc3.reduce(x)  # reduce to -1 ,8,  sequence length , 2, 2
        return x


class VideoVae(GeneralVAE):
    def __init__(
        self,
        num_classes=5,
        sequence_length=8,
        upblock=UpBlockTranspose,
        stem=StemTranspose,
        activation=nn.ReLU,
        *args,
        **kwargs
    ):
        super(GeneralVAE, self).__init__(
            num_classes=num_classes,
            sequence_length=sequence_length,
            contains_dropout=False,
        )

        self.encoder = Encoder(self.sequence_length)

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
        return x[:, :8], x[:, 8:]  # output shape - batch_size x 128

    def decode(self, z):
        z = self.decoder(z)
        z = torch.tanh(z)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self._reparametrize(mu, logvar)  # .view(-1, 16, 2, 2, 2)
        return {
            RECON_X: self.decode(z),
            PRED: torch.ones((x.shape[0], self.num_classes), device=x.device),
            MU: mu,
            LOG_VAR: logvar,
        }

    def reconstruction_loss(self, recon_x, x):
        return F.l1_loss(recon_x, x)

    def loss(self, logits, labels):
        return torch.zeros((1,), device=logits.device)


class VideoVaeUpsample(VideoVae):
    def __init__(self, num_classes=5, sequence_length=8, *args, **kwargs):
        super().__init__(
            num_classes=num_classes,
            sequence_length=sequence_length,
            upblock=UpBlockSample,
            stem=StemSample,
            activation=nn.ELU,
            *args,
            **kwargs
        )


class VideoVaeSupervised(VideoVaeUpsample):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.classifier = nn.Sequential(
            nn.Linear(256, 50), nn.ReLU(), nn.Linear(50, self.num_classes - 1)
        )

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self._reparametrize(mu, logvar)
        if self.training:
            pred = self.classifier(z.flatten(1))
        else:
            pred = self.classifier(mu.flatten(1))
        return self.decode(z), pred, mu, logvar

    def loss(self, logits, labels):
        # for now just remove it here
        logits = logits[labels != 5]
        labels = labels[labels != 5]
        if logits.shape[0] == 0:
            return NAN_TENSOR.cuda(device=logits.device)
        return F.cross_entropy(logits, labels)

    def calculate_accuracy(self, pred, target):
        pred = pred[target != 5]
        target = target[target != 5]
        if pred.shape[0] == 0:
            return NAN_TENSOR
        labels_hat = torch.argmax(pred, dim=1)
        acc = labels_hat.eq(target).float().mean()
        return acc


class VideoVaeDetachedSupervised(VideoVaeSupervised):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self._reparametrize(mu, logvar)
        if self.training:
            pred = self.classifier(z.flatten(1).detach())
        else:
            pred = self.classifier(mu.flatten(1).detach())
        return self.decode(z), pred, mu, logvar


class VideoVaeSupervisedBCE(VideoVaeSupervised):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def reconstruction_loss(recon_x, x):
        return F.binary_cross_entropy(recon_x.mul(0.5).add(0.5), x.mul(0.5).add(0.5))

    def loss(self, logits, labels):
        return super().loss(logits, labels) / 10.0


class VideoAE(VideoVaeUpsample):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder.mc3.reduce.add_module(
            "ae_conv3d", Conv3DNoTemporal(16, 8, stride=2)
        )
        self.encoder.mc3.reduce.add_module("ae_bn3d", nn.BatchNorm3d(8))
        self.encoder.mc3.reduce.add_module("ae_elu", nn.ELU())

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.encode(x)

        return (
            self.decode(x),
            torch.ones((x.shape[0], self.num_classes), device=x.device),
            torch.zeros_like(x),
            torch.ones_like(x),
        )

    def loss(self, logits, labels):
        return torch.zeros((1,), device=logits.device)


class PretrainedVAE(VideoVaeUpsample):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        state_dict = torch.load(
            "/mnt/raid5/sebastian/model_checkpoints/ff_vae_video_upsample/model.ckpt"
        )["state_dict"]

        mapped_state_dict = OrderedDict()
        for key, value in state_dict.items():
            mapped_state_dict[key.replace("model.", "")] = value

        self.load_state_dict(mapped_state_dict)
