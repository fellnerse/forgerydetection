import torch
from torch import nn
from torchvision.models.video import mc3_18

from forgery_detection.models.utils import GeneralVAE


class UpBlock3d(nn.Module):
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
        super(UpBlock3d, self).__init__()
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


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, input):
        return input.transpose(self.dim0, self.dim1)


class VideoVae(GeneralVAE):
    def __init__(self, *args, **kwargs):
        super(GeneralVAE, self).__init__(
            num_classes=5, sequence_length=4, contains_dropout=False
        )

        self.encoder = mc3_18(pretrained=True)
        self.encoder.layer4 = nn.Identity()
        self.encoder.fc = nn.Identity()
        # output: 256 -> mu and sigma are 128

        # input 4 x 2 x 2 x 2
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(16, 32, (1, 3, 3), stride=(3, 4, 4)),
            nn.ReLU(),
            UpBlock3d(32, 64, num_conv=1),
            UpBlock3d(64, 128, num_conv=1),
            UpBlock3d(128, 256, num_conv=2),
            UpBlock3d(256, 256, num_conv=2),
            nn.Conv3d(
                256, 3, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
            ),
            Transpose(2, 1),
        )

    def encode(self, x):
        """return mu_z and logvar_z"""
        x = x.transpose(1, 2)
        x = self.encoder(x)
        return x[:, :128], x[:, 128:]  # output shape - batch_size x 128

    def decode(self, z):
        z = self.decoder(z)
        z = torch.tanh(z)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar).view(-1, 16, 2, 2, 2)
        return (
            self.decode(z),
            torch.ones((x.shape[0], self.num_classes), device=x.device),
            mu,
            logvar,
        )

    def loss(self, logits, labels):
        return torch.zeros((1,), device=logits.device)
