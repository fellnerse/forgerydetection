# https://github.com/atinghosh/VAE-pytorch/blob/master/VAE_celeba.py
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torchvision.utils import make_grid

from forgery_detection.lightning.utils import VAL_ACC
from forgery_detection.models.utils import LightningModel

no_of_sample = 10
BATCH_SIZE = 32


class Conv_Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        stride,
        pool_kernel_size=(2, 2),
    ):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding, stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding, stride)
        self.pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = self.pool(x)

        return x


class GeneralVAE(LightningModel):
    def loss(self, recon_x, x, mu, logvar) -> Variable:
        # how well do input x and output recon_x agree?
        if self.training:
            BCE = 0
            for recon_x_one in recon_x:
                BCE += self.calculate_accuracy(recon_x_one, x)
            BCE /= len(recon_x)
        else:
            BCE = self.calculate_accuracy(recon_x, x)

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= BATCH_SIZE * 3 * 128 * 128

        return BCE + KLD

    def training_step(self, batch, batch_nb, system):
        x, target = batch
        recon_x, mu, logvar = self.forward(x)

        loss = self.loss(recon_x, x, mu, logvar)
        lightning_log = {"loss": loss}
        tensorboard_log = {"loss": {"train": loss}}
        return tensorboard_log, lightning_log

    def aggregate_outputs(self, outputs, system):
        network_output = [x["pred"] for x in outputs]
        x_recon = torch.cat([x[0] for x in network_output], 0)
        mu = torch.cat([x[1] for x in network_output], 0)
        logvar = torch.cat([x[2] for x in network_output], 0)

        # target = torch.cat([x["target"] for x in outputs], 0)
        x = torch.cat([x["x"] for x in outputs], 0)

        loss_mean = self.loss(x_recon, x, mu, logvar)
        acc_mean = self.calculate_accuracy(x_recon, x)

        tensorboard_log = {"loss": loss_mean, "acc": acc_mean}
        lightning_log = {VAL_ACC: acc_mean}

        # log 10 images
        x_10 = x[:10]
        x_10_recon = x_recon[:10]
        x_10 = torch.cat((x_10, x_10_recon), dim=0)
        datapoints = make_grid(x_10, nrow=10, range=(-1, 1), normalize=False)
        system.logger.experiment.add_image(
            "reconstruction",
            datapoints,
            dataformats="CHW",
            global_step=system.global_step,
        )

        return tensorboard_log, lightning_log

    def calculate_accuracy(self, pred, target):
        return F.binary_cross_entropy(pred, target)


class SimpleVAE(GeneralVAE):
    def __init__(self, *args, **kwargs):
        super(SimpleVAE, self).__init__(
            num_classes=5, sequence_length=1, contains_dropout=False
        )

        # Encoder
        self.block1 = Conv_Block(3, 64, (3, 3), 1, 1)  # 64
        self.block2 = Conv_Block(64, 128, (3, 3), 1, 1)  # 32
        self.block3 = Conv_Block(128, 256, (3, 3), 1, 1)  # 16
        self.block4 = Conv_Block(256, 32, (3, 3), 1, 1)  # 8

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

    def reparameterize(self, mu: Variable, logvar: Variable) -> Variable:

        if self.training:
            # multiply log variance with 0.5, then in-place exponent
            # yielding the standard deviation

            sample_z = []
            for _ in range(no_of_sample):
                std = logvar.mul(0.5).exp_()  # type: Variable
                eps = Variable(std.data.new(std.size()).normal_())
                sample_z.append(eps.mul(std).add_(mu))

            return sample_z

        else:
            return mu

    def decode(self, z):

        z = self.fct_decode(z)
        z = self.final_decod_mean(z)
        z = F.sigmoid(z)

        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        if self.training:
            return [self.decode(z) for z in z], mu, logvar
        else:
            return self.decode(z), mu, logvar
