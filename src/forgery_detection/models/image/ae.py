import logging

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import conv1x1
from torchvision.utils import make_grid

from forgery_detection.lightning.utils import NAN_TENSOR
from forgery_detection.lightning.utils import VAL_ACC
from forgery_detection.models.image.utils import ConvBlock
from forgery_detection.models.mixins import FaceNetLossMixin
from forgery_detection.models.mixins import FourierLossMixin
from forgery_detection.models.mixins import L1LossMixin
from forgery_detection.models.mixins import LaplacianLossMixin
from forgery_detection.models.mixins import PretrainedNet
from forgery_detection.models.mixins import SupervisedNet
from forgery_detection.models.mixins import TwoHeadedSupervisedNet
from forgery_detection.models.mixins import VGGLossMixin
from forgery_detection.models.utils import ACC
from forgery_detection.models.utils import CLASS_ACC
from forgery_detection.models.utils import CLASSIFICATION_LOSS
from forgery_detection.models.utils import GeneralAE
from forgery_detection.models.utils import LOSS
from forgery_detection.models.utils import PRED
from forgery_detection.models.utils import RECON_X
from forgery_detection.models.utils import TARGET
from forgery_detection.models.utils import X

logger = logging.getLogger(__file__)


class SimpleAE(GeneralAE):
    def __init__(self, *args, **kwargs):
        super(SimpleAE, self).__init__(
            num_classes=5, sequence_length=1, contains_dropout=False
        )

        # Encoder
        self.block1 = ConvBlock(3, 64, (3, 3), 1, 1)  # 64
        self.block2 = ConvBlock(64, 128, (3, 3), 1, 1)  # 32
        self.block3 = ConvBlock(128, 256, (3, 3), 1, 1)  # 16
        self.block4 = ConvBlock(256, 16, (3, 3), 1, 1)  # 8

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

        x = F.elu(self.block1(x))
        x = F.elu(self.block2(x))
        x = F.elu(self.block3(x))
        x = F.elu(self.block4(x))

        return x

    def decode(self, z):

        z = self.fct_decode(z)
        z = self.final_decod_mean(z)
        z = torch.tanh(z)

        return z

    def forward(self, x):
        x = self.encode(x)

        return {
            RECON_X: self.decode(x),
            PRED: torch.ones((x.shape[0], self.num_classes), device=x.device),
        }

    def reconstruction_loss(self, recon_x, x):
        return {
            "bce_loss": F.binary_cross_entropy_with_logits(recon_x, torch.sigmoid(x))
        }

    def loss(self, logits, labels):
        return torch.zeros((1,), device=logits.device)


class SimpleAEVGG(SimpleAE, VGGLossMixin):
    def reconstruction_loss(self, recon_x, x):
        return {"vgg_content_loss": self.content_loss(recon_x, x)}


class SimpleAEL1(SimpleAE, L1LossMixin):
    def reconstruction_loss(self, recon_x, x):
        return self.l1_loss(recon_x, x)


class SimpleAEL1Pretrained(
    PretrainedNet(
        "/mnt/raid5/sebastian/model_checkpoints/avspeech_ff_100/image/ae/l1/model.ckpt"
    ),
    SimpleAEL1,
):
    pass


class SimpleAEVggPretrained(
    PretrainedNet(
        "/mnt/raid5/sebastian/model_checkpoints/avspeech_ff_100/image/ae/vgg/model_ported.ckpt"
    ),
    SimpleAEVGG,
):
    pass


class StackedAE(GeneralAE, L1LossMixin):
    RECON_X_2 = "recon_x_2"
    RECON_X_DIFF = "recon_x_diff"

    def __init__(self, *args, **kwargs):
        super().__init__(sequence_length=1, contains_dropout=False, *args, **kwargs)

        self.ae1 = SimpleAEL1Pretrained()
        self.ae2 = SimpleAEL1Pretrained()

        self.classifier = nn.Sequential(
            nn.Linear(784, 50), nn.ReLU(), nn.Linear(50, self.num_classes - 1)
        )

    def training_step(self, batch, batch_nb, system):
        x, target = batch
        forward_dict = self.forward(x)

        forward_dict[TARGET] = target
        forward_dict[X] = x

        metric_dict = self._calculate_metrics(**forward_dict)

        if hasattr(system, "_log_training") and system._log_training:
            system._log_training = False
            _x = torch.cat((x, forward_dict[self.RECON_X_DIFF]), dim=1)
            _recon_x = torch.cat(
                (forward_dict[RECON_X], forward_dict[self.RECON_X_2]), dim=1
            )
            self._log_reconstructed_images(system, _x, _recon_x, suffix="train")

        lightning_log = {"loss": metric_dict[LOSS]}
        tensorboard_log = self._metric_dict_to_tensorboard_log(
            metric_dict, suffix="train"
        )
        return tensorboard_log, lightning_log

    def aggregate_outputs(self, output, system):
        """Aggregates outputs from val_step"""

        should_log_images = self._should_log_images()

        # make sure this can see how many data loaders we have
        random_batch, static_batch = output

        # process the random batch
        output_dict = self._transform_output_dict(random_batch)
        with torch.no_grad():
            # calculated metrics based on outputs
            metric_dict = self._calculate_metrics(**output_dict)

            if should_log_images:
                # log reconstructed images
                _x = torch.cat((output_dict[X], output_dict[self.RECON_X_DIFF]), dim=1)
                _recon_x = torch.cat(
                    (output_dict[RECON_X], output_dict[self.RECON_X_2]), dim=1
                )
                self._log_reconstructed_images(
                    system, _x, _recon_x, suffix="val/random_batch"
                )
                # the next training step should log reconstructed images as well
                system._log_training = True

            # confusion matrix
            class_accuracies = system.log_confusion_matrix(
                output_dict[TARGET].cpu(), output_dict[PRED].cpu()
            )

            metric_dict[CLASS_ACC] = class_accuracies

            tensorboard_log = metric_dict
            lightning_log = {VAL_ACC: metric_dict[ACC]}

        # process the static batch
        # for the static batch we log only the images
        if should_log_images:
            output_dict = self._transform_output_dict(static_batch)
            _x = torch.cat((output_dict[X], output_dict[self.RECON_X_DIFF]), dim=1)
            _recon_x = torch.cat(
                (output_dict[RECON_X], output_dict[self.RECON_X_2]), dim=1
            )
            self._log_reconstructed_images(
                system, _x, _recon_x, suffix="val/static_batch"
            )

        return tensorboard_log, lightning_log

    def forward(self, x):
        forward_dict1 = self.ae1.forward(x)
        recon_x_1 = forward_dict1[RECON_X]

        recon_x_diff = torch.tanh(x - recon_x_1)
        h2 = self.ae2.encode(recon_x_diff)

        return {
            RECON_X: recon_x_1,
            self.RECON_X_DIFF: recon_x_diff,
            PRED: self.classifier(h2.flatten(1)),
            self.RECON_X_2: self.ae2.decode(h2),
        }

    def loss(self, logits, labels):
        # for now just remove it here
        logits = logits[labels != 5]
        labels = labels[labels != 5]
        if logits.shape[0] == 0:
            return NAN_TENSOR.cuda(device=logits.device)
        return F.cross_entropy(logits, labels) / 100

    def calculate_accuracy(self, pred, target):
        pred = pred[target != 5]
        target = target[target != 5]
        if pred.shape[0] == 0:
            return NAN_TENSOR
        labels_hat = torch.argmax(pred, dim=1)
        acc = labels_hat.eq(target).float().mean()
        return acc

    def encode(self, x):
        pass

    def decode(self, z):
        pass

    def _transform_output_dict(self, output: dict):
        network_output = [x[PRED] for x in output]
        x_recon_1 = torch.cat([x[RECON_X] for x in network_output], 0)
        pred = torch.cat([x[PRED] for x in network_output], 0)
        x_recon_2 = torch.cat([x[self.RECON_X_2] for x in network_output], 0)
        x_recon_diff = torch.cat([x[self.RECON_X_DIFF] for x in network_output], 0)

        target = torch.cat([x[TARGET] for x in output], 0)
        x = torch.cat([x[X] for x in output], 0)
        return {
            RECON_X: x_recon_1,
            self.RECON_X_2: x_recon_2,
            self.RECON_X_DIFF: x_recon_diff,
            PRED: pred,
            TARGET: target,
            X: x,
        }

    def reconstruction_loss(self, recon_x, x):
        return self.l1_loss(recon_x, x)

    def _calculate_metrics(self, **kwargs):
        reconstruction_loss_1 = self.reconstruction_loss(kwargs[RECON_X], kwargs[X])
        reconstruction_loss_2 = self.reconstruction_loss(
            kwargs[self.RECON_X_2], kwargs[self.RECON_X_DIFF]
        )
        reconstruction_loss = reconstruction_loss_1 + reconstruction_loss_2

        classification_loss = self.loss(kwargs[PRED], kwargs[TARGET])

        if not torch.isnan(classification_loss):
            loss = reconstruction_loss + classification_loss
        else:
            loss = reconstruction_loss

        pred = F.softmax(kwargs[PRED], dim=1)

        with torch.no_grad():
            acc_mean = self.calculate_accuracy(pred, kwargs[TARGET])

        return {
            ACC: acc_mean,
            CLASSIFICATION_LOSS: classification_loss,
            LOSS: loss,
            "reconstruction_loss_1": reconstruction_loss_1,
            "reconstruction_loss_2": reconstruction_loss_2,
        }


class SupervisedAEL1(
    SupervisedNet(input_units=16 * 7 * 7, num_classes=5), SimpleAEL1Pretrained
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        h = self.encode(x)
        return {RECON_X: self.decode(h), PRED: self.classifier(h.flatten(1))}

    def loss(self, logits, labels):
        return super().loss(logits, labels) / 20


class SupervisedAEVgg(
    SupervisedNet(input_units=16 * 7 * 7, num_classes=5), SimpleAEVggPretrained
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        h = self.encode(x)
        return {RECON_X: self.decode(h), PRED: self.classifier(h.flatten(1))}

    def loss(self, logits, labels):
        return super().loss(logits, labels)


class SupervisedTwoHeadedAEVGG(
    TwoHeadedSupervisedNet(input_units=16 * 7 * 7, num_classes=5), SimpleAEVggPretrained
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        h = self.encode(x)
        return {RECON_X: self.decode(h), PRED: self.classifier(h.flatten(1))}

    def loss(self, logits, labels):
        return super().loss(logits, labels)


class AEL1VGG(L1LossMixin, SimpleAEVggPretrained):
    def reconstruction_loss(self, recon_x, x):
        return {
            "l1_loss": self.l1_loss(recon_x, x),
            "vgg_content_loss": self.content_loss(recon_x, x) / 4,
        }


class AEFullVGG(SimpleAEVggPretrained):
    def reconstruction_loss(self, recon_x, x):
        return {"perceptual_loss": self.full_loss(recon_x, x, slices=4)}


class AEFullFaceNet(FaceNetLossMixin, SimpleAEVggPretrained):
    def reconstruction_loss(self, recon_x, x):
        return {
            "style_loss": self.style_loss(recon_x, x) * 20,
            "content_loss": self.content_loss(recon_x, x) * 10,
        }


class StyleNet(SimpleAE, VGGLossMixin):
    def reconstruction_loss(self, recon_x, x):
        return {"style_loss": self.style_loss(recon_x, x)}


class SqrtNet(SimpleAE, VGGLossMixin):
    def reconstruction_loss(self, recon_x, x):
        return {"perceptual_loss": self.full_loss(recon_x, x, slices=2)}


class LaplacianLossNet(LaplacianLossMixin, SimpleAE):
    def reconstruction_loss(self, recon_x, x):
        return {"laplacian_loss": self.laplacian_loss(recon_x, x)}


class PretrainedLaplacianLossNet(
    PretrainedNet(
        "/home/sebastian/log/runs/TRAIN/laplacian_loss/version_2/checkpoints/_ckpt_epoch_4.ckpt"
    ),
    LaplacianLossNet,
):
    pass


class KrakenAE(GeneralAE, L1LossMixin):
    def __init__(self, *args, **kwargs):
        super(KrakenAE, self).__init__(
            num_classes=5, sequence_length=1, contains_dropout=False
        )
        # Encoder
        self.encoder = nn.Sequential(
            ConvBlock(3, 64, (3, 3), 1, 1),
            nn.ELU(),
            ConvBlock(64, 128, (3, 3), 1, 1),
            nn.ELU(),
            ConvBlock(128, 256, (3, 3), 1, 1),
            nn.ELU(),
            ConvBlock(256, 16, (3, 3), 1, 1),
            nn.ELU(),
        )

        self.decoder = [self._get_decoder(idx) for idx in range(self.num_classes + 1)]

    def _get_decoder(self, idx):

        decoder = nn.Sequential(
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
            nn.Conv2d(16, 3, (3, 3), padding=1),
        )
        self.__setattr__(f"decoder_{idx}", decoder)
        return decoder

    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = [decoder(x) for decoder in self.decoder]
        x = torch.stack(x, dim=1)
        return torch.tanh(x)

    def forward(self, x):
        h = self.encode(x)
        decoded_images = self.decode(h)

        return {
            RECON_X: decoded_images,
            # PRED: torch.ones((x.shape[0], self.num_classes), device=x.device),
            PRED: self._calculated_predictions(x, decoded_images)[
                :, : self.num_classes
            ],
        }

    def _calculated_predictions(self, x, decoded_images):
        x = torch.stack([x] * decoded_images.shape[1], dim=1)
        reconstruction_losses = torch.mean(
            self.reconstruction_loss(decoded_images, x, reduction="none")["l1_loss"],
            dim=[-3, -2, -1],
        )
        return reconstruction_losses

    def loss(self, logits, labels):
        # for now just remove it here
        logits = logits[labels != 5]
        labels = labels[labels != 5]
        if logits.shape[0] == 0:
            return NAN_TENSOR.cuda(device=logits.device)
        return F.cross_entropy(logits, labels)

    def reconstruction_loss(self, recon_x, x, **kwargs):
        return {"l1_loss": self.l1_loss(recon_x, x, **kwargs)}

    def _pick_correct_reconstructions(self, **kwargs):
        recon_x, targets = kwargs[RECON_X], kwargs[TARGET]
        recon_x = recon_x[range(recon_x.shape[0]), targets].squeeze(1)
        return recon_x

    def _calculate_batched_reconstruction_loss(self, batch_size, kwargs):
        recon_x = kwargs[RECON_X]
        kwargs[RECON_X] = self._pick_correct_reconstructions(**kwargs)
        reconstruction_loss = super()._calculate_batched_reconstruction_loss(
            batch_size, kwargs
        )
        kwargs[RECON_X] = recon_x
        return reconstruction_loss

    def _log_reconstructed_images(self, system, x, x_recon, suffix="train"):
        x_12 = x[:4].view(-1, 3, 112, 112)
        x_12_recon = (
            x_recon[:4].contiguous().view(-1, 3, 112, 112).reshape(4, -1, 3, 112, 112)
        )
        x_12 = x_12.unsqueeze(1)
        x_12 = torch.cat((x_12, x_12_recon), dim=1).reshape(-1, 3, 112, 112)
        datapoints = make_grid(x_12, nrow=7, range=(-1, 1), normalize=True)
        system.logger.experiment.add_image(
            f"reconstruction/{suffix}",
            datapoints,
            dataformats="CHW",
            global_step=system.global_step,
        )


class FourierAE(FourierLossMixin, SimpleAE):
    def reconstruction_loss(self, recon_x, x):
        return {"complex_loss": self.fourier_loss(recon_x, x)}


class BiggerAE(SimpleAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block1 = BasicBlock(
            3, 64, 2, downsample=self._downsample(3, 64 * 1, 2)
        )  # 64
        self.block2 = BasicBlock(
            64, 128, 2, downsample=self._downsample(64, 128, 2)
        )  # 32
        self.block3 = BasicBlock(
            128, 256, 2, downsample=self._downsample(128, 256, 2)
        )  # 16
        self.block4 = BasicBlock(
            256, 16, 2, downsample=self._downsample(256, 16, 2)
        )  # 8

        self.fct_decode = nn.Sequential(
            self._upblock(16, 64, 3),
            self._upblock(64, 128, 3),
            self._upblock(128, 128, 3),
            self._upblock(128, 16, 3),
        )

    def _downsample(self, in_planes, out_planes, stride):
        return nn.Sequential(
            conv1x1(in_planes, out_planes, stride), nn.BatchNorm2d(out_planes)
        )

    def _upblock(self, in_size, out_size, num_conv, activation=nn.ReLU):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_size, out_size, 3, 1, 1),
            nn.BatchNorm2d(out_size),
            activation(),
            *[
                nn.Sequential(
                    nn.Conv2d(out_size, out_size, 3, 1, 1),
                    nn.BatchNorm2d(out_size),
                    activation(),
                )
                for _ in range(num_conv - 1)
            ],
        )

    def encode(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        return x


class BiggerFourierAE(FourierLossMixin, BiggerAE):
    def reconstruction_loss(self, recon_x, x):
        return {"complex_loss": self.fourier_loss(recon_x, x)}


class BiggerL1AE(L1LossMixin, BiggerAE):
    def reconstruction_loss(self, recon_x, x):
        return {"l1_loss": self.l1_loss(recon_x, x)}


class SupervisedBiggerFourierAE(
    SupervisedNet(16 * 7 * 7, num_classes=5), BiggerFourierAE
):
    avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        h = self.encode(x)
        # avg = self.avgpool(h)
        # return {RECON_X: self.decode(h), PRED: self.classifier(avg.flatten(1))}
        return {RECON_X: self.decode(h), PRED: self.classifier(h.flatten(1))}

    def loss(self, logits, labels):
        return super().loss(logits, labels) * 100


class SupervisedBiggerAEL1(
    SupervisedNet(input_units=16 * 7 * 7, num_classes=5), BiggerAE, L1LossMixin
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reconstruction_loss(self, recon_x, x):
        return {"l1_loss": self.l1_loss(recon_x, x) * 200 * 4}

    def forward(self, x):
        h = self.encode(x)
        return {RECON_X: self.decode(h), PRED: self.classifier(h.flatten(1))}

    def loss(self, logits, labels):
        return super().loss(logits, labels) * 20 * 4
