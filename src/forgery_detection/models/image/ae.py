import logging

import torch
import torch.nn.functional as F
from torch import nn

from forgery_detection.lightning.utils import NAN_TENSOR
from forgery_detection.lightning.utils import VAL_ACC
from forgery_detection.models.image.utils import ConvBlock
from forgery_detection.models.mixins import FaceNetLossMixin
from forgery_detection.models.mixins import L1LossMixin
from forgery_detection.models.mixins import PretrainedNet
from forgery_detection.models.mixins import SupervisedNet
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
        """return mu_z and logvar_z"""

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
        return F.binary_cross_entropy_with_logits(recon_x, torch.sigmoid(x))

    def loss(self, logits, labels):
        return torch.zeros((1,), device=logits.device)


class SimpleAEVGG(SimpleAE, VGGLossMixin):
    def reconstruction_loss(self, recon_x, x):
        return self.vgg_content_loss(recon_x, x)


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
