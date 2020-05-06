import json
import logging
import pickle
from collections import OrderedDict
from datetime import datetime
from typing import List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import make_grid

from forgery_detection.data.utils import irfft
from forgery_detection.data.utils import rfft
from forgery_detection.data.utils import windowed_rfft
from forgery_detection.lightning.logging.confusion_matrix import confusion_matrix
from forgery_detection.lightning.logging.const import NAN_TENSOR
from forgery_detection.lightning.logging.const import VAL_ACC
from forgery_detection.lightning.logging.utils import get_logger_dir
from forgery_detection.models.sliced_nets import FaceNet
from forgery_detection.models.sliced_nets import SlicedNet
from forgery_detection.models.sliced_nets import Vgg16

logger = logging.getLogger(__file__)


class PerceptualLossMixin(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net: SlicedNet

    def content_loss(self, recon_x, x, slices=2):
        features_recon_x, features_x = self._calculate_features(
            recon_x, x, slices=slices
        )
        return F.mse_loss(features_recon_x, features_x)

    def style_loss(self, recon_x, x, slices=2):
        features_recon_x, features_x = self._calculate_features(
            recon_x, x, slices=slices
        )

        gram_style_recon_x = self._gram_matrix(features_recon_x)
        gram_style_x = self._gram_matrix(features_x)

        return F.mse_loss(gram_style_x, gram_style_recon_x) * features_recon_x.shape[0]

    def full_loss(self, recon_x, x, slices=2):
        features_recon_x, features_x = self._calculate_features(
            recon_x, x, slices=slices
        )

        gram_style_recon_x = self._gram_matrix(features_recon_x)
        gram_style_x = self._gram_matrix(features_x)

        return F.mse_loss(gram_style_x, gram_style_recon_x) * features_recon_x.shape[
            0
        ] + F.mse_loss(features_recon_x, features_x)

    def _calculate_features(self, recon_x, x, slices=2):
        features_recon_x = self.net(
            recon_x.view(-1, *recon_x.shape[-3:]), slices=slices
        )
        features_x = self.net(x.view(-1, *x.shape[-3:]), slices=slices)
        return features_recon_x, features_x

    @staticmethod
    def _gram_matrix(y):
        features = y.view(*y.shape[: -(len(y.shape) - 2)], y.shape[-2] * y.shape[-1])
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (y.shape[-3] * y.shape[-2] * y.shape[-1])
        return gram


class VGGLossMixin(PerceptualLossMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = Vgg16(requires_grad=False)
        self.net.eval()
        self._set_requires_grad_for_module(self.net, requires_grad=False)


class FaceNetLossMixin(PerceptualLossMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = FaceNet(requires_grad=False)
        self.net.eval()
        self._set_requires_grad_for_module(self.net, requires_grad=False)


class L1LossMixin(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def l1_loss(self, recon_x, x, **kwargs):
        return F.l1_loss(recon_x, x, **kwargs)


class LaplacianLossMixin(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weights = (
            torch.tensor([[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]])
            .view(1, 1, 3, 3)
            .repeat(1, 1, 1, 1)
        )
        self.weights.requires_grad_(False)

    def laplacian_loss(self, recon_x, x):
        if recon_x.device != self.weights.device:
            self.weights = self.weights.to(recon_x.device)

        recon_x_laplacian = F.conv2d(
            recon_x.view(-1, 1, *recon_x.shape[-2:]), self.weights, stride=1, padding=1
        ).view(-1, 3, *recon_x.shape[-2:])
        x_laplacian = F.conv2d(
            x.view(-1, 1, *x.shape[-2:]), self.weights, stride=1, padding=1
        ).view(-1, 3, *x.shape[-2:])

        return F.l1_loss(recon_x_laplacian, x_laplacian)


class FourierLoggingMixin:
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.circles = self.generate_circles(112, 112, bins=4, dist_exponent=3)

    def _log_reconstructed_images(self, system, x, x_recon, suffix="train"):
        x = rfft(x[:4])
        x_recon = rfft(x_recon[:4].contiguous())

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

    def generate_circles(self, rows, cols, bins=10, dist_exponent=2):
        x, y = np.meshgrid(
            np.linspace(-cols // 2, cols // 2, cols),
            np.linspace(-rows // 2, rows // 2, rows),
        )
        d = np.sqrt(x * x + y * y)
        circles = []
        biggest_dist = d.max()
        for i in range(bins):
            inner_dist = i ** dist_exponent * biggest_dist / bins ** dist_exponent
            outer_dist = (i + 1) ** dist_exponent * biggest_dist / bins ** dist_exponent
            circles.append(
                torch.from_numpy(((inner_dist < d) & (d <= outer_dist))).float()
            )

        # add one mask with only 1 for full reconstruction
        circles.append(circles[-1] * 0 + 1)

        circles = torch.stack(circles)
        circles = torch.roll(
            circles,
            [-1 * (dim // 2) for dim in circles.shape[1:]],
            tuple(range(1, len(circles.shape))),
        )
        circles = circles.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # add b x 2 x c
        return circles


class FourierLossMixin(FourierLoggingMixin, nn.Module):
    def fourier_loss(self, recon_x, x):
        complex_recon_x, complex_x = rfft(recon_x), rfft(x)
        # loss = torch.mean(
        #     torch.sum(
        #         torch.sqrt(
        #             torch.sum(
        #                 torch.pow(complex_recon_x - complex_x, 2),
        #                 dim=(1,),  # dim=(-4, -3, -2 ,- 1)
        #             )
        #         ),
        #         dim=(1, 2, 3),
        #     )
        # )  # todo this dimension is pretty sure completely wrong
        # old loss:
        # torch.sqrt(torch.sum((complex_recon_x - complex_x) ** 2, dim=-1))
        loss = torch.mean(torch.norm(complex_recon_x - complex_x, p=2, dim=1))
        return loss


def WeightedFourierLoss(weights: List[float]):
    class WeightedFourierLossMixin(FourierLossMixin):
        __weights = np.array(weights, dtype=float)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # self.__weights /= np.sum(self.__weights)
            # last circle entry is full mask, we can ignore it
            if len(self.__weights) != self.circles.shape[0] - 1:
                raise ValueError(
                    f"Number of weights need to be the same as number of circles, "
                    f"but only {len(self.__weights)}/{self.circles.shape[0]} given."
                )
            self.weight = torch.zeros_like(self.circles[0])
            for idx, weight in enumerate(self.__weights):
                self.weight += self.circles[idx] * weight
            self.weight = self.weight
            self.weight.requires_grad = False

        def fourier_loss(self, recon_x, x):
            if recon_x.device != self.weight.device:
                self.weight = self.weight.to(recon_x.device)

            complex_recon_x, complex_x = rfft(recon_x), rfft(x)
            loss = torch.sqrt(
                torch.sum(self.weight * (complex_recon_x - complex_x) ** 2, dim=-1)
            )
            # loss = loss * self.weight
            return loss.mean()

    return WeightedFourierLossMixin


class WindowedFourierLossMixin(FourierLoggingMixin):
    def windowed_fourier_loss(self, recon_x: torch.tensor, x: torch.tensor):
        complex_recon_x, complex_x = windowed_rfft(recon_x), windowed_rfft(x)
        loss = torch.mean(torch.norm(complex_recon_x - complex_x, p=2, dim=-4))
        return loss


def PretrainedNet(path_to_model: str):
    class PretrainedNetMixin(nn.Module):
        __path_to_model = path_to_model

        def __init__(self, *args, **kwargs):
            super().__init__()
            try:
                state_dict = torch.load(self.__path_to_model)["state_dict"]
            except FileNotFoundError:
                logger.error(
                    f"Could not find the desired model checkpoint: {self.__path_to_model}."
                )
                input("Press any button to continue.")
                return

            mapped_state_dict = OrderedDict()
            for key, value in state_dict.items():
                if not key.startswith("net."):
                    mapped_state_dict[key.replace("model.", "")] = value

            self.load_state_dict(mapped_state_dict)

    return PretrainedNetMixin


def SupervisedNet(input_units: int, num_classes: int):
    class SupervisedNetMixin(nn.Module):
        __input_units = input_units
        __num_classes = num_classes

        def __init__(self, *args, **kwargs):
            super().__init__()
            self.classifier = nn.Sequential(
                nn.Linear(self.__input_units, 50),
                nn.ReLU(),
                nn.Linear(50, self.__num_classes),
            )

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

    return SupervisedNetMixin


def TwoHeadedSupervisedNet(input_units: int, num_classes: int):
    class TwoheadedSupervisedNetMixin(nn.Module):
        __input_units = input_units
        __num_classes = num_classes

        def __init__(self, *args, **kwargs):
            super().__init__()

            class TwoWayForward(nn.Module):
                def __init__(self, way_0: nn.Module, way_1: nn.Module):
                    super().__init__()
                    self.out_0 = way_0
                    self.out_1 = way_1

                def forward(self, x):
                    return torch.cat((self.out_0(x), self.out_1(x)), dim=1)

            self.classifier = nn.Sequential(
                nn.Linear(self.__input_units, 50),
                nn.ReLU(),
                TwoWayForward(nn.Linear(50, self.__num_classes), nn.Linear(50, 2)),
            )

        def loss(self, logits, labels):
            # for now just remove it here
            method_predictions = logits[:, :5][labels != 5]
            binary_predictions = logits[:, 5:][labels != 5]
            labels = labels[labels != 5]

            if labels.shape[0] == 0:
                return NAN_TENSOR.cuda(device=labels.device)

            binary_labels = (labels == 4).long()

            return F.cross_entropy(method_predictions, labels) * (
                4 / 5
            ) + F.cross_entropy(binary_predictions, binary_labels) * (1 / 5)

        def calculate_accuracy(self, pred, target):
            pred = pred[:, :5]
            pred = pred[target != 5]
            target = target[target != 5]
            if pred.shape[0] == 0:
                return NAN_TENSOR
            labels_hat = torch.argmax(pred, dim=1)
            acc = labels_hat.eq(target).float().mean()
            return acc

    return TwoheadedSupervisedNetMixin


class BinaryEvaluationMixin:
    def aggregate_test_output(self, outputs, system):
        # if there are more then one dataloader we ignore the additional data
        if len(system.val_dataloader()) > 1:
            outputs = outputs[0]

        with torch.no_grad():
            if isinstance(outputs[0]["pred"], tuple):
                pred = torch.cat([x["pred"][0] for x in outputs], 0)
                pred_shape = outputs[0]["pred"][0].shape
            else:
                pred = torch.cat([x["pred"] for x in outputs], 0)
                pred_shape = outputs[0]["pred"].shape

            if outputs[0]["target"].shape[0] != pred_shape[0]:
                label = torch.cat([x["target"][0] for x in outputs], 0)
            else:
                label = torch.cat([x["target"] for x in outputs], 0)
            target = label // 4

            loss_mean = self.loss(pred, target)
            pred = pred.cpu()
            target = target.cpu()
            pred = F.softmax(pred, dim=1)
            acc_mean = self.calculate_accuracy(pred, target)

            # confusion matrix
            cm = confusion_matrix(label, torch.argmax(pred, dim=1), num_classes=5)
            cm = cm[:, :2]  # this is only binary classification

            cm /= torch.sum(cm, dim=1, keepdim=True)
            class_accuracies = system.log_confusion_matrix(label, pred)
            class_accuracies[list(class_accuracies.keys())[0]] = cm[0, 0]
            class_accuracies[list(class_accuracies.keys())[1]] = cm[1, 0]
            class_accuracies[list(class_accuracies.keys())[2]] = cm[2, 0]
            class_accuracies[list(class_accuracies.keys())[3]] = cm[3, 0]
            class_accuracies[list(class_accuracies.keys())[4]] = cm[4, 1]

            tensorboard_log = {
                "loss": loss_mean,
                "acc": acc_mean,
                "class_acc": class_accuracies,
            }

            lightning_log = {VAL_ACC: acc_mean}

        return tensorboard_log, lightning_log

    def test_epoch_end(self, outputs, system):
        with torch.no_grad():
            with open(
                get_logger_dir(system.logger)
                / f"outputs_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pkl",
                "wb",
            ) as f:
                pickle.dump(outputs, f)

            tensorboard_log, lightning_log = self.aggregate_test_output(outputs, system)
            logger.info(f"Test accuracy is: {tensorboard_log}")
            print(
                f"{''.join('{:.2%};'.format(float(x.cpu())) for x in tensorboard_log['class_acc'].values())}"
            )
            with open(
                get_logger_dir(system.logger)
                / f"log_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.json",
                "w",
            ) as f:
                json.dump(str(tensorboard_log), f)
            return system._construct_lightning_log(
                tensorboard_log, lightning_log, suffix="test"
            )
