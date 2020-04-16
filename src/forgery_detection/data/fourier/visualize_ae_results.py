import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import RandomSampler
from torchvision.utils import make_grid

from forgery_detection.data.file_lists import FileList
from forgery_detection.data.loading import get_fixed_dataloader
from forgery_detection.data.utils import irfft
from forgery_detection.data.utils import resized_crop
from forgery_detection.data.utils import rfft
from forgery_detection.models.image.ae import BiggerFourierAE
from forgery_detection.models.image.ae import PretrainedBiggerFourierAE
from forgery_detection.models.image.ae import PretrainedBiggerFourierCorrectLoss
from forgery_detection.models.image.ae import (
    PretrainedBiggerFourierLossSummedOverLast4Dims,
)
from forgery_detection.models.image.ae import PretrainedBiggerL1AE
from forgery_detection.models.image.ae import PretrainedBiggerWindowedAECorrectLoss
from forgery_detection.models.image.ae import (
    PretrainedBiggerWindowedAECorrectLossStrided,
)
from forgery_detection.models.image.ae import (
    PretrainedBiggerWindowedAELossSummedOverLast4Dims,
)
from forgery_detection.models.image.ae import WeightedBiggerFourierAE
from forgery_detection.models.image.frequency_ae import BiggerFrequencyAE
from forgery_detection.models.image.frequency_ae import FrequencyAE
from forgery_detection.models.mixins import PretrainedNet


def get_static_batch_faces():
    f = FileList.load(
        "/data/ssd1/file_lists/avspeech/resampled_and_avspeech_100_samples_consolidated.json"
    )
    dataset = f.get_dataset("val", image_transforms=resized_crop(112))

    dataset.samples_idx = dataset.samples_idx[:: len(dataset) // 3]
    static_batch_loader = get_fixed_dataloader(
        dataset,
        4,
        sampler=RandomSampler,
        num_workers=1,
        worker_init_fn=lambda worker_id: np.random.seed(worker_id),
    )
    return next(static_batch_loader.__iter__())[0]


def visualize_frequencies(ae, x):
    x = x.cuda(1)
    ae = ae.cuda(1)
    ae.circles = ae.circles.cuda(1)
    recon_x = ae.forward(x)["recon_x"].contiguous()
    recon_x_ = rfft(recon_x)
    ae.circles[-1] = 1
    recon_x_frequencies = torch.cat([irfft(recon_x_ * mask) for mask in ae.circles])

    x_ = rfft(x)
    x_frequencies = torch.cat([irfft(x_ * mask) for mask in ae.circles])

    x_ = torch.cat((x_frequencies, recon_x_frequencies), dim=2)
    return x_


def compare_models():
    x = get_static_batch_faces()
    ae = PretrainedBiggerFourierAE().eval()
    fourirer_images = visualize_frequencies(ae, x)
    ae = PretrainedBiggerL1AE().eval()
    l1_images = visualize_frequencies(ae, x)
    datapoints = torch.cat((fourirer_images, l1_images), dim=3)
    datapoints = make_grid(datapoints, nrow=x.shape[0], range=(-1, 1), normalize=True)
    d = datapoints.detach().permute(1, 2, 0).cpu().numpy() * 255
    d = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"frequency_analysis_id_random_val.png", d)


class PretrainedWeightedBiggerFourierAE(
    PretrainedNet(
        "/home/sebastian/log/runs/TRAIN/weighted_bigger_fourier_ae/without_pretraining_and_no_weight_rescaling/checkpoints/_ckpt_epoch_5.ckpt"
    ),
    WeightedBiggerFourierAE,
):
    pass


class PretrainedRelativeBiggerFourierAE(
    PretrainedNet(
        "/home/sebastian/log/runs/TRAIN/fourier_ae_relative_loss/version_1/checkpoints/_ckpt_epoch_5.ckpt"
    ),
    BiggerFourierAE,
):
    pass


class PretrainedFrequencyNet(
    PretrainedNet(
        "/home/sebastian/log/runs/TRAIN/frequency_ae/version_0/checkpoints/_ckpt_epoch_5.ckpt"
    ),
    FrequencyAE,
):
    def forward(self, x):
        d = super().forward(rfft(x))
        d["recon_x"] = irfft(d["recon_x"])
        return d


class PretrainedFrequencyAEraw(
    PretrainedNet(
        "/home/sebastian/log/runs/TRAIN/frequency_ae/version_2/checkpoints/_ckpt_epoch_5.ckpt"
    ),
    BiggerFrequencyAE,
):
    def forward(self, x):
        d = super().forward(rfft(x))
        d["recon_x"] = irfft(d["recon_x"])
        return d


if __name__ == "__main__":
    print("niceroo")

    x = get_static_batch_faces()
    x = x[[3, 2, 0, 1]]

    models_old = {
        "l1": PretrainedBiggerL1AE,
        "fourier": PretrainedBiggerFourierAE,
        "fourier_loss_weighted": PretrainedWeightedBiggerFourierAE,
        "fourier_loss_relative": PretrainedRelativeBiggerFourierAE,
        "frequency_space_tanh": PretrainedFrequencyNet,
        "frequency_space_raw": PretrainedFrequencyAEraw,
    }

    models = {
        "windowed_ae_loss_summed_over_last_4_dims": PretrainedBiggerWindowedAELossSummedOverLast4Dims,
        "windowed_ae_correct_loss": PretrainedBiggerWindowedAECorrectLoss,
        "windowed_ae_correct_loss_strided": PretrainedBiggerWindowedAECorrectLossStrided,
        "fourier_ae_loss_summed_over_last_4_dims": PretrainedBiggerFourierLossSummedOverLast4Dims,
        "fourier_ae_correct_loss": PretrainedBiggerFourierCorrectLoss,
    }

    for name, model in models.items():
        model = model().eval()
        recon_x = model.forward(x)["recon_x"]
        imgs = (
            make_grid(recon_x, nrow=1, normalize=True, range=(-1, 1))
            .permute(1, 2, 0)
            .detach()
            .numpy()
        )
        plt.imshow(imgs)
        plt.title(name)
        plt.show()
        d = cv2.cvtColor(imgs * 255, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"model_comparison_images/{name}.png", d)
