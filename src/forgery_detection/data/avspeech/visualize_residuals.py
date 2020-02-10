import cv2
import numpy as np
import torch
from torchvision.utils import make_grid

from forgery_detection.data.set import FileList
from forgery_detection.data.set import FileListDataset
from forgery_detection.models.image.ae import SimpleAEL1Pretrained
from forgery_detection.models.utils import RECON_X


def _get_images_from_class(dataset: FileListDataset, class_id: int, num_images=10):
    targets = np.array(dataset.targets)
    correct_class_idx = np.argwhere(targets == class_id)
    sampled_images_idx = correct_class_idx[
        np.linspace(0, len(correct_class_idx) - 1, num=num_images, dtype=int)
    ]

    images = []
    for idx in sampled_images_idx:
        image, target = dataset[idx[0]]
        assert target == class_id
        images += [image]

    return images


def visualize():
    net = SimpleAEL1Pretrained().eval()

    file_list = FileList.load(
        "/data/ssd1/file_lists/avspeech/resampled_and_avspeech_100_samples_consolidated.json"
    )
    dataset = file_list.get_dataset("train", sequence_length=1)
    for class_id in range(6):
        sample_images = _get_images_from_class(
            dataset, class_id=class_id, num_images=10
        )
        sample_images = torch.stack(sample_images)
        output_images = net(sample_images)[RECON_X]
        diff = sample_images - output_images

        x_12 = sample_images.view(-1, 3, 112, 112)
        x_12_recon = output_images.contiguous().view(-1, 3, 112, 112)
        diff_recon = diff.contiguous().view(-1, 3, 112, 112)
        x_12 = torch.cat(
            (x_12, x_12_recon, diff_recon), dim=2
        )  # this needs to stack the images differently
        datapoints = make_grid(x_12, nrow=10, range=(-1, 1), normalize=True)

        d = datapoints.detach().permute(1, 2, 0).numpy() * 255
        d = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"images_classes_{class_id}.png", d)


if __name__ == "__main__":
    visualize()
