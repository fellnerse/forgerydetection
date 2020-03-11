import cv2
import numpy as np
import torch
from torch.utils.data import RandomSampler
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.utils import make_grid

from forgery_detection.data.loading import get_fixed_dataloader
from forgery_detection.data.set import FileList
from forgery_detection.data.utils import irfft
from forgery_detection.data.utils import resized_crop
from forgery_detection.data.utils import rfft
from forgery_detection.models.image.ae import PretrainedBiggerFourierAE
from forgery_detection.models.image.ae import PretrainedBiggerL1AE


def get_faces_():
    f = FileList.load(
        "/data/ssd1/file_lists/avspeech/resampled_and_avspeech_100_samples_consolidated.json"
    )
    dataset = f.get_dataset("val", image_transforms=resized_crop(112))

    # dataset.samples_idx = dataset.samples_idx[:: len(dataset) // 3]
    static_batch_loader = get_fixed_dataloader(
        dataset,
        24,
        sampler=RandomSampler,
        num_workers=1,
        worker_init_fn=lambda worker_id: np.random.seed(worker_id),
    )
    return next(static_batch_loader.__iter__())[0]


def get_val_face():
    path = "/data/ssd2/set/tracked_resampled_faces_112/manipulated_sequences/Deepfakes/c40/face_images_tracked/004_982/0000.png"
    im = default_loader(path)
    trans = transforms.Compose(
        resized_crop(112)
        + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return trans(im).unsqueeze(0)


def visualize(ae, x):
    x = x.cuda(1)
    ae = ae.cuda(1)
    ae.circles = ae.circles.cuda(1)
    recon_x = ae.forward(x)["recon_x"].contiguous()
    print(recon_x.shape)
    recon_x_ = rfft(recon_x)
    ae.circles[-1] = 1
    recon_x_frequencies = torch.cat([irfft(recon_x_ * mask) for mask in ae.circles])

    x_ = rfft(x)
    x_frequencies = torch.cat([irfft(x_ * mask) for mask in ae.circles])

    x_ = torch.cat((x_frequencies, recon_x_frequencies), dim=2)
    return x_


if __name__ == "__main__":
    pass
    # x = get_val_face()
    # x = get_faces(sequence_length=1)
    x = get_faces_()
    ae = PretrainedBiggerFourierAE().eval()
    fourirer_images = visualize(ae, x)
    ae = PretrainedBiggerL1AE().eval()
    l1_images = visualize(ae, x)

    datapoints = torch.cat((fourirer_images, l1_images), dim=3)
    datapoints = make_grid(datapoints, nrow=x.shape[0], range=(-1, 1), normalize=True)
    d = datapoints.detach().permute(1, 2, 0).cpu().numpy() * 255
    d = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)

    cv2.imwrite(f"frequency_analysis_id_random_val.png", d)
