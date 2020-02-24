import cv2
import dlib
import numpy as np
import torch
from scipy.ndimage import correlate1d
from scipy.ndimage import generic_laplace
from torch.nn import functional as F
from torchvision.datasets.folder import default_loader
from torchvision.transforms import transforms
from torchvision.utils import make_grid

from forgery_detection.models.utils import GeneralAE
from forgery_detection.models.video.ae import PretrainedVideoAE


class AESampler:
    def __init__(self, ae: GeneralAE):
        self.ae = ae
        self.latent_vector = torch.zeros_like(
            self.ae.encode(torch.zeros(1, self.ae.sequence_length, 3, 112, 112))
        )
        self.latent_vector.normal_()

        # self.latent_vector *= 0
        self.n = 0

    def sample(self):

        # self.latent_vector[:, :, self.n].normal_()
        self.latent_vector.normal_()
        # self.n += 1
        # gaussian_sample = self.latent_vector.normal_()
        # for i in range(7):
        #     gaussian_sample[:, i + 1] = gaussian_sample[:, 0]
        sample = self.ae.decode(self.latent_vector)
        # self.latent_vector[:, 0, 0] -= 0.01
        return sample


def get_faces():
    image = cv2.cv2.imread("./../../../me.png")
    image_gray = cv2.cv2.cvtColor(image, cv2.cv2.COLOR_RGB2GRAY)
    detector = dlib.get_frontal_face_detector()
    faces = detector(image_gray, 1)

    image = default_loader("./../../../me.png")

    face_images = list(
        map(
            lambda face: image.crop(
                (
                    face.left() - face.width() * 0.3,
                    face.top() - face.height() * 0.3,
                    face.left() + face.width() * 1.3,
                    face.top() + face.width() * 1.3,
                )
            ),
            faces,
        )
    )
    trans = transforms.Compose(
        [
            transforms.Resize(112),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    face_images = [trans(face_images[0]), trans(face_images[1])]
    face_images = torch.stack(
        (torch.stack([face_images[0]] * 8), torch.stack([face_images[1]] * 8))
    )
    return face_images


def sample():

    ae = PretrainedVideoAE()
    ae_sampler = AESampler(ae)

    sample_images = []
    for _ in range(8):
        sample_images += [ae_sampler.sample().squeeze(0)]

    sample_images = torch.cat(sample_images, dim=0)
    datapoints = make_grid(sample_images, nrow=8, range=(-1, 1), normalize=True)

    d = datapoints.detach().permute(1, 2, 0).numpy() * 255
    d = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)

    cv2.imwrite(f"sampled_images_random.png", d)

    # reconstruct
    face_image = get_faces()
    latent_code = ae.encode(face_image)

    sample_images = []
    for idx in range(11):
        images = ae.decode(
            (
                latent_code[0] * (idx / 10) + latent_code[1] * ((10 - idx) / 10)
            ).unsqueeze(0)
        )
        sample_images += [image for image in images]

    sample_images = torch.cat(sample_images, dim=0)
    datapoints = make_grid(sample_images, nrow=8, range=(-1, 1), normalize=True)

    d = datapoints.detach().permute(1, 2, 0).numpy() * 255
    d = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)

    cv2.imwrite(f"sampled_images_interpolated.png", d)


def do_laplace_stuff():
    image = cv2.imread(f"sampled_images_interpolated.png")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    new_image = cv2.Laplacian(image, cv2.CV_32F, ksize=3)
    # new_image = cv2.convertScaleAbs(new_image)
    cv2.imwrite("laplacian.png", new_image)
    cv2.imwrite("added_laplacian.png", image - new_image)


def my_laplace_stuff():
    image = cv2.imread(f"sampled_images_interpolated.png")
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    tensor = tensor.view(-1, 1, *tensor.shape[-2:])
    weights = torch.tensor([[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]])
    weights = weights.view(1, 1, 3, 3)

    output = F.conv2d(tensor, weights, stride=1, padding=1)
    output = output.view(-1, 3, *output.shape[-2:]).squeeze(0).permute(1, 2, 0)

    output = output.clamp(0, 255)

    output = output.squeeze(0).numpy()

    cv2.imwrite("my_laplace.png", output)


def do_frequency_filter():
    image = cv2.imread(f"sampled_images_interpolated.png")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    # shift the zero-frequncy component to the center of the spectrum
    dft_shift = np.fft.fftshift(dft)
    # save image of the image in the fourier domain.
    magnitude_spectrum = 20 * np.log(
        cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    )
    magnitude_spectrum = 20 * np.log(np.sqrt(dft_shift[:, :, 0] + dft_shift[:, :, 1]))
    cv2.imwrite("magnitude_spectrum.png", magnitude_spectrum)

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 30 : crow + 30, ccol - 30 : ccol + 30] = 1
    mask -= 1
    mask = np.abs(mask)
    # apply mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back = img_back / img_back.max() * 255
    cv2.imwrite("filtered_image_low.png", img_back)


def do_scipy_laplace():
    image = cv2.imread(f"sampled_images_interpolated.png")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def derivative2(input, axis, output, mode, cval):
        return correlate1d(input, [-1, 2, -1], axis, output, mode, cval, 0)

    image2 = generic_laplace(image, derivative2, output=None, mode="reflect", cval=0.0)

    cv2.imwrite("scipy_laplace.png", image2)
    from scipy import ndimage, misc

    ascent = misc.ascent()
    result = ndimage.laplace(ascent)

    cv2.imwrite("ascent.png", ascent)
    cv2.imwrite("result.png", result)


if __name__ == "__main__":
    sample()
