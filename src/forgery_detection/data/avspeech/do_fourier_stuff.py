import matplotlib as mpl
import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt


mpl.rcParams["figure.dpi"] = 300


def get_image(grey=True):
    image = cv2.imread("./../../../me2.png")
    if grey:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_image_fake(grey=True):
    image = cv2.imread("./../../../fakerones.png")
    if grey:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_image_real(grey=True):
    image = cv2.imread("./../../../realones.png")
    if grey:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def do_stuff():
    image = get_image()
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    # shift the zero-frequncy component to the center of the spectrum
    dft_shift = np.fft.fftshift(dft)
    # save image of the image in the fourier domain.
    magnitude_spectrum = 20 * np.log(
        cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    )
    # magnitude_spectrum = 20 * np.log(np.sqrt(dft_shift[:, :, 0] + dft_shift[:, :, 1]))
    cv2.imwrite("magnitude_spectrum.png", magnitude_spectrum)

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 30 : crow + 30, ccol - 30 : ccol + 30] = 1
    # mask -= 1
    # mask = np.abs(mask)
    # apply mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back = img_back / img_back.max() * 255
    cv2.imwrite("filtered_image_low.png", img_back)


def do_cv_sample_stuff():
    grey = True
    img = get_image(grey)
    f = np.fft.fftn(img, axes=(-2, -1))
    fshift = np.fft.fftshift(f, axes=[-2, -1])
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    plt.subplot(111), plt.imshow(magnitude_spectrum, cmap="gray")
    plt.title("magnitude spectrum"), plt.xticks([]), plt.yticks([])
    plt.show()

    if grey:
        rows, cols = img.shape
    else:
        rows, cols, channels = img.shape
    crow, ccol = rows // 2, cols // 2
    square_length = 10

    fshift_high = fshift.copy()
    fshift_high[
        crow - square_length : crow + square_length,
        ccol - square_length : ccol + square_length,
    ] = 0
    img_back_high = reconstruct_from_fftshift(fshift_high)

    fshift_both = fshift.copy()
    fshift_both *= 0
    fshift_both[
        crow - square_length : crow + square_length,
        ccol - square_length : ccol + square_length,
    ] = fshift[
        crow - square_length : crow + square_length,
        ccol - square_length : ccol + square_length,
    ]
    img_back_both = reconstruct_from_fftshift(fshift_both)

    img_back_full = reconstruct_from_fftshift(fshift)

    plt.subplot(221), plt.imshow(img, cmap="gray")
    plt.title("Input Image"), plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(img_back_high, cmap="gray")
    plt.title("Image after HPF"), plt.xticks([]), plt.yticks([])
    plt.subplot(223), plt.imshow(img_back_both, cmap="gray")
    plt.title("Result after LPF"), plt.xticks([]), plt.yticks([])
    plt.subplot(224), plt.imshow(
        (img_back_both + img_back_high) * 1 + img_back_full * 0, cmap="gray"
    )
    plt.title("LPF and HPF"), plt.xticks([]), plt.yticks([])

    plt.show()


def process_channel(img: np.ndarray, sigma=5):
    if len(img.shape) != 2:
        raise ValueError("Image channel should be 2d.")

    rows, cols = img.shape
    mask = generate_gaussian(rows, cols, sigma=sigma)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    fshift_low = fshift * mask
    img_back_low = reconstruct_from_fftshift(fshift_low)

    # invert binary mask
    mask = 1 - mask
    fshift_high = fshift * mask
    img_back_high = reconstruct_from_fftshift(fshift_high)

    return img_back_high, img_back_low


def process_channel_with_mask(img: np.ndarray, mask: np.ndarray):
    if img.shape != mask.shape:
        raise ValueError(
            f"img shape ({img.shape}) and mask shape ({mask.shape}) should be the same!"
        )
    f = np.fft.fft2(img)

    mask = np.fft.ifftshift(mask)
    fshift = f * mask
    img_back = reconstruct_from_fftshift(fshift)
    return img_back


def reconstruct_from_fftshift(fshift_high):
    img_back_high = np.fft.ifft2(fshift_high)
    img_back_high = np.abs(img_back_high).astype(int)
    return img_back_high


def do_cv_sample_stuff_rgb(sigma=20):
    img = get_image(grey=False)
    img_out_low = np.zeros(img.shape, dtype=img.dtype)
    img_out_high = np.zeros(img.shape, dtype=img.dtype)
    for c in range(img.shape[-1]):
        channel = img[..., c]
        img_back_high, img_back_low = process_channel(channel, sigma=sigma)

        # img_out_high[:, :, c] = np.clip(img_back_high, 0, 255)
        img_out_high[:, :, c] = (img_back_high / img_back_high.max()) * 255 // 1
        img_out_low[:, :, c] = np.clip(img_back_low, 0, 255)

    gray_img = get_image(grey=True)
    img_out_gray_high, img_out_gray_low = process_channel(gray_img, sigma=sigma)

    visualize_reconstruced_images(
        img, img_out_gray_high, img_out_gray_low, img_out_high, img_out_low
    )


def visualize_reconstruced_images(
    img, img_out_gray_high, img_out_gray_low, img_out_high, img_out_low
):
    plt.subplot(321), plt.imshow(img)
    plt.title("Input Image"), plt.xticks([]), plt.yticks([])
    plt.subplot(322), plt.imshow(img_out_high)
    plt.title("Image after HPF"), plt.xticks([]), plt.yticks([])
    plt.subplot(323), plt.imshow(img_out_low)
    plt.title("Image after LPF"), plt.xticks([]), plt.yticks([])
    plt.subplot(324), plt.imshow(img_out_gray_high, cmap="gray")
    plt.title("Gray after HPF"), plt.xticks([]), plt.yticks([])
    plt.subplot(325), plt.imshow(img_out_gray_low, cmap="gray")
    plt.title("Gray after LPF"), plt.xticks([]), plt.yticks([])
    plt.show()


def generate_gaussian(rows, cols, sigma=1.0):
    x, y = np.meshgrid(
        np.linspace(-cols // 2, cols // 2, cols),
        np.linspace(-rows // 2, rows // 2, rows),
    )
    d = np.sqrt(x * x + y * y)
    g = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
    return g


def generate_circles(rows, cols, bins=10, dist_exponent=2):
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
        circles.append((inner_dist < d) & (d <= outer_dist))
    return circles


def visualize_circles():
    # circles = generate_circles(1600, 900, bins=4, dist_exponent=3)
    circles = generate_circles(112, 112, bins=4, dist_exponent=3)

    # img = get_image(grey=False)
    img = get_image_fake(grey=False)

    reconstructed_images = []
    for mask in circles:
        reconstructed_images.append(reconstruct_rgb_image_with_mask(img, mask))

    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = 6
    for i in range(len(circles)):
        circle = circles[i]
        fig.add_subplot(rows, columns, (i + 1) * 2 - 1)
        plt.imshow(circle, cmap="gray")
        fig.add_subplot(rows, columns, (i + 1) * 2)
        plt.imshow(reconstructed_images[i])

    # add all mask together
    fig.add_subplot(rows, columns, (len(circles) + 1) * 2 - 1)
    mask = sum(circles)
    plt.imshow(mask)
    fig.add_subplot(rows, columns, (len(circles) + 1) * 2)
    reconstruction = reconstruct_rgb_image_with_mask(img, mask)
    plt.imshow(reconstruction, cmap="gray")

    # visualize difference between input and reconstructed image
    fig.add_subplot(rows, columns, (len(circles) + 2) * 2 - 1)
    residual = img - reconstruction
    plt.imshow(residual * 255, cmap="gray")
    plt.show()

    if np.sum(np.abs(residual)) != 0:
        raise ValueError(
            f"reconstruted image != input image: {np.sum(np.abs(residual))}"
        )


def reconstruct_rgb_image_with_mask(img: np.ndarray, mask: np.ndarray):
    img_out = np.zeros(img.shape, dtype=img.dtype)
    for c in range(img.shape[-1]):
        channel = img[..., c]
        img_back = process_channel_with_mask(channel, mask=mask)

        # img_out[:, :, c] = img_back
        # img_out[:, :, c] = np.clip(img_back, 0, 255)
        img_out[:, :, c] = (img_back / img_back.max()) * 255 // 1

    return img_out


if __name__ == "__main__":
    # do_stuff()
    # do_cv_sample_stuff()
    # do_cv_sample_stuff_rgb(sigma=10)
    visualize_circles()
