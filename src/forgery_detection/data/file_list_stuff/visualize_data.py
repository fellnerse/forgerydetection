import matplotlib
from cv2 import cv2
from torchvision.utils import make_grid

from forgery_detection.data.face_forensics.splits import TRAIN_NAME
from forgery_detection.data.loading import get_fixed_dataloader
from forgery_detection.data.set import FileList

matplotlib.use("agg")

avspeech_filelist = FileList.load(
    "/data/ssd1/file_lists/avspeech/avspeech_seti_20k_100_samples.json"
)

dataloader = get_fixed_dataloader(
    avspeech_filelist.get_dataset(TRAIN_NAME, sequence_length=8), 4, num_workers=1
)

for batch in dataloader:
    # generate image out of batch
    x, target = batch
    x = x.view(-1, 3, 112, 112)
    im = make_grid(x, normalize=True, range=(-1, 1)).permute(1, 2, 0)
    im = im.numpy() * 255
    # torch converts images to bgr colour space
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # save it for now
    cv2.imwrite("/home/sebastian/loaded_iamge.png", im)
    # plt.imshow(im)
    # plt.show()
    # subprocess.check_output(["imgcat", "/home/sebastian/loaded_image.png"], shell=True)

    input("Pls hit return")
