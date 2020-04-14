# flake8: noqa
import logging
from pathlib import Path


logger = logging.getLogger(__file__)


pristine = Path(
    "/data/hdd/c40_resampled/original_sequences/youtube/c40/face_images_tracked/077"
)
pristine_images = sorted(pristine.glob("*.png"))
fakerones_A = Path(
    "/data/hdd/c40_resampled/manipulated_sequences/NeuralTextures/c40/face_images_tracked/077_100"
)
fakerones_A_images = sorted(fakerones_A.glob("*.png"))
fakerones_B = Path(
    "/data/hdd/c40_resampled/manipulated_sequences/NeuralTextures/c40/face_images_tracked/100_077"
)
fakerones_B_images = sorted(fakerones_B.glob("*.png"))


logger.warning(f"pristine images: {len(pristine_images)}")
logger.warning(f"fakerones_A images: {len(fakerones_A_images)}")
logger.warning(f"fakerones_B images: {len(fakerones_B_images)}")


#%%
min_length = min(
    len(list(pristine.glob("*.png"))),
    len(list(fakerones_A.glob("*.png"))),
    len(list(fakerones_B.glob("*.png"))),
)
import matplotlib.pyplot as plt
from torchvision.datasets.folder import default_loader


def show_image(path):
    img_rgb = default_loader(str(path))
    plt.imshow(img_rgb)
    plt.show()


show_image(pristine_images[8])
show_image(fakerones_A_images[8])
show_image(fakerones_B_images[8])

#%%
from torchvision.models.detection import maskrcnn_resnet50_fpn

m = maskrcnn_resnet50_fpn(pretrained=True).eval().cuda(2)

#%%
from torchvision.transforms import ToTensor
from tqdm import tqdm
import torch


def get_lable_scores(path_list):
    label_scores = []
    to_tensor = ToTensor()
    i = 0
    for path in tqdm(path_list):
        img = default_loader(str(path))
        img = to_tensor(img).unsqueeze(0).cuda(2)
        out = m(img)[0]
        labels = out["labels"].detach().cpu()
        scores = out["scores"].detach().cpu()
        label_scores += [(labels, scores)]
        # i += 1
        # if i > 10:
        #     break
    return label_scores


pristine_label_scores = get_lable_scores(pristine_images[:min_length])
fakerones_A_label_scores = get_lable_scores(fakerones_A_images[:min_length])

#%%
pristine_label_length = [len(x[0]) for x in pristine_label_scores]
fakerones_A_label_length = [len(x[0]) for x in fakerones_A_label_scores]

plt.hist(pristine_label_length)
plt.hist(fakerones_A_label_length)
plt.show()

#%%
pristine_all_label = torch.cat([x[0] for x in pristine_label_scores])
fakerones_A_all_labels = torch.cat([x[0] for x in fakerones_A_label_scores])

plt.hist(pristine_all_label)
# plt.show()
plt.hist(fakerones_A_all_labels)
plt.show()
