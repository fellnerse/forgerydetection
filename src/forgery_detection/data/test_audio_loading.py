# flake8: noqa
#%%
from pathlib import Path

from forgery_detection.data.loading import get_fixed_dataloader
from forgery_detection.data.set import FileList
from forgery_detection.data.utils import resized_crop

f = FileList.load("/data/ssd1/file_lists/c40/tracked_resampled_faces.json")
a = f.get_dataset(
    "test",
    audio_file="/data/hdd/audio_features/audio_features.npy",
    sequence_length=8,
    transform=resized_crop(112),
)
path = Path(
    "/data/ssd1/set/tracked_resampled_faces/original_sequences/youtube/c40/face_images_tracked"
)
for p in sorted(path.iterdir()):
    if p.is_dir():
        try:
            a.extended_default_loader.audio[p.name]
        except KeyError:
            print(f"{p.name} not in audio")

#%%
from forgery_detection.data.loading import get_fixed_dataloader
from forgery_detection.data.loading import BalancedSampler
from forgery_detection.data.set import FileList
from forgery_detection.data.utils import resized_crop

f = FileList.load("/data/ssd1/file_lists/c40/tracked_resampled_faces_.json")
a = f.get_dataset(
    "val",
    audio_file="/data/hdd/audio_features/audio_features.npy",
    sequence_length=8,
    transform=[],
)
b = a[0]

data_loader = get_fixed_dataloader(
    a, batch_size=12, num_workers=12, sampler=BalancedSampler
)
iter = data_loader.__iter__()
print("len dataloader", len(data_loader))
from tqdm import tqdm

b = next(iter)
for x in tqdm(iter):
    pass
#%%

from forgery_detection.data.loading import get_fixed_dataloader
from forgery_detection.data.loading import BalancedSampler
from forgery_detection.data.set import FileList
from forgery_detection.data.utils import resized_crop

f = FileList.load("/data/ssd1/file_lists/c40/tracked_resampled_faces_.json")
print(f.root)

#%%
from torchsummary import summary
from torchvision.models.resnet import _resnet, BasicBlock
from torch import nn

resnet10 = _resnet(
    "resnet18",
    BasicBlock,
    [1, 1, 1, 1],
    pretrained=False,
    progress=True,
    num_classes=256,
)
resnet10.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
resnet10.bn1 = nn.BatchNorm2d(32)
resnet10.layer4 = nn.Identity()
resnet10.fc = nn.Linear(256, 256)
resnet10.cuda()
summary(resnet10, (1, 8, 40))

#%%
import numpy as np

a = np.load("/data/hdd/audio_features/audio_features.npy", allow_pickle=True)[()]
b = list(a.values())
c = np.concatenate(b, axis=0)
print(c.mean(), c.std(), c.min(), c.max())
# c -= c.mean(axis=0)
# c /= c.std(axis=0)
print(c.mean(), c.std(), c.min(), c.max())
