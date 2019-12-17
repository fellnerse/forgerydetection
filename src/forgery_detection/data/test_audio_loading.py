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

f = FileList.load("/data/ssd1/file_lists/c40/tracked_resampled_faces.json")
a = f.get_dataset(
    "test",
    audio_file="/data/hdd/audio_features/audio_features.npy",
    sequence_length=8,
    transform=resized_crop(112),
)
b = a[0]

data_loader = get_fixed_dataloader(
    a, batch_size=8, num_workers=12, sampler=BalancedSampler
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

f = FileList.load("/data/ssd1/file_lists/c40/tracked_resampled_faces.json")
print(f.root)
