# flake8: noqa
#%%
from pathlib import Path

from forgery_detection.data.file_lists import FileList
from forgery_detection.data.file_lists import SimpleFileList
from forgery_detection.data.utils import resized_crop


f = FileList.load(
    "/home/sebastian/data/file_lists/avspeech_crop_tests/aligned_faces.json"
)
a = f.get_dataset(
    "test",
    audio_file_list=SimpleFileList.load(
        "/home/sebastian/data/file_lists/avspeech_crop_tests/mfcc_features.json"
    ),
    sequence_length=8,
    image_transforms=resized_crop(112),
)
path = Path("/home/ondyari/avspeech_test_formats/cropped_images_aligned_faces")
for p in sorted(path.iterdir()):
    if p.is_dir():
        try:
            a.audio_file_list.files[p.name]
        except KeyError:
            print(f"{p.name} not in audio")

#%%
from forgery_detection.data.loading import get_fixed_dataloader
from forgery_detection.data.loading import BalancedSampler
from forgery_detection.data.file_lists import FileList
from forgery_detection.data.utils import resized_crop
from tqdm import trange
from importlib import reload
import forgery_detection.data.file_lists as file_lists
from forgery_detection.lightning.logging.const import AudioMode

reload(file_lists)

f = FileList.load(
    "/data/ssd1/file_lists/c40/trf_100_100_full_size_relative_bb_8_sl.json"
)
s = file_lists.SimpleFileList.load(
    "/data/hdd/audio_features/mfcc_features_file_list.json"
)
a = f.get_dataset(
    "train",
    audio_file_list=s,
    audio_mode=AudioMode.SAME_VIDEO_MAX_DISTANCE,
    sequence_length=8,
    image_transforms=resized_crop(112),
)

data_loader = get_fixed_dataloader(
    a, batch_size=12, num_workers=12, sampler=BalancedSampler
)
iter = data_loader.__iter__()
print("len dataloader", len(data_loader))
from tqdm import tqdm

b = next(iter)

for x in trange(len(data_loader)):
    try:
        next(iter)
    except KeyError as e:
        print(e)
    pass
#%%

from forgery_detection.data.set import FileList


f = FileList.load("/data/ssd1/file_lists/c40/tracked_resampled_faces.json")
d = f.get_dataset(
    "train",
    audio_file_list="/data/hdd/audio_features/audio_features_deep_speech.npy",
    sequence_length=8,
    image_transforms=[],
)

#%%
p_idx = 1117399 + 7
nt_a_idx = 870468 + 7
nt_b_idx = nt_a_idx + 6696 + 7

pristine_data = d[p_idx, p_idx][0]
nt_a_data = d[nt_a_idx, nt_a_idx][0]
nt_b_data = d[nt_b_idx, nt_b_idx][0]

print(d._samples[p_idx])
print(d._samples[nt_a_idx])
print(d._samples[nt_b_idx])

import matplotlib.pyplot as plt

plt.imshow(pristine_data[0].permute(1, 2, 0)), plt.show()
plt.imshow(nt_a_data[0].permute(1, 2, 0)), plt.show()
plt.imshow(nt_b_data[0].permute(1, 2, 0)), plt.show()
