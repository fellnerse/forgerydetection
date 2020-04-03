# flake8: noqa
#%%
from forgery_detection.data.set import FileList
from forgery_detection.models.audio.multi_class_classification import (
    PretrainedSimilarityNet,
)

p = PretrainedSimilarityNet()
f = FileList.load("/data/ssd1/file_lists/c40/tracked_resampled_faces.json")


#%%
import torch
from forgery_detection.data.loading import get_fixed_dataloader

d = f.get_dataset(
    "val",
    sequence_length=8,
    audio_file="/data/hdd/audio_features/audio_features_deep_speech.npy",
)
loader = get_fixed_dataloader(d, batch_size=10, num_workers=1)

#%%
def get_sample(idx):
    print(d._samples[idx])
    zeros = [d[idx + i, idx] for i in range(8)]
    vid = torch.stack([zeros[i][0][0] for i in range(8)])
    aud = torch.stack([torch.tensor(zeros[i][0][1]) for i in range(8)])
    return (vid.unsqueeze(0), aud.unsqueeze(0))


with torch.no_grad():
    idx = 1189500 * 1 + 100 + 1000 - 309

    # samples = [get_sample(idx + i * 8) for i in range(10)]
    # samples = (
    #     torch.cat([sample[0] for sample in samples]),
    #     torch.cat([sample[1] for sample in samples]),
    # )
    samples, target = next(loader.__iter__())
    out = p(samples)
    printerones = torch.sum((out[0] - out[1]).pow(2), dim=1)

    four = printerones[target == 4]
    unfour = printerones[target != 4]

    print("four", torch.mean(four), torch.std(four), four)
    print("unfour", torch.mean(unfour), torch.std(unfour), unfour)
