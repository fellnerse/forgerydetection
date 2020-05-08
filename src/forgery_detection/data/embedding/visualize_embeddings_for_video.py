# flake8: noqa
#%%
from pathlib import Path

from forgery_detection.data.file_lists import FileList
from forgery_detection.data.file_lists import SimpleFileList

# trained with /mnt/ssd1/sebastian/file_lists/c40/tracked_resampled_faces_yt_only_112_8_sequence_length.json
# and /data/hdd/audio_features/mfcc_features.npy -> /data/hdd/audio_features/mfcc_features_file_list.json

audio_features = SimpleFileList.load(
    "/mnt/raid5/sebastian/audio_features/mfcc_features_file_list.json"
)
f = FileList.load(
    "/mnt/ssd1/sebastian/file_lists/c40/tracked_resampled_faces_yt_only_112_8_sequence_length.json"
)
#%%

vids = sorted(set(map(lambda x: x[0].split("/")[-2], f.samples["train"])))[
    :10
]  # take first 10 videos in train set
images = {}


def _path_to_frame_nr(path):
    return int(path.split("/")[-1].split(".")[0])


for vid in vids:
    for path, _ in f.samples["train"]:
        if vid in path.split("/")[-2]:
            frame_nr = _path_to_frame_nr(path)
            try:
                images[vid][frame_nr] = path
            except KeyError:
                images[vid] = {frame_nr: path}
    print(vid, len(images[vid]), list(images[vid].keys())[-1])

    assert list(images[vid].keys()) == sorted(list(images[vid].keys()))
    assert list(images[vid].keys())[-1] == len(images[vid]) - 1

#%%
from torchvision.datasets.folder import default_loader
import os
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

loaded_images = {x: [] for x in vids}

transform = Compose(
    (ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
)

for vid in tqdm(vids):
    for image_path in sorted(list(images[vid].values())):
        img = default_loader(os.path.join(f.root, image_path))
        loaded_images[vid].append(transform(img))

# %%
from importlib import reload
import forgery_detection.models.audio.similarity_stuff as s

reload(s)
from forgery_detection.models.audio.utils import ContrastiveLoss
import torch
from collections import OrderedDict

model_path = "/home/sebastian/log/showcasings/17/binary_similarity_net/margin_1/checkpoints/_ckpt_epoch_5.ckpt"

p = s.SimilarityNet().eval()
p.c_loss = ContrastiveLoss(1)
state_dict = torch.load(model_path)["state_dict"]
better_state_dict = OrderedDict()
for key, value in state_dict.items():
    better_state_dict[key.replace("model.", "")] = value

p.load_state_dict(better_state_dict)

#%%
import forgery_detection.data.file_lists as file_lists

reload(file_lists)
audio_features = file_lists.SimpleFileList.load(
    "/mnt/raid5/sebastian/audio_features/mfcc_features_file_list.json"
)

batch = (
    torch.stack(loaded_images[vids[0]][:8]).unsqueeze(0),
    torch.Tensor(
        audio_features.files[
            "original_sequences/youtube/c40/face_images_tracked/" + vids[0]
        ][:8]
    ).unsqueeze(0),
)

out = p.forward(batch)

loss = p.loss(out, torch.ones((1)))

#%%
device = torch.device("cuda:3")
#%%
p = p.to(device)

outputs = {x: [] for x in vids}

for vid in vids:
    curr_images = loaded_images[vid]
    num_samples = len(curr_images) - 7

    for i in tqdm(range(num_samples)):
        with torch.no_grad():
            batch = (
                torch.stack(curr_images[i : i + 8]).unsqueeze(0).to(device),
                torch.Tensor(
                    audio_features.files[
                        "original_sequences/youtube/c40/face_images_tracked/" + vid
                    ][i : i + 8]
                )
                .unsqueeze(0)
                .to(device),
            )
            out = p.forward(batch)
            outputs[vid].append((out[0].cpu(), out[1].cpu()))

#%%
stacked_outputs = {
    vid: (
        torch.cat(list(map(lambda x: x[0], tensors))),
        torch.cat(list(map(lambda x: x[1], tensors))),
    )
    for vid, tensors in outputs.items()
}

#%%
for vid in vids:
    print(
        vid,
        "loss:",
        p.loss(stacked_outputs[vid], torch.ones(len(stacked_outputs[vid][0]))),
    )

for i in range(len(vids)):
    curr = stacked_outputs[vids[i]][0]
    other = stacked_outputs[vids[(i + 1) % len(vids)]][1]
    min_len = min(len(curr), len(other))

    print(
        i,
        (i + 1) % len(vids),
        "loss:",
        p.loss((curr[:min_len], other[:min_len]), torch.ones(min_len)),
    ),

curr = stacked_outputs[vids[1]][0]
other = stacked_outputs[vids[0]][1]
min_len = min(len(curr), len(other))
print(len(curr), len(other))
print(
    "my_mix" "loss:",
    p.loss((curr[:min_len], other[len(other) - min_len :]), torch.ones(min_len)),
),

#%%
import matplotlib.pyplot as plt
import numpy as np

colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.clf()
    plt.figure(figsize=(10, 10))
    for i in range(10):
        inds = np.where(targets == i)[0]
        plt.scatter(
            embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, color=colors[i]
        )
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(vids)


#%%
aud_vid_embeddings = torch.cat([torch.cat(x[1]) for x in stacked_outputs.items()])
targets_embeddings = torch.cat(
    [
        torch.ones_like(torch.cat(x[1])) * idx
        for idx, x in enumerate(stacked_outputs.items())
    ]
)

#%%
aud_vid_embeddings = torch.cat([x[1][0] for x in stacked_outputs.items()])
targets_embeddings = torch.cat(
    [torch.ones_like(x[1][0]) * idx for idx, x in enumerate(stacked_outputs.items())]
)

#%%
aud_vid_embeddings = torch.cat([x[1][1] for x in stacked_outputs.items()])
targets_embeddings = torch.cat(
    [torch.ones_like(x[1][1]) * idx for idx, x in enumerate(stacked_outputs.items())]
)
#%%

stds = torch.std(aud_vid_embeddings, dim=0)
# _, highest_stds_idx = torch.topk(stds, k=2)
aud_vid_embeddings = aud_vid_embeddings[:, highest_stds_idx]
#%%
plot_embeddings(aud_vid_embeddings, targets_embeddings)
plt.savefig("/home/sebastian/differen_video_embeddings_vid_std_fixed.png")
