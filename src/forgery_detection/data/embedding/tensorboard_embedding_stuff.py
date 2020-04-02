# flake8: noqa
#%%
from pytorch_lightning.logging import TestTubeLogger

logger = TestTubeLogger(save_dir="/log", name="test_embedding")
writer = logger.experiment

#%%
import torch

bs = 11 * 5
video_logits = torch.randn((bs, 64))
audio_logits = torch.randn((bs, 64))
target = torch.randint(0, 5, (bs,))
meta_data = list(map(lambda x: ["yt", "nt", "f2f", "deep", "swap"][x], target)) + list(
    map(lambda x: ["yt", "nt", "f2f", "deep", "swap"][x] + "_audio", target)
)


def get_colour(target, video=True):
    base_value = 255 if video else 128
    colour_dict = [
        torch.ones((3, 32, 32))
        * torch.tensor([base_value * 1, base_value * 0, base_value * 0])
        .unsqueeze(1)
        .unsqueeze(1),
        torch.ones((3, 32, 32))
        * torch.tensor([base_value * 0, base_value * 1, base_value * 1])
        .unsqueeze(1)
        .unsqueeze(1),
        torch.ones((3, 32, 32))
        * torch.tensor([base_value * 1, base_value * 0, base_value * 1])
        .unsqueeze(1)
        .unsqueeze(1),
        torch.ones((3, 32, 32))
        * torch.tensor([base_value * 0, base_value * 1, base_value * 0])
        .unsqueeze(1)
        .unsqueeze(1),
        torch.ones((3, 32, 32))
        * torch.tensor([base_value * 0, base_value * 0, base_value * 1])
        .unsqueeze(1)
        .unsqueeze(1),
    ]
    return torch.stack(list(map(lambda x: colour_dict[x], target)))


label_image = torch.cat(
    (get_colour(target, video=True), get_colour(target, video=False))
)

#%%
writer.add_embedding(
    torch.cat((video_logits, audio_logits)),
    metadata=meta_data,
    label_img=label_image,
    global_step=0,
)
