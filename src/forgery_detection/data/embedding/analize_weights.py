# flake8: noqa
#%%
from importlib import reload
from pathlib import Path

import forgery_detection.models.audio.audionet as s
import forgery_detection.models.audio.ff_sync_net as f
import forgery_detection.models.audio.noisy_audio as a

reload(s)
reload(f)
import torch
from collections import OrderedDict

# model_path = (
#     "/home/sebastian/delete_me/audionet/version_2/checkpoints/_ckpt_epoch_4.ckpt"
# )
# p = s.AudioNet().eval()

# model_path = "/log/runs/TRAIN/audionet_34/version_1/_ckpt_epoch_4.ckpt"
# p = s.AudioNet34().eval()

# model_path = (
#     "/mnt/raid/sebastian/log/runs/TRAIN/sync_audio_net/version_8/_ckpt_epoch_4.ckpt"
# )
# p = s.SyncAudioNet().eval()

model_path = Path(
    "/home/sebastian/log/consolidation/noisy_audio_experiments/baseline/checkpoints/_ckpt_epoch_7.ckpt"
)
p = a.NoisySyncAudioNet(num_classes=2).eval()

state_dict = torch.load(model_path)["state_dict"]
better_state_dict = OrderedDict()
for key, value in state_dict.items():
    better_state_dict[key.replace("model.", "")] = value

p.load_state_dict(better_state_dict)

#%%
# p = s.AudioNet34().eval()
p = f.FFSyncNetClassifier().eval()
#%%
first_fc = p.out[0]
vid_weights = first_fc.weight[:, :64]
aud_weights = first_fc.weight[:, 1024:]

import matplotlib.pyplot as plt

plt.clf()
plt.hist(
    vid_weights.reshape(-1).detach().numpy(),
    100,
    log=True,
    label="vid_weights",
    range=(-0.6, 0.6),
)
plt.hist(
    aud_weights.reshape(-1).detach().numpy(),
    100,
    log=True,
    label="aud_weights",
    range=(-0.6, 0.6),
)
plt.legend()

# plt.savefig("/home/sebastian/delete_me/hist.png")
# plt.savefig("/home/sebastian/hist.png")
plt.savefig(
    model_path.parent.parent / f"hist{model_path.with_suffix('').name}_ranged.png"
)
