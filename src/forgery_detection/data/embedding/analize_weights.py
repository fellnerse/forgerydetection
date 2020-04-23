# flake8: noqa
#%%
from importlib import reload

import forgery_detection.models.audio.audionet as s
import forgery_detection.models.audio.ff_sync_net as f

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

model_path = "/home/sebastian/log/debug/version_92/_ckpt_epoch_0_v6.ckpt"
p = f.FFSyncNetClassifier().eval()

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
vid_weights = first_fc.weight[:, :1024]
aud_weights = first_fc.weight[:, 1024:]

import matplotlib.pyplot as plt

plt.clf()
plt.hist(vid_weights.reshape(-1).detach().numpy(), 100, log=True, label="vid_weights")
plt.hist(aud_weights.reshape(-1).detach().numpy(), 100, log=True, label="aud_weights")
plt.legend()

# plt.savefig("/home/sebastian/delete_me/hist.png")
plt.savefig("/home/sebastian/hist.png")
