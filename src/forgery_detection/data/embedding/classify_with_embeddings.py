# flake8: noqa
#%%
from importlib import reload

from forgery_detection.data import set

reload(set)
from forgery_detection.data.utils import resized_crop
from forgery_detection.models.audio.multi_class_classification import PretrainedSyncNet
from forgery_detection.models.audio.multi_class_classification import (
    PretrainedSimilarityNet,
)

p = PretrainedSyncNet().eval()  # .to("cuda:2")
p._shuffle_audio = lambda x: x
# f = FileList.load("/data/ssd1/file_lists/c40/tracked_resampled_faces.json")
f = set.FileList.load(
    "/data/ssd1/file_lists/c40/tracked_resampled_faces_yt_only_112_16_sequence_length.json"
)


#%%
import torch
from forgery_detection.data import loading

reload(loading)
from torchvision import transforms

d = f.get_dataset(
    "test",
    sequence_length=5,
    audio_file="/data/hdd/audio_features/mfcc_features.npy",
    image_transforms=resized_crop(224),
    tensor_transforms=[
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
        lambda x: x * 255,
    ],
)
loader = loading.get_fixed_dataloader(d, batch_size=10, num_workers=1)

#%%
def get_sample(idx, sequence_length=8):
    print(d._samples[idx])
    zeros = [d[idx + i, idx, 0] for i in range(sequence_length)]
    vid = torch.stack([zeros[i][0][0] for i in range(sequence_length)])
    aud = torch.stack([torch.tensor(zeros[i][0][1]) for i in range(sequence_length)])
    return (vid.unsqueeze(0), aud.unsqueeze(0))


with torch.no_grad():
    idx = 0  # 2247 + 20 + 100 + 300 + 600  # 1189500 * 1 + 100 + 1000 - 309

    # samples = [
    #     get_sample(idx + i, sequence_length=p.sequence_length) for i in range(396 - 4)
    # ]
    # samples = (
    #     torch.cat([sample[0] for sample in samples]).to("cuda:2"),
    #     torch.cat([sample[1] for sample in samples]).to("cuda:2"),
    # )
    # samples, target = next(loader.__iter__())
    with torch.no_grad():
        out = p(sampels)
    printerones = torch.sum((out[0] - out[1]).pow(2), dim=1)

    # four = printerones[target == 4]
    # unfour = printerones[target != 4]

    four = torch.zeros((1,))
    unfour = printerones

    print("four", torch.mean(four), torch.std(four), four)
    print("unfour", torch.mean(unfour), torch.std(unfour), unfour)

#%%
import numpy as np
from scipy import signal

vshift = 15


def calc_pdist(feat1, feat2, vshift=10):
    win_size = vshift * 2 + 1

    feat2p = torch.nn.functional.pad(feat2, (0, 0, vshift, vshift))

    dists = []

    for i in range(0, len(feat1)):
        dists.append(
            torch.nn.functional.pairwise_distance(
                feat1[[i], :].repeat(win_size, 1), feat2p[i : i + win_size, :]
            )
        )

    return dists


dists = calc_pdist(out[0].cpu()[:100], out[1].cpu()[:100], vshift)

mdist = torch.mean(torch.stack(dists, 1), 1)

minval, minidx = torch.min(mdist, 0)

offset = vshift - minidx
conf = torch.median(mdist) - minval

fdist = np.stack([dist[minidx].numpy() for dist in dists])
# fdist   = numpy.pad(fdist, (3,3), 'constant', constant_values=15)
fconf = torch.median(mdist).numpy() - fdist
fconfm = signal.medfilt(fconf, kernel_size=9)

np.set_printoptions(formatter={"float": "{: 0.3f}".format})
print("Framewise conf: ")
print(fconfm)
print("AV offset: \t%d \nMin dist: \t%.3f\nConfidence: \t%.3f" % (offset, minval, conf))

#%%
import matplotlib.pyplot as plt

plt.plot(mdist), plt.show()

#%%
import matplotlib.pyplot as plt

plt.imshow(np.array(torch.stack(dists))), plt.show()

#%% load samples extracted by syncnet pipeline
root = "/home/sebastian/repos/syncnet_python/output/pytmp/yt_test"
root = "/data/hdd/c40_resampled/original_sequences/youtube/c40/face_images_tracked/000"
import glob
import os
import cv2.cv2
import numpy as np

images = []

flist = glob.glob(os.path.join(root, "*.png"))
flist.sort()

for fname in flist:
    images.append(cv2.resize(cv2.imread(fname), (224, 224)))

im = np.stack(images, axis=3)
#%%
im = np.transpose(im, (3, 2, 0, 1))

imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())

#%%
from tqdm import tqdm

with torch.no_grad():
    syncnet_out = []
    for idx in tqdm(range(imtv.shape[0] - 5)):
        data = (imtv[idx : idx + 5].unsqueeze(0), samples[1][idx].cpu().unsqueeze(0))
        out = p(data)
        syncnet_out.append(out)

#%%
syncnet_out_vid = torch.cat([x[0] for x in syncnet_out], dim=0)
syncnet_out_audio = torch.cat([x[1] for x in syncnet_out], dim=0)

#%%
dists = calc_pdist(syncnet_out_vid, syncnet_out_audio, vshift)

mdist = torch.mean(torch.stack(dists, 1), 1)

minval, minidx = torch.min(mdist, 0)

offset = vshift - minidx
conf = torch.median(mdist) - minval

fdist = np.stack([dist[minidx].numpy() for dist in dists])
# fdist   = numpy.pad(fdist, (3,3), 'constant', constant_values=15)
fconf = torch.median(mdist).numpy() - fdist
fconfm = signal.medfilt(fconf, kernel_size=9)

np.set_printoptions(formatter={"float": "{: 0.3f}".format})
print("Framewise conf: ")
print(fconfm)
print("AV offset: \t%d \nMin dist: \t%.3f\nConfidence: \t%.3f" % (offset, minval, conf))
