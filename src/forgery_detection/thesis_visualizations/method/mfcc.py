import cv2
import matplotlib.pyplot as plt
import numpy as np

from forgery_detection.thesis_visualizations.utils import export_pdf
from forgery_detection.thesis_visualizations.utils import figsize

plt.clf()
plt.cla()

start = 170

fig = plt.figure(figsize=figsize)

data = []
for i in range(start, start + 8):
    data += [plt.imread(f"./visualization_data/{i:04d}.png")]

data = np.concatenate(data, axis=1)
# data = np.resize(data, (32, 32*8, 3))
data = cv2.resize(data, dsize=(8 * 4 * 100, 4 * 100), interpolation=cv2.INTER_CUBIC)
print(data.shape)
ax = plt.subplot(111)
# ax.imshow(data, interpolation="none")
# ax = plt.subplot(212)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

mfcc = np.load("./visualization_data/000.npy", allow_pickle=True)[
    :, start : start + 4 * 8
]
# mfcc -= np.expand_dims(np.mean(mfcc, axis=0), axis=0)
# mfcc /= np.expand_dims(np.std(mfcc, axis=0), axis=0)
mfcc -= np.min(mfcc, axis=0)
mfcc /= np.max(mfcc, axis=0)
mfcc = cv2.resize(mfcc, dsize=(32 * 100, 13 * 100), interpolation=cv2.INTER_NEAREST)
mfcc = np.stack((mfcc, mfcc, mfcc), axis=2)


print(mfcc.shape, data.shape)
data = np.concatenate((data, mfcc), axis=0)

# mfcc = np.concatenate((np.zeros(*mfcc.shape), mfcc), dim=0)
# ax.imshow(data, extent=[0, 320, 0, 13 * 10])
ax.imshow(data, interpolation="none")
for i in range(1, 8):
    ax.axvline(x=i * 4 * 100 - 0.5, linewidth=1, color="#185D5E")
plt.xlabel("time in ms")
plt.ylabel("frequency bands")
plt.xticks(np.arange(8) / 8 * 3200, np.arange(8) * 4)
plt.yticks([])
plt.title("Extracted MFCC-features")
ax.yaxis.set_label_coords(-0.02, 0.375)
# plt.show()
export_pdf("mfcc_features", "method")
