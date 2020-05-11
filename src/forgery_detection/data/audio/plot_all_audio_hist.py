import matplotlib.pyplot as plt
import numpy as np

mfcc = np.load(
    "/home/sebastian/data/audio_features/mfcc_features.npy", allow_pickle=True
)[()]
all = np.concatenate(list(mfcc.values()))
all_audio_data = all

shaped_all = np.reshape(all, (-1,))

plt.clf(), plt.hist(np.reshape(all, (-1)), 1000), plt.savefig(
    "/home/sebastian/all_audio_data_hist.png"
)

plt.clf(), plt.hist(
    shaped_all + np.random.normal(0, 1, shaped_all.shape).astype(shaped_all.dtype), 1000
), plt.savefig("/home/sebastian/all_audio_data_hist.png")
