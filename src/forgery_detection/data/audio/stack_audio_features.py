# flake8: noqa
#%%
import numpy as np

audio_features = np.load(
    "/mnt/raid5/sebastian/audio_features/mfcc_features.npy", allow_pickle=True
)

#%%
from tqdm import tqdm

stacked_audio_features = {}

for video_name, mfcc_features in tqdm(audio_features[()].items()):
    fov = 16
    stacked_mfcc = np.zeros((mfcc_features.shape[0], fov, *mfcc_features.shape[1:]))

    for idx, mfcc in enumerate(mfcc_features):
        for jdx in range(-(fov // 2 - 1), fov // 2 + 1):
            curr_idx = idx + jdx
            if 0 <= curr_idx < mfcc_features.shape[0]:
                stacked_mfcc[curr_idx][fov // 2 - jdx] = mfcc

    # remove this if input is only vector and not matrix (i.e. for deep speech features
    stacked_mfcc = np.reshape(
        stacked_mfcc, (stacked_mfcc.shape[0], -1, *stacked_mfcc.shape[3:])
    )
    stacked_audio_features[video_name] = stacked_mfcc.astype(np.float32)

#%%
np.save(
    "/mnt/raid5/sebastian/audio_features/mfcc_features_stacked.npy",
    stacked_audio_features,
    allow_pickle=True,
)
