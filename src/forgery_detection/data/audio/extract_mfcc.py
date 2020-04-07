from pathlib import Path

import numpy as np
import python_speech_features
from scipy.io import wavfile
from tqdm import tqdm


audios_folder = Path("/data/hdd/ff_audio_only_16k")

mfccs = {}

for wav in tqdm(sorted(audios_folder.iterdir())):
    sample_rate, audio = wavfile.read(wav.absolute())
    mfcc_ = zip(*python_speech_features.mfcc(audio, sample_rate))
    mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
    mfcc = np.stack([np.array(i) for i in mfcc]).transpose((1, 0))

    # the audio never aligns perfectly with the video -> if the resulting features are
    # not divisible by 4 (100hz / 4 = 25fps) we need to pad it with zeros
    missing_audio = mfcc.shape[0] % 4
    if missing_audio:
        mfcc = np.concatenate((mfcc, np.zeros((4 - missing_audio, 13))))

    # 4 consecutive windows correspond to one video frame
    mfcc = np.reshape(mfcc, (-1, 4, 13))
    wav: Path
    mfccs[wav.with_suffix("").name] = mfcc.astype(np.float32)


audio_features_output_path = Path("/data/hdd/audio_features")
audio_features_output_path.mkdir(exist_ok=True)
mfccs_np = np.array(mfccs)
np.save(audio_features_output_path / "mfcc_features.npy", mfccs_np)
