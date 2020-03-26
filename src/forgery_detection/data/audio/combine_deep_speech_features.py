from pathlib import Path

import numpy as np
from tqdm import tqdm

from forgery_detection.data.audio.utils import normalize_dict

deep_speech_features_folder = Path("/data/hdd/audio_features/deepspeech")

deep_speech_features = {}

for npy_feature_file in tqdm(sorted(deep_speech_features_folder.glob("*.npy"))):
    features = np.load(npy_feature_file)
    deep_speech_features[
        npy_feature_file.with_suffix("").with_suffix("").with_suffix("").name
    ] = features.astype(np.float32)

audio_features_output_path = Path("/data/hdd/audio_features")
normalize_dict(deep_speech_features)
deep_speech_features_np = np.array(deep_speech_features)
np.save(
    audio_features_output_path / "audio_features_deep_speech.npy",
    deep_speech_features_np,
)
