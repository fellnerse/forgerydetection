from pathlib import Path

import numpy as np
from moviepy.editor import VideoFileClip
from python_speech_features.base import fbank
from tqdm import tqdm

audio_videos_folder = Path("/data/hdd/resampled_audio_videos")

filter_banks = {}

for video in tqdm(sorted(audio_videos_folder.iterdir())[:10]):
    video_clip = VideoFileClip(str(video))
    audio_clip = video_clip.audio
    fb = fbank(
        audio_clip.to_soundarray()[:, 0],
        samplerate=audio_clip.fps,
        winlen=1 / video_clip.fps,
        winstep=1 / video_clip.fps,
        nfft=1024,
        nfilt=40,
    )
    video: Path
    filter_banks[video.with_suffix("").name] = fb

audio_features_output_path = Path("/data/hdd/audio_features")
audio_features_output_path.mkdir(exist_ok=True)
for video_name, filter_bank in filter_banks.items():
    video_path = audio_features_output_path / video_name
    video_path.mkdir(exist_ok=True)
    for idx, feature in enumerate(filter_bank[0]):
        np.save(video_path / f"{idx:03d}.npy", feature)
