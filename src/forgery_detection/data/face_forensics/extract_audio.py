from pathlib import Path

import numpy as np
from moviepy.editor import VideoFileClip
from python_speech_features.base import fbank
from tqdm import tqdm

audio_videos_folder = Path("/data/hdd/resampled_audio_videos")

filter_banks = {}

for video in tqdm(sorted(audio_videos_folder.iterdir())):
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
    filter_banks[video.with_suffix("").name] = fb[0].astype(np.float32)

audio_features_output_path = Path("/data/hdd/audio_features")
audio_features_output_path.mkdir(exist_ok=True)
filter_banks_np = np.array(filter_banks)
np.save(audio_features_output_path / "audio_features_.npy", filter_banks_np)
