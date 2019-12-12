from pathlib import Path

from forgery_detection.data.face_forensics import Compression
from forgery_detection.data.face_forensics import DataType
from forgery_detection.data.face_forensics import FaceForensicsDataStructure

audio_videos_folder = Path("/data/hdd/resampled_audio_videos")
existing_videos = [x.with_suffix("").name for x in audio_videos_folder.iterdir()]

resampled_videos_data_structure = FaceForensicsDataStructure(
    "/data/hdd/c40_resampled",
    compressions=Compression.c40,
    data_types=DataType.resampled_videos,
)

for subdir in resampled_videos_data_structure.get_subdirs():
    delete_list = []
    for video in sorted(subdir.iterdir()):
        if video.with_suffix("").name.split("_")[0] not in existing_videos:
            delete_list.append(video)
            video.unlink()
    print(len(delete_list), subdir)
