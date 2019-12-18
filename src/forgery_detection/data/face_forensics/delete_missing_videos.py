from pathlib import Path

from forgery_detection.data.face_forensics import Compression
from forgery_detection.data.face_forensics import DataType
from forgery_detection.data.face_forensics import FaceForensicsDataStructure


def get_delete_take_list(data_structure):
    lists = {}
    for subdir in data_structure.get_subdirs():
        delete_list = []
        take_list = []
        for video in sorted(subdir.iterdir()):
            if "Deepfakes" in str(subdir) or "FaceSwap" in str(subdir):
                # these two manipulation methods keep the expression the same but replace
                # the face -> the audio stays the same
                delete = video.with_suffix("").name.split("_")[0] not in existing_videos
            elif "Face2Face" in str(subdir) or "NeuralTextures" in str(subdir):
                # these two manipulation methods replace only the expression of the video
                # but keep the face the same -> the audio is different
                delete = video.with_suffix("").name.split("_")[1] not in existing_videos
            elif "youtube" in str(subdir):
                delete = video.with_suffix("").name not in existing_videos
            else:
                print("something seems of")
                raise ValueError("name of subdir is wrong", subdir)

            if delete:
                delete_list.append(video.name)
                # face_images_tracked_path = (
                #     video.parent.parent / "face_images_tracked" / video.with_suffix("")
                #     .name
                # )
                # shutil.rmtree(face_images_tracked_path)
                # video.unlink()
            else:
                take_list.append(video.name)
        lists[subdir.parent.parent.name] = {"delete": delete_list, "take": take_list}
    return lists


audio_videos_folder = Path("/data/hdd/resampled_audio_videos")
existing_videos = [x.with_suffix("").name for x in audio_videos_folder.iterdir()]

resampled_videos_data_structure = FaceForensicsDataStructure(
    "/data/hdd/c40_resampled",
    compressions=Compression.c40,
    data_types=DataType.resampled_videos,
)

missing_videos_data_structure = FaceForensicsDataStructure(
    "/data/hdd/c40_resampled_missing",
    compressions=Compression.c40,
    data_types=DataType.resampled_videos,
)

resampled_lists = get_delete_take_list(resampled_videos_data_structure)
for subdir, list_dict in resampled_lists.items():
    print(len(list_dict["delete"]), len(list_dict["take"]), subdir)

print("##############")

missing_lists = get_delete_take_list(missing_videos_data_structure)
for subdir, list_dict in missing_lists.items():
    print(len(list_dict["delete"]), len(list_dict["take"]), subdir)

# now lets remove videos that are in resampled_lists from missing_lists
for subdir, list_dict in resampled_lists.items():
    resampled_take = list_dict["take"]
    missing_take = missing_lists[subdir]["take"]
    missing_delete = missing_lists[subdir]["delete"]
    for take in resampled_take:
        missing_take.remove(take)
        missing_delete.append(take)

print("##############")

for subdir, list_dict in missing_lists.items():
    print(len(list_dict["delete"]), len(list_dict["take"]), subdir)
print("##############")

# and now we generate the path again for deletion
for subdir in missing_videos_data_structure.get_subdirs():
    delete_list = []
    take_list = []
    for video in sorted(subdir.iterdir()):
        if video.name in missing_lists[subdir.parent.parent.name]["delete"]:
            delete_list.append(video)
            video.unlink()
        else:
            take_list.append(video)
    print(len(delete_list), len(take_list), subdir)
