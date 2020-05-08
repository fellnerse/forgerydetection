# flake8: noqa
#%%
from forgery_detection.data.file_lists import FileList
from forgery_detection.data.utils import resized_crop


f = FileList.load(
    "/home/sebastian/data/file_lists/c40/trf_-1_-1_full_size_relative_bb_8_sl.json"
)
a = f.get_dataset(
    "test", audio_file_list=None, sequence_length=8, image_transforms=resized_crop(112)
)
#%%
f_2 = FileList.load(
    "/home/sebastian/data/file_lists/c40/tracked_resampled_faces_all_112_8_sequence_length.json"
)
a_2 = f_2.get_dataset(
    "test", audio_file_list=None, sequence_length=8, image_transforms=[]
)

#%%
zero = a[((100000, 100000), None)][0]
zero_2 = a_2[((100001, 100001), None)][0]

#%%
print((zero == zero_2).all())
