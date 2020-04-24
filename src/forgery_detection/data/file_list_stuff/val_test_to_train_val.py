from forgery_detection.data.file_lists import FileList

resampled_file_list = FileList.load(
    "/mnt/ssd1/sebastian/file_lists/c40/tracked_resampled_faces.json"
)
resampled_file_list.samples["train"] = resampled_file_list.samples["val"]
resampled_file_list.samples["val"] = resampled_file_list.samples["test"]

resampled_file_list.samples_idx["train"] = resampled_file_list.samples_idx["val"]
resampled_file_list.samples_idx["val"] = resampled_file_list.samples_idx["test"]

resampled_file_list.save(
    "/mnt/ssd1/sebastian/file_lists/c40/tracked_resampled_faces_val_test_as_train_val.json"
)
