from forgery_detection.data.file_lists import FileList
from forgery_detection.data.loading import get_fixed_dataloader

if __name__ == "__main__":
    file_list = (
        "/mnt/ssd1/sebastian/file_lists/c40/"
        "youtube_Deepfakes_Face2Face_FaceSwap_NeuralTextures_c40_face_images_tracked_100_100_8.json"
    )
    f = FileList.load(file_list)

    # first_val_path = f.samples["val"][f.samples_idx["val"][0]]
    val_loader = get_fixed_dataloader(
        f.get_dataset("val", sequence_length=1, should_align_faces=True),
        batch_size=1,
        num_workers=1,
    )
    iter = val_loader.__iter__()
    next(iter)
    next(iter)
    next(iter)
    next(iter)
    next(iter)
    next(iter)
