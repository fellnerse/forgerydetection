import tqdm

from forgery_detection.data.file_lists import FileList
from forgery_detection.data.loading import get_fixed_dataloader


def change_class_order():
    file_list = (
        "/mnt/ssd1/sebastian/file_lists/c40/"
        "youtube_Deepfakes_Face2Face_FaceSwap_NeuralTextures_c40_face_images_tracked_100_100_8.json"
    )
    # file_list = "/data/ssd1/file_lists/c40/tracked_resampled_faces.json"
    f_new = FileList.load(file_list)

    f_new.class_to_idx = {
        "Deepfakes": 0,
        "Face2Face": 1,
        "FaceSwap": 2,
        "NeuralTextures": 3,
        "youtube": 4,
    }

    f_new.classes = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "youtube"]

    for split in f_new.samples.values():
        for item in split:
            item[1] = (item[1] + 4) % 5


if __name__ == "__main__":
    # change_class_order()
    # 1 / 0

    file_list = "/data/ssd1/file_lists/c40/trf_100_100_full_size_relative_bb_8_sl.json"
    # file_list = "/data/ssd1/file_lists/c40/tracked_resampled_faces.json"
    f = FileList.load(file_list)

    # first_val_path = f.samples["val"][f.samples_idx["val"][0]]
    val_loader = get_fixed_dataloader(
        f.get_dataset("val", sequence_length=1, should_align_faces=True),
        batch_size=1,
        num_workers=1,
    )
    iter = val_loader.__iter__()
    for i in tqdm.trange(1000):
        next(iter)
