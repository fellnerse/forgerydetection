import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from forgery_detection.data.file_lists import FileList
from forgery_detection.models.image.multi_class_classification import Resnet18
from forgery_detection.models.mixins import PretrainedNet


class Five00BatchesResnet18(
    PretrainedNet(
        "/home/sebastian/log/debug/version_32/checkpoints/_ckpt_epoch_1.ckpt"
    ),
    Resnet18,
):
    pass


def do_error_stuff():
    r = Five00BatchesResnet18().cuda().eval()
    f = FileList.load("/data/ssd1/file_lists/imagenet/ssd_raw.json")
    val_data = f.get_dataset(
        "val", image_transforms=[transforms.Resize(256), transforms.CenterCrop(224)]
    )
    val_data_loader = DataLoader(
        val_data, batch_size=256, shuffle=True, num_workers=2, pin_memory=True
    )
    with torch.no_grad():
        acc = 0
        num_items = 0
        t_bar = tqdm(val_data_loader)
        for batch in t_bar:
            images, targets = batch
            targets, images = targets.cuda(), images.cuda()
            predictions = r.forward(images)
            top1_error = r.calculate_accuracy(predictions, targets)
            acc += top1_error
            num_items += 1
            t_bar.set_description(f"Curr acc: {acc/num_items}")

        acc /= num_items
        print(acc)


if __name__ == "__main__":
    do_error_stuff()
