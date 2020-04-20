import logging
import pickle
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal
from torch.utils.data import SequentialSampler

from forgery_detection.data.file_lists import FileList
from forgery_detection.data.file_lists import SimpleFileList
from forgery_detection.data.loading import get_fixed_dataloader
from forgery_detection.lightning.logging.const import AudioMode
from forgery_detection.lightning.logging.utils import get_logger_dir
from forgery_detection.lightning.logging.utils import PythonLiteralOptionGPUs
from forgery_detection.lightning.utils import get_model_and_trainer

logger = logging.getLogger(__file__)


def run_inference_for_video(audio_mode, folder, model, trainer):
    image_file_list = folder / "image_file_list.json"
    audio_file_list = folder / "audio_file_list.json"
    f = FileList.load(str(image_file_list))
    test_video_dataset = f.get_dataset(
        "test",
        image_transforms=model.resize_transform,
        tensor_transforms=model.tensor_augmentation_transforms,
        sequence_length=model.model.sequence_length,
        audio_file_list=SimpleFileList.load(audio_file_list),
        audio_mode=audio_mode,
    )
    loader = get_fixed_dataloader(
        test_video_dataset,
        model.hparams["batch_size"],
        sampler=SequentialSampler,
        num_workers=model.hparams["n_cpu"],
        worker_init_fn=lambda worker_id: np.random.seed(worker_id),
    )
    model.test_dataloader = lambda: loader  # this seems not to work at all
    print(folder, len(loader))
    trainer.test(model)


def calc_pdist(feat1, feat2, vshift=15):
    win_size = vshift * 2 + 1

    feat2p = torch.nn.functional.pad(feat2, (0, 0, vshift, vshift))

    dists = []

    for i in range(0, len(feat1)):
        dists.append(
            torch.nn.functional.pairwise_distance(
                feat1[[i], :].repeat(win_size, 1), feat2p[i : i + win_size, :]
            )
        )

    return dists


@click.command()
@click.option(
    "--evaluation_folder",
    required=True,
    type=click.Path(exists=True),
    help="Folder containing video folders with audio and video file lists.",
)
@click.option(
    "--audio_mode",
    type=click.Choice(AudioMode.__members__.keys()),
    default=AudioMode.EXACT.name,
    help="How the audio should be loaded.",
)
@click.option(
    "--checkpoint_dir",
    required=True,
    type=click.Path(exists=True),
    help="Folder containing logs and checkpoint.",
)
@click.option(
    "--log_dir",
    required=True,
    type=click.Path(exists=True),
    help="Folder used for logging.",
    default="/mnt/raid/sebastian/log",
)
@click.option("--vshift", type=int, default=15, help="Folder used for logging.")
@click.option("--gpus", cls=PythonLiteralOptionGPUs, default="[0]")
@click.option("--debug", is_flag=True)
def run_syncnet_evaluation(vshift, *args, **kwargs):

    model, trainer = get_model_and_trainer(**kwargs)

    audio_mode = AudioMode[kwargs["audio_mode"]]

    img_folder = get_logger_dir(model.logger) / "images"
    img_folder.mkdir()

    evaluation_metrics = {}

    for video in Path(kwargs["evaluation_folder"]).iterdir():
        run_inference_for_video(audio_mode, video, model, trainer)

        with open(get_logger_dir(model.logger) / "outputs.pkl", "rb") as f:
            outputs = pickle.load(f)
            video_logits = torch.cat([x["pred"][0] for x in outputs], 0)
            audio_logtis = torch.cat([x["pred"][1] for x in outputs], 0)

            dists = calc_pdist(video_logits, audio_logtis, vshift=vshift)
            mdist = torch.mean(torch.stack(dists, 1), 1)

            minval, minidx = torch.min(mdist, 0)

            offset = vshift - minidx
            conf = torch.median(mdist) - minval

            fdist = np.stack([dist[minidx].cpu().numpy() for dist in dists])
            # fdist   = numpy.pad(fdist, (3,3), 'constant', constant_values=15)
            fconf = torch.median(mdist).cpu().numpy() - fdist
            fconfm = signal.medfilt(fconf, kernel_size=9)

            np.set_printoptions(formatter={"float": "{: 0.3f}".format})
            print("Framewise conf: ")
            print(fconfm)
            print(
                "AV offset: \t%d \nMin dist: \t%.3f\nConfidence: \t%.3f"
                % (offset, minval, conf)
            )

            dists_npy = np.array([dist.cpu().numpy() for dist in dists])
            plt.clf()
            plt.imshow(dists_npy)
            plt.savefig(img_folder / ("dists_" + (video.with_suffix(".png").name)))
            plt.clf()
            plt.plot(mdist.cpu())
            plt.savefig(img_folder / ("conf_" + (video.with_suffix(".png").name)))
            # return offset.numpy(), conf.numpy(), dists_npy
            evaluation_metrics[video.name] = {
                "offset": offset.cpu(),
                "conf": conf.cpu(),
                "dists": dists_npy,
            }

    with open(get_logger_dir(model.logger) / "evaluation_metrics.pkl", "wb") as f:
        pickle.dump(evaluation_metrics, f)

    mean_conf = torch.stack([x["conf"] for x in evaluation_metrics.values()])
    mean_offset = torch.stack([x["offset"] for x in evaluation_metrics.values()])

    logger.warning(f"Mean conf: {mean_conf}. Mean offset: {mean_offset}")


if __name__ == "__main__":
    run_syncnet_evaluation()
