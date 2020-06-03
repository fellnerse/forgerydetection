import csv
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from forgery_detection.lightning.system import Supervised
from forgery_detection.thesis_visualizations.utils import export_pdf
from forgery_detection.thesis_visualizations.utils import figsize


def save_mlp_weight_histogram(
    ckpt="/mnt/raid/sebastian/log/consolidated_results/5.Unfrozen/ff_syncnet_end2end/version_1/checkpoints/_ckpt_epoch_3.ckpt",
    vid_embedding_size=1024,
    audio_embedding_size=1024,
    use_merge_conv=False,
):

    model_path = Path(ckpt)

    with open(model_path.parent.parent / "meta_tags.csv") as f:
        csv_reader = csv.reader(f, delimiter=",")
        model = None
        for line in csv_reader:
            if line[0] == "model":
                model = line[1]
                break
        if model is None:
            raise RuntimeError("model name not found")

    print("model using: ", model)

    m = Supervised.MODEL_DICT[model](num_classes=2).eval()
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)[
        "state_dict"
    ]
    better_state_dict = OrderedDict()
    for key, value in state_dict.items():
        better_state_dict[key.replace("model.", "")] = value

    m.load_state_dict(better_state_dict)

    if use_merge_conv:
        embedding_layer = m.merge_conv[0][
            0
        ]  # merge_conv is sequential, with [0] conv2plus1d which again is a sequential with [0] conv3d
    else:
        embedding_layer = m.out[0]

    concatenated_embedding_size = embedding_layer.weight.shape[1]
    print("size of mlp input: ", concatenated_embedding_size)
    if concatenated_embedding_size != vid_embedding_size + audio_embedding_size:
        raise ValueError(
            "summing video and audio embedding size up does not result in mlp input size: ",
            f"{vid_embedding_size}+{audio_embedding_size}={vid_embedding_size+audio_embedding_size} != {concatenated_embedding_size}",
            f"full shape: {embedding_layer.weight.shape}",
        )
    vid_weights = embedding_layer.weight[:, :vid_embedding_size]
    aud_weights = embedding_layer.weight[:, vid_embedding_size:]

    vid_weights = vid_weights.reshape(-1).detach().numpy()
    aud_weights = aud_weights.reshape(-1).detach().numpy()

    name = (
        f"hist_"
        + f"{model_path.parent.parent.parent.name + model_path.with_suffix('').name}"
    )

    with open(f"./visualization_output/results/{name}.txt", "w") as f:
        f.write(f"vid: {np.mean(vid_weights)} {np.std(vid_weights)}")
        f.write("\n")
        f.write(f"aud: {np.mean(aud_weights)} {np.std(aud_weights)}")

        print("vid:", np.mean(vid_weights), np.std(vid_weights))
        print("aud:", np.mean(aud_weights), np.std(aud_weights))

    do_hist(aud_weights, model_path, vid_weights, True)
    do_hist(aud_weights, model_path, vid_weights, False)


def do_hist(aud_weights, model_path, vid_weights, use_log_y_scale):
    plt.clf()
    plt.cla()
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    range = (-0.15, 0.15)
    # range = (-3, 3)
    plt.hist(
        vid_weights,
        100,
        log=use_log_y_scale,
        label="$m_v$",
        range=range,
        histtype="stepfilled",
        bottom=0,
    )
    plt.hist(
        aud_weights,
        100,
        log=use_log_y_scale,
        label="$m_a$",
        range=range,
        histtype="stepfilled",
        bottom=0,
    )
    fig.legend()

    plt.xlabel("weight value")
    plt.ylabel("number of occurrences")
    # plt.title("Distribution of weights in merge layer")

    export_pdf(
        f"hist_"
        f"{model_path.parent.parent.parent.name + model_path.with_suffix('').name}"
        f"{'_log_y' if use_log_y_scale else ''}",
        "results",
    )


if __name__ == "__main__":
    root_dir = "/home/sebastian/log/consolidation/audio_merging"
    # root_dir = "/mnt/raid/sebastian/log/consolidated_results/5.Unfrozen"

    # unfrozen late merging
    save_mlp_weight_histogram(
        ckpt=root_dir + "/ff_syncnet_end2end/version_1/checkpoints/_ckpt_epoch_2.ckpt",
        vid_embedding_size=1024,
        audio_embedding_size=1024,
    )

    # unfrozen middle merging
    save_mlp_weight_histogram(
        ckpt=root_dir
        + "/middle_merge_net_3_layer/version_0/checkpoints/_ckpt_epoch_2.ckpt",
        vid_embedding_size=128,
        audio_embedding_size=128,
        use_merge_conv=True,
    )

    # unfrozen early merging
    save_mlp_weight_histogram(
        ckpt=root_dir + "/a_early_merge_net/version_3/checkpoints/_ckpt_epoch_2.ckpt",
        vid_embedding_size=64,
        audio_embedding_size=64,
        use_merge_conv=True,
    )

    # our proposed method
    save_mlp_weight_histogram(
        ckpt="/mnt/raid5/sebastian/umoja_consolidated/method/late_merging/version_3/checkpoints/_ckpt_epoch_2.ckpt",
        vid_embedding_size=512,
        audio_embedding_size=512,
    )
    # method without filter
    save_mlp_weight_histogram(
        ckpt="/mnt/raid5/sebastian/umoja_consolidated/method/non_filter/late_merging_no_filter/version_0/checkpoints/_ckpt_epoch_2.ckpt",
        vid_embedding_size=512,
        audio_embedding_size=512,
    )

    # multi modal model
    save_mlp_weight_histogram(
        ckpt="/mnt/raid5/sebastian/umoja_consolidated/method/multimodalmodel/frozen_r2+1(4)_resnet18(4)_bn_lrelu_filter_non_detach/checkpoints/_ckpt_epoch_4.ckpt",
        vid_embedding_size=512,
        audio_embedding_size=512,
    )

    # different video
    save_mlp_weight_histogram(
        ckpt="/mnt/raid5/sebastian/umoja_consolidated/1.audio_setup_works_bce/frozen_r2+1(4)_resnet18(4)_bn_lrelu_filtered/checkpoints/_ckpt_epoch_17.ckpt",
        vid_embedding_size=512,
        audio_embedding_size=512,
    )
    # different video smaller fov
    # save_mlp_weight_histogram(
    #     ckpt="/mnt/raid5/sebastian/umoja_consolidated/2.Filter/2.non_filter/version_0/checkpoints/_ckpt_epoch_2.ckpt",
    #     vid_embedding_size=512,
    #     audio_embedding_size=512,
    # )
    save_mlp_weight_histogram(
        ckpt="/mnt/raid5/sebastian/umoja_consolidated/3.FoV/3.FoV/version_0/checkpoints/_ckpt_epoch_2.ckpt",
        vid_embedding_size=512,
        audio_embedding_size=512,
    )
    # 50 shift
    save_mlp_weight_histogram(
        ckpt="/mnt/raid5/sebastian/umoja_consolidated/4.50shift/version_8/checkpoints/_ckpt_epoch_30.ckpt",
        vid_embedding_size=512,
        audio_embedding_size=512,
    )
    # unfrozen method
    save_mlp_weight_histogram(
        ckpt="/mnt/raid5/sebastian/umoja_consolidated/method/unfrozen_with_audio/temp/checkpoints/_ckpt_epoch_1.ckpt",
        vid_embedding_size=512,
        audio_embedding_size=512,
    )

    # audionet34 with frozen syncnet
    save_mlp_weight_histogram(
        ckpt="/home/sebastian/log/showcasings/17_audionet_stuff/sync_audionet/version_8/embedding_visualization_checkpoint_folder/_ckpt_epoch_4.ckpt",
        vid_embedding_size=128,
        audio_embedding_size=1024,
    )
