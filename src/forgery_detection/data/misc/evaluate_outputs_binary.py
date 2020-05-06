import pickle
from pathlib import Path

import numpy as np
import torch


def load_outputs(outputs_file):
    with open(outputs_file, "rb") as f:
        outputs = pickle.load(f)

    if isinstance(outputs[0]["pred"], tuple):
        pred = torch.cat([x["pred"][0] for x in outputs], 0)
        pred_shape = outputs[0]["pred"][0].shape
    else:
        pred = torch.cat([x["pred"] for x in outputs], 0)
        pred_shape = outputs[0]["pred"].shape

    if outputs[0]["target"].shape[0] != pred_shape[0]:
        print(outputs[0]["target"].shape, pred.shape)
        label = torch.cat([x["target"][0] for x in outputs], 0)
    else:
        label = torch.cat([x["target"] for x in outputs], 0)

    target = label // 4

    return pred, target, label


def calculate_accuracy(pred, target):
    labels_hat = torch.argmax(pred, dim=1)
    acc = labels_hat.eq(target).float().mean()
    return acc


def class_acc(pred, target, classes=5):
    accs = np.zeros((5,))
    for c in range(5):
        class_mask = target == c
        p, t = pred[class_mask], c // 4
        accs[c] = p.argmax(dim=1).eq(t).float().mean()
    return accs


def get_output_file_names_ordered(folder):
    return sorted(Path(folder).glob("*.pkl"))


def calculate_metrics(outputs_file):
    pred, target, label = load_outputs(outputs_file)
    acc = calculate_accuracy(pred, target)
    class_accs = class_acc(pred, label)
    return acc, class_accs


if __name__ == "__main__":
    folder_to_evaluate = "/mnt/raid/sebastian/log/runs/TRAIN/r2plus1_3_layer_binary/version_0/test/version_0"

    output_files = get_output_file_names_ordered(folder_to_evaluate)
    train_acc, train_class_accs = calculate_metrics(output_files[0])
    val_acc, val_class_accs = calculate_metrics(output_files[1])

    print(
        f"{train_acc:.2%}/{val_acc:.2%};"
        + "".join("{:.2%};".format(x) for x in train_class_accs)
        + "".join("{:.2%};".format(x) for x in val_class_accs)
    )
