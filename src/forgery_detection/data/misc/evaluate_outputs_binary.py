import pickle
from pathlib import Path

import numpy as np
import torch

from forgery_detection.lightning.logging.const import NAN_TENSOR


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

    binary_target = label // 4

    return pred, binary_target, label


def calculate_accuracy(class_accuracies, binary):
    if binary:
        return (class_accuracies[:4].sum() + 4.0 * class_accuracies[4]) / 8.0
    else:
        return np.mean(class_accuracies)


def class_acc(pred, target, binary, classes=5):
    accs = np.zeros((classes,))
    for c in range(classes):
        class_mask = target == c
        if binary:
            p, t = pred[class_mask], c // 4
        else:
            p, t = pred[class_mask], c
        if p.shape[0] == 0:
            accs[c] = NAN_TENSOR
        else:
            accs[c] = p.eq(t).float().mean()
    return accs


# both inputs are 0 or 1
def binary_acc(pred_arg_maxed, binary_target):
    accs = np.zeros((2,))
    for c in range(2):
        class_mask = binary_target == c
        p = pred_arg_maxed[class_mask]

        if p.shape[0] == 0:
            accs[c] = NAN_TENSOR
        else:
            accs[c] = p.eq(c).float().mean()
    return accs


# first 0 or 1, last 5
def binary_class_acc(pred_arg_maxed, labels):
    accs = np.zeros((5,))
    for c in range(5):
        class_mask = labels == c
        p, t = pred_arg_maxed[class_mask], c // 4

        if p.shape[0] == 0:
            accs[c] = NAN_TENSOR
        else:
            accs[c] = p.eq(t).float().mean()
    return accs


# both in range of 0 to 5
def multi_class_acc(pred_arg_maxed, labels):
    accs = np.zeros((5,))
    for c in range(5):
        class_mask = labels == c
        p, t = pred_arg_maxed[class_mask], c

        if p.shape[0] == 0:
            accs[c] = NAN_TENSOR
        else:
            accs[c] = p.eq(t).float().mean()
    return accs


def get_output_file_names_ordered(folder):
    return sorted(Path(folder).glob("*.pkl"))


def calculate_metrics(outputs_file, binary=True):
    pred, binary_target, label = load_outputs(outputs_file)
    pred = pred.argmax(dim=1)

    if binary:
        # if we want to evaluate the binary case but predict multiple classes we need to
        # map the predictions to the binary case
        if pred.max() > 1:
            pred = pred // 4

        acc = binary_acc(pred, binary_target).mean()
        class_accs = binary_class_acc(pred, label)

    else:
        class_accs = multi_class_acc(pred, label)
        acc = class_accs.mean()

    return acc, class_accs


def print_evaluation_for_test_folder(folder_to_evaluate):

    output_files = get_output_file_names_ordered(folder_to_evaluate)
    for binary in [True, False]:
        train_acc, train_class_accs = calculate_metrics(output_files[0], binary=binary)
        val_acc, val_class_accs = calculate_metrics(output_files[1], binary=binary)

        print(f"binary case: {binary}")
        print(
            f"{train_acc:.2%}/{val_acc:.2%};"
            + "".join("{:.2%};".format(x) for x in train_class_accs)
            + "".join("{:.2%};".format(x) for x in val_class_accs)
        )


if __name__ == "__main__":
    folder_to_evaluate = "/mnt/raid/sebastian/log/runs/TRAIN/r2plus1_3_layer_binary/version_0/test/version_0"

    print_evaluation_for_test_folder(folder_to_evaluate)
