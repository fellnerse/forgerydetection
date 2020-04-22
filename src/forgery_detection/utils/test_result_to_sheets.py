# flake8: noqa
#%%


def tensor(number, *args, **kwargs):
    return number


test_result_train = {
    "metrics/acc": {"test": tensor(0.9887)},
    "metrics/class_acc": {
        "Deepfakes": tensor(0.9995),
        "Face2Face": tensor(0.9927),
        "FaceSwap": tensor(0.9942),
        "NeuralTextures": tensor(0.9738),
        "youtube": tensor(0.9833),
    },
    "metrics/loss": {"test": tensor(0.0326, device="cuda:0")},
    "val_acc": 0.988726019859314,
}

test_result_val = {
    "metrics/acc": {"test": tensor(0.8036)},
    "metrics/class_acc": {
        "Deepfakes": tensor(0.9333),
        "Face2Face": tensor(0.8114),
        "FaceSwap": tensor(0.8698),
        "NeuralTextures": tensor(0.7376),
        "youtube": tensor(0.6657),
    },
    "metrics/loss": {"test": tensor(0.7729, device="cuda:0")},
    "val_acc": 0.8035615086555481,
}

tab = "\t"

out_string = (
    f"{test_result_train['val_acc']:.2%}\t"
    f"{test_result_val['val_acc']:.2%}\t\t"
    f"{''.join('{:.2%}{}'.format(x,tab) for x in test_result_train['metrics/class_acc'].values())}"
    f"{''.join('{:.2%}{}'.format(x,tab) for x in test_result_val['metrics/class_acc'].values())}"
)

print(out_string)
