# flake8: noqa
#%%


def tensor(number, *args, **kwargs):
    return number


test_result_train = {
    "metrics/acc": {"test": tensor(0.9759)},
    "metrics/class_acc": {
        "Deepfakes": tensor(0.9991),
        "Face2Face": tensor(0.9833),
        "FaceSwap": tensor(0.9945),
        "NeuralTextures": tensor(0.9070),
        "youtube": tensor(0.9958),
    },
    "metrics/loss": {"test": tensor(0.0663, device="cuda:0")},
    "val_acc": 0.9759498834609985,
}

test_result_val = {
    "metrics/acc": {"test": tensor(0.8082)},
    "metrics/class_acc": {
        "Deepfakes": tensor(0.9077),
        "Face2Face": tensor(0.7776),
        "FaceSwap": tensor(0.9088),
        "NeuralTextures": tensor(0.6667),
        "youtube": tensor(0.7806),
    },
    "metrics/loss": {"test": tensor(0.7202, device="cuda:0")},
    "val_acc": 0.8082452416419983,
}

tab = "\t"

out_string = (
    f"{test_result_train['val_acc']:.2%}\t"
    f"{test_result_val['val_acc']:.2%}\t\t"
    f"{''.join('{:.2%}{}'.format(x,tab) for x in test_result_train['metrics/class_acc'].values())}"
    f"{''.join('{:.2%}{}'.format(x,tab) for x in test_result_val['metrics/class_acc'].values())}"
)

print(out_string)
