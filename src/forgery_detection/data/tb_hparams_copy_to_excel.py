# flake8: noqa
#%%
from typing import List

import numpy as np


def extract_version(path: str):
    return path.split("/")[1]


def process_numbers(numbers: List[str]):
    return list(map(lambda x: np.format_float_positional(float(x)), numbers))


transforms = True
if transforms:
    header = ["name", "transforms", "lr", "weight_decay", "acc", "loss"]
else:
    header = ["name", "lr", "weight_decay", "acc", "loss"]

experiments = []
experiments_str = []


# with open("tb_hparams_augmentation.txt") as f:
with open("tb_hparams_lr_weight_decay.txt") as f:
    # remove headers
    text = "".join(f.readlines()[10:])
    text = text.replace("\n", "").replace(" ", "")

    # filter out empty lines
    text = text.split(",")
    text = list(filter(lambda x: len(x) > 0, text))

for i in range(len(text) // len(header)):
    start_idx = i * len(header)
    version = extract_version(text[start_idx])
    transform = text[start_idx + 1].replace("\n", "")

    experiment = [version]
    if transforms:
        experiment += [transform]

    numbers = process_numbers(
        text[start_idx + len(experiment) : start_idx + len(header)]
    )
    experiment += numbers
    experiments.append(experiment)
    experiments_str.append("\t".join(experiment))

print("\t".join(header))
print("\n".join(experiments_str))

#%%
import pandas as pd
import numpy as np

df = pd.DataFrame(data=experiments, columns=header, dtype=float)
df.set_index("name", inplace=True)
df.sort_values(by=["acc"], inplace=True, ascending=False)

np_arr = df.to_numpy()[:, :-1]

if transforms:
    np_arr = np_arr[:, 1:]
print(sorted(set(np_arr[:, 2])))
x_axis = dict([(t[1], t[0]) for t in enumerate(sorted(set(np_arr[:, 0])))])
y_axis = dict([(t[1], t[0]) for t in enumerate(sorted(set(np_arr[:, 1])))])

np_heatmap = np.zeros((len(x_axis), len(y_axis))) * np.nan

for row in np_arr:
    x_idx = x_axis[row[0]]
    y_idx = y_axis[row[1]]

    if not np.isnan(np_heatmap[x_idx, y_idx]):
        np_heatmap[x_idx, y_idx] = max(np_heatmap[x_idx, y_idx], row[2])
        print("added higher value for ", x_idx, y_idx)
    else:
        np_heatmap[x_idx, y_idx] = row[2]

np_heatmap = np_heatmap.T
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
im = ax.imshow(np_heatmap)

plt.title(f"Heatmap - {df.columns[0]}, {df.columns[1]} vs. {df.columns[2]}")

cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel(str(df.columns[3]), rotation=-90, va="bottom")

ax.set_xticks(np.arange(len(x_axis)))
ax.set_yticks(np.arange(len(y_axis)))

ax.set_xticklabels(x_axis.keys())
ax.set_yticklabels(y_axis.keys())

plt.xlabel(df.columns[1])
plt.ylabel(df.columns[2])

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

for i in range(len(y_axis)):
    for j in range(len(x_axis)):
        text = ax.text(
            j, i, f"{np_heatmap[i, j]*100:.2f}", ha="center", va="center", color="w"
        )

plt.tight_layout()
plt.savefig("./delete_me.png")
plt.close(fig)

#%%
