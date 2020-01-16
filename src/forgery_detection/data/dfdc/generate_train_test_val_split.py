#%% # noqa E265
import json
import logging
from pathlib import Path


logger = logging.getLogger(__file__)
root_dir = Path("/data/hdd/dfdc")
all_meta_data = {}

for folder in sorted(root_dir.iterdir()):
    try:
        meta_data_file = next(folder.glob("*.json"))
        with open(meta_data_file, "r") as f:
            meta_data: dict = json.load(f)
            for key, value in meta_data.items():
                all_meta_data[folder.name + "/" + key] = value
    except StopIteration:
        logger.warning(f"Ignoring {folder}. Does not contain a .json file.")

with open(root_dir / "all_metadata.json", "w") as f:
    json.dump(all_meta_data, f)

train_data_names = list(
    map(
        lambda item: item[0].split("/")[-1].split(".")[0],
        filter(lambda item: item[1]["split"] == "train", all_meta_data.items()),
    )
)
# there are no val images -> have to create those myself
val_data_names = list(
    map(
        lambda item: item[0].split("/")[-1].split(".")[0],
        filter(lambda item: item[1]["split"] == "val", all_meta_data.items()),
    )
)
