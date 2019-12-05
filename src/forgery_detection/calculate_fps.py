import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import click
import numpy as np


@click.command()
@click.option("--folder", required=True, type=click.Path(exists=True))
def calculate_fps(folder):
    folder = Path(folder)

    fps_list = []
    regex = re.compile(r"\d+\sfps")
    for video in folder.iterdir():
        try:
            subprocess.check_output(
                f"/home/sebastian/bin/ffmpeg -i {video}",
                stderr=subprocess.STDOUT,
                shell=True,
            )
        except subprocess.CalledProcessError as e:
            try:
                match = regex.findall(str(e.output))[0].split(" ")[0]
                fps_list.append(float(match))
            except IndexError:
                print(f"Didn't find fps for {video.name}")
    with open("fps.json", "w") as f:
        json.dump(fps_list, f)
    print(np.histogram(fps_list))


if __name__ == "__main__":
    # calculate_fps()
    with open("fps.json", "r") as f:
        fps_list = json.load(f)
        print(Counter(fps_list))
        print(f"missing videos {1000-len(fps_list)}")
