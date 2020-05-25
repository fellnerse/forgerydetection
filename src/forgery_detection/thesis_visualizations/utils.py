from pathlib import Path

from matplotlib import pyplot as plt

SMALL_SIZE = 8
MEDIUM_SIZE = 10
figsize = (2.75, 1.8)
figsize_double = (2.75 * 2, 1.8)


def export_pdf(name, chapter):
    import os

    plt.style.use("default")

    path = Path(f"./visualization_output/{chapter}/{name}.pdf")
    path.parent.mkdir(exist_ok=True)

    os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin/"
    plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
    plt.rc("text", usetex=True)

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=MEDIUM_SIZE)  # fontsize of the figure title

    plt.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close()
