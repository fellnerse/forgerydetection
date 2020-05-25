import matplotlib.pyplot as plt
import numpy as np

from forgery_detection.thesis_visualizations.utils import export_pdf
from forgery_detection.thesis_visualizations.utils import figsize

plt.clf()
plt.cla()

fps = np.array("30 25 24 60 15 50 29 18".split(" "), dtype=int)
count = np.array("480 479 25 7 4 2 2 1".split(" "), dtype=int)

count = [x for _, x in sorted(zip(fps, count), key=lambda x: x[0])]
fps = sorted(fps)

fig = plt.figure(figsize=figsize)
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for i, v in enumerate(count):
    if v < 10:
        ax.text(i, v + 5, str(v), color="black", fontweight="bold", ha="center")

plt.bar(range(len(count)), count, tick_label=fps)
plt.xlabel("frame-rate")
plt.ylabel("number of occurrences")
plt.title("Distribution of frame-rates in FaceForensics data")

export_pdf("fps_bar_plot", "method")
