import matplotlib.pyplot as plt

from forgery_detection.thesis_visualizations.utils import export_pdf
from forgery_detection.thesis_visualizations.utils import figsize

plt.clf()
plt.cla()
img = plt.imread("./visualization_data/0000_tracked.png")


fig = plt.figure(figsize=figsize)
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.imshow(img)
plt.title("Global crop")


export_pdf("tracking_crop", "method")
