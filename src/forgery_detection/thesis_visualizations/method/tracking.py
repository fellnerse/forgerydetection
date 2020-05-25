import cv2
import face_recognition
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from forgery_detection.thesis_visualizations.utils import export_pdf
from forgery_detection.thesis_visualizations.utils import figsize

plt.clf()
plt.cla()
img = plt.imread("./visualization_data/0000_full.png")


fig = plt.figure(figsize=figsize)
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

face_locations = face_recognition.face_locations(
    cv2.imread("./visualization_data/0000_full.png")
)

# Display the image
ax.imshow(img)
local_coords = [116, 366, 223, 259]
video_bb = [232, 82, 172, 172]


def coords_to_better_coords(local_coords):
    left_bottom = [local_coords[3], local_coords[2]]
    width = local_coords[1] - local_coords[3]
    height = local_coords[0] - local_coords[2]
    return [left_bottom, width, height]


local_rect = patches.Rectangle(
    *coords_to_better_coords(local_coords), linewidth=1, edgecolor="r", facecolor="none"
)
ax.add_patch(local_rect)

global_rect = patches.Rectangle(
    video_bb[:2], video_bb[2], video_bb[3], linewidth=1, edgecolor="g", facecolor="none"
)
ax.add_patch(global_rect)

plt.title("Visualization of bounding boxes")
ax.legend(["local", "global"])


export_pdf("tracking_video_bb", "method")
