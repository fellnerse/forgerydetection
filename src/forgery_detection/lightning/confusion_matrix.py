import itertools

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

# https://www.tensorflow.org/tensorboard/image_summaries


def plot_cm(cm, class_names):
    """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


def plot_to_image(fig):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
    # # Save the plot to a PNG in memory.
    # buf = io.BytesIO()
    # plt.savefig(buf, format="png")
    # # Closing the figure prevents it from being displayed directly inside
    # # the notebook.
    # plt.close(figure)
    # buf.seek(0)
    # # Convert PNG buffer to TF image
    # image = tf.image.decode_png(buf.getvalue(), channels=4)
    # # Add the batch dimension
    # image = tf.expand_dims(image, 0)
    #
    # return image
    # todo can we do it like the tf-example above?
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(
        fig.canvas.get_width_height()[::-1] + (3,)
    )
    return image_from_plot


def generate_confusion_matrix_image(pred, target):
    cm = confusion_matrix(pred.cpu(), torch.argmax(target, dim=1).cpu())
    figure = plot_cm(cm, class_names=["fake", "real"])
    cm_image = plot_to_image(figure)
    return cm_image


def log_confusion_matrix_image(global_step, pred, target, logger):
    cm_image = generate_confusion_matrix_image(target, pred)
    logger.experiment.add_image(
        "confusion matrix", cm_image, dataformats="HWC", global_step=global_step
    )
