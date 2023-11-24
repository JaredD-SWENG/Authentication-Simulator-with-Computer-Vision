from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from facenet_models import FacenetModel
from matplotlib.patches import Rectangle

# Estimated matching cutoff for cosine-distance between descriptors.
# I.e. if the best proposed match distance:
#    `cosine-dist(d_unknown, d_person)`
# falls above this cutoff, then this is not a sufficiently-strong match,
# and we say that the person associated with `d_unknown` is not recognized.
#
# This value is estimated by computing the cos-dists between descriptors of known
# good-matches, and also computing the cos-dists between known bad-matches.
# One can then plot the respective distributions of these values and look to see
# where the distributions overlap
CUTOFF = 0.65


def show_image_with_detections(image: np.ndarray) -> Tuple[plt.Axes, plt.Figure]:
    """Given the path to an image file, or a numpy array of RGB values,
    show the image and the bounding boxes for any faces within it.

    Parameters
    ----------
    image : Union[PathLike, numpy.ndarray]
        Path to image or (H, W, 3)-shaped image array of RGB values.

    Returns
    -------
    (fig, ax)"""

    model = FacenetModel()

    fig, ax = plt.subplots()
    if not isinstance(image, np.ndarray):
        image = io.imread(str(image))
        if image.shape[-1] == 4:
            image = image[..., :-1]  # png -> RGB
    ax.imshow(image)
    boxes, _, _ = model.detect(image)
    for box in boxes:
        l, t, r, b = box
        x, y = l, b
        width = r - l
        height = t - b

        rect = Rectangle((x, y), width, height, fill=None, color="red", lw=1)
        ax.add_patch(rect)
    return fig, ax


def show_image_with_recognition(
    image: np.ndarray, cutoff: float = CUTOFF
) -> Tuple[plt.Figure, plt.Axes, List[Optional[str]], np.ndarray, np.ndarray]:
    """Plot an image with detection boxes around faces, and if a face is
    "recognized" in the database, annotate the detection with the person's
    name.

    Parameters
    ----------
    image : Union[PathLike, numpy.ndarray]
        Path to an image-file, or an RGB-valued image array.

    cutoff : float
        Maximum cosine distance permitted as being a "match".

    Returns
    -------
    Tuple[matplotlib.Figure, matplotlib.Axis, List[Union[str, None]], List[numpy.ndarray]]
        Given N detected faces, returns a tuple of:
          - the matplotlib figure instance
          - the matplotlib axis instance
          - matches: a list of the N names of the best matches from the database;
              a `None` entry occurs if no good match was found for a given descriptor.
          - detections: a list of the N rectangles containing the face detections
          - descriptors: a list of N 128-dimensional face descriptors"
    """
    from matplotlib.patches import Rectangle

    from .face_db import image_to_best_matches

    if not isinstance(image, np.ndarray):
        image = io.imread(str(image))

        if image.shape[-1] == 4:
            image = image[..., :-1]  # png -> RGB

    matches, detections, descriptors = image_to_best_matches(image, cutoff=cutoff)

    fig, ax = plt.subplots()
    ax.imshow(image)
    for name, det in zip(matches, detections):
        l, t, r, b = det
        x, y = l, b
        width = r - l
        height = t - b

        rect = Rectangle((x, y), width, height, fill=None, color="red", lw=1)
        ax.add_patch(rect)
        if name is not None:
            ax.text(x + 0.1, y + 16.1, name, fontsize=10, color="white")
    return fig, ax, matches, detections, descriptors