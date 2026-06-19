from __future__ import annotations

import numpy as np
from matplotlib import colormaps


def colorize_label_image(label_image: np.ndarray) -> np.ndarray:
    """Convert an integer label image to an RGB preview."""
    labels = np.asarray(label_image)
    if labels.ndim != 2:
        raise ValueError("Label image preview expects a 2D integer mask.")

    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    positive = labels > 0
    if not np.any(positive):
        return rgb

    normalized = (labels.astype(np.uint64) * 2654435761 % 256).astype(np.uint8)
    colors = (colormaps["tab20"](normalized / 255.0)[..., :3] * 255).astype(np.uint8)
    rgb[positive] = colors[positive]
    return rgb


# TODO: Add overlays, checkerboards, match lines, and registration quality views.
