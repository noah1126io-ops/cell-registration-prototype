from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def label_density_map(label_image: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """Create a simple smoothed density map from non-background labels."""
    foreground = np.asarray(label_image) > 0
    return gaussian_filter(foreground.astype(np.float32), sigma=sigma)


# TODO: Add cell-type-specific density maps and multi-scale density features.
