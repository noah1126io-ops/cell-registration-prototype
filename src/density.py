from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


def create_density_map(cell_features: pd.DataFrame, image_shape: tuple[int, int], sigma: float) -> np.ndarray:
    """Create a density map by placing Gaussian kernels at cell centroids."""
    height, width = image_shape[:2]
    impulses = np.zeros((height, width), dtype=np.float32)

    if cell_features.empty:
        return impulses

    required_columns = {"centroid_x", "centroid_y"}
    missing_columns = required_columns - set(cell_features.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Cell features are missing required columns: {missing}")

    xs = np.rint(cell_features["centroid_x"].to_numpy(dtype=float)).astype(int)
    ys = np.rint(cell_features["centroid_y"].to_numpy(dtype=float)).astype(int)

    valid = (0 <= xs) & (xs < width) & (0 <= ys) & (ys < height)
    np.add.at(impulses, (ys[valid], xs[valid]), 1.0)

    return gaussian_filter(impulses, sigma=float(sigma)).astype(np.float32)


def label_density_map(label_image: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """Create a simple smoothed density map from non-background labels."""
    foreground = np.asarray(label_image) > 0
    return gaussian_filter(foreground.astype(np.float32), sigma=sigma)


# TODO: Add cell-type-specific density maps and multi-scale density features.
