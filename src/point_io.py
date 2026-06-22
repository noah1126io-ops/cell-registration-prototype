from __future__ import annotations

from os import PathLike
from typing import BinaryIO

import numpy as np
import pandas as pd


def load_npy_centers(
    source: str | PathLike | BinaryIO,
    *,
    point_source: str = "npy",
    coordinate_order: str = "xy",
) -> pd.DataFrame:
    """Load precomputed nucleus center coordinates from a .npy file."""
    if coordinate_order not in {"xy", "yx"}:
        raise ValueError('coordinate_order must be "xy" or "yx".')

    if hasattr(source, "seek"):
        source.seek(0)

    centers = np.load(source, allow_pickle=False)
    centers = np.asarray(centers)

    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("Nucleus center .npy files must contain an array with shape (n_points, 2).")
    if centers.shape[0] == 0:
        raise ValueError("Nucleus center .npy files must contain at least one point.")
    if not np.issubdtype(centers.dtype, np.number):
        raise ValueError("Nucleus center .npy files must contain numeric coordinates.")

    centers = centers.astype(float)
    if not np.isfinite(centers).all():
        raise ValueError("Nucleus center .npy files must not contain NaN or infinite coordinates.")

    if coordinate_order == "xy":
        centroid_x = centers[:, 0]
        centroid_y = centers[:, 1]
    else:
        centroid_y = centers[:, 0]
        centroid_x = centers[:, 1]

    return pd.DataFrame(
        {
            "point_id": np.arange(1, len(centers) + 1, dtype=np.int64),
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "source": point_source,
        }
    )


# TODO: Add coordinate-convention metadata for StarDist outputs and HE image coordinate frames.
# TODO: Add upload validation for non-pixel coordinate systems.
