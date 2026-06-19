from __future__ import annotations

from os import PathLike
from typing import BinaryIO

import numpy as np
import pandas as pd


def load_npy_centers(source: str | PathLike | BinaryIO, *, point_source: str = "npy") -> pd.DataFrame:
    """Load precomputed nucleus center coordinates from a .npy file."""
    if hasattr(source, "seek"):
        source.seek(0)

    centers = np.load(source, allow_pickle=False)
    centers = np.asarray(centers)

    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("Nucleus center .npy files must contain an array with shape (n_points, 2).")

    return pd.DataFrame(
        {
            "point_id": np.arange(1, len(centers) + 1, dtype=np.int64),
            "centroid_x": centers[:, 0].astype(float),
            "centroid_y": centers[:, 1].astype(float),
            "source": point_source,
        }
    )


# TODO: Add coordinate-convention metadata for StarDist outputs that store centers as y,x.
# TODO: Add upload validation for empty arrays, NaN coordinates, and non-pixel coordinate systems.
