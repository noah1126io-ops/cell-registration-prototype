from __future__ import annotations

from os import PathLike
from typing import BinaryIO

import numpy as np
import pandas as pd


POINT_COLUMNS = ["point_id", "centroid_x", "centroid_y", "source"]


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


def load_csv_points(source: str | PathLike | BinaryIO, *, point_source: str = "csv") -> pd.DataFrame:
    """Load point coordinates from a CSV file."""
    if hasattr(source, "seek"):
        source.seek(0)

    points = pd.read_csv(source)
    return normalize_point_table(points, point_source=point_source)


def normalize_point_table(points: pd.DataFrame, *, point_source: str = "table") -> pd.DataFrame:
    """Normalize supported point-table variants to the app point schema."""
    if points.empty:
        raise ValueError("Point tables must contain at least one point.")

    if {"centroid_x", "centroid_y"}.issubset(points.columns):
        centroid_x = points["centroid_x"]
        centroid_y = points["centroid_y"]
    elif {"x", "y"}.issubset(points.columns):
        centroid_x = points["x"]
        centroid_y = points["y"]
    else:
        raise ValueError('CSV point tables must contain either "x,y" or "centroid_x,centroid_y" columns.')

    normalized = pd.DataFrame(
        {
            "point_id": points["point_id"] if "point_id" in points.columns else np.arange(1, len(points) + 1),
            "centroid_x": pd.to_numeric(centroid_x, errors="raise").astype(float),
            "centroid_y": pd.to_numeric(centroid_y, errors="raise").astype(float),
            "source": points["source"] if "source" in points.columns else point_source,
        }
    )

    if not np.isfinite(normalized[["centroid_x", "centroid_y"]].to_numpy(dtype=float)).all():
        raise ValueError("Point tables must not contain NaN or infinite coordinates.")

    return normalized[POINT_COLUMNS]


# TODO: Add coordinate-convention metadata for StarDist outputs and HE image coordinate frames.
# TODO: Add upload validation for non-pixel coordinate systems.
