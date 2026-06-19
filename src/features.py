from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table


def _as_intensity_image(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None

    image = np.asarray(image)
    if image.ndim == 3:
        return image.mean(axis=2)
    return image


def extract_cell_features(label_image, image=None) -> pd.DataFrame:
    """Extract per-cell features from an integer label mask."""
    label_image = np.asarray(label_image)
    intensity_image = _as_intensity_image(image)

    if intensity_image is not None and intensity_image.shape != label_image.shape:
        raise ValueError("Image and label mask must have the same height and width.")

    properties = (
        "label",
        "centroid",
        "area",
        "perimeter",
        "eccentricity",
        "major_axis_length",
        "minor_axis_length",
        "bbox",
    )
    if intensity_image is not None:
        properties = (*properties, "mean_intensity")

    props = regionprops_table(
        label_image,
        intensity_image=intensity_image,
        properties=properties,
    )
    features = pd.DataFrame(props)

    if features.empty:
        columns = [
            "cell_id",
            "centroid_x",
            "centroid_y",
            "area",
            "perimeter",
            "eccentricity",
            "major_axis_length",
            "minor_axis_length",
            "bbox",
        ]
        if intensity_image is not None:
            columns.append("mean_intensity")
        return pd.DataFrame(columns=columns)

    features = features.rename(
        columns={
            "label": "cell_id",
            "centroid-0": "centroid_y",
            "centroid-1": "centroid_x",
        }
    )
    features["bbox"] = list(
        zip(
            features.pop("bbox-0"),
            features.pop("bbox-1"),
            features.pop("bbox-2"),
            features.pop("bbox-3"),
        )
    )

    ordered_columns = [
        "cell_id",
        "centroid_x",
        "centroid_y",
        "area",
        "perimeter",
        "eccentricity",
        "major_axis_length",
        "minor_axis_length",
        "bbox",
    ]
    if "mean_intensity" in features.columns:
        ordered_columns.append("mean_intensity")

    return features[ordered_columns]


def extract_basic_region_features(label_image) -> pd.DataFrame:
    """Extract minimal region features from an integer label mask."""
    return extract_cell_features(label_image)


# TODO: Add neighborhood, cell-type, and section-aware descriptors.
