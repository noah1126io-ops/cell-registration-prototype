from __future__ import annotations

import pandas as pd
from skimage.measure import regionprops_table


def extract_basic_region_features(label_image) -> pd.DataFrame:
    """Extract minimal region features from an integer label mask."""
    props = regionprops_table(
        label_image,
        properties=("label", "area", "centroid", "bbox"),
    )
    return pd.DataFrame(props)


# TODO: Add morphology, intensity, neighborhood, and section-aware cell descriptors.
