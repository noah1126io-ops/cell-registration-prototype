import numpy as np
import pandas as pd

from src.registration import estimate_affine_transform, identity_transform, transform_cell_centroids


def test_estimate_affine_transform_falls_back_for_empty_density_maps():
    fixed = np.zeros((10, 10), dtype=np.float32)
    moving = np.zeros((10, 10), dtype=np.float32)

    result = estimate_affine_transform(fixed, moving)

    assert result.success is False
    np.testing.assert_allclose(result.affine_matrix, identity_transform())


def test_transform_cell_centroids_applies_affine_matrix():
    features = pd.DataFrame(
        {
            "cell_id": [1],
            "centroid_x": [10.0],
            "centroid_y": [20.0],
            "area": [5.0],
            "eccentricity": [0.1],
        }
    )
    affine = identity_transform()
    affine[0, 2] = 3.0
    affine[1, 2] = -2.0

    transformed = transform_cell_centroids(features, affine)

    assert transformed.loc[0, "centroid_x"] == 13.0
    assert transformed.loc[0, "centroid_y"] == 18.0
