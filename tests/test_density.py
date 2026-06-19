import numpy as np
import pandas as pd

from src.density import create_density_map, label_density_map


def test_create_density_map_from_artificial_centroids():
    features = pd.DataFrame(
        {
            "cell_id": [1, 2],
            "centroid_x": [2.0, 7.0],
            "centroid_y": [3.0, 6.0],
        }
    )

    density = create_density_map(features, image_shape=(10, 12), sigma=1.0)

    assert density.shape == (10, 12)
    assert density.dtype == np.float32
    assert density.max() > 0
    assert density[3, 2] > 0
    assert density[6, 7] > 0


def test_label_density_map_preserves_shape():
    labels = np.zeros((8, 10), dtype=np.int32)
    labels[3, 4] = 1

    density = label_density_map(labels, sigma=1.0)

    assert density.shape == labels.shape
    assert density.dtype == np.float32
    assert density.max() > 0
