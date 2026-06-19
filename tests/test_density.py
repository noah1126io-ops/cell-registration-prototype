import numpy as np

from src.density import label_density_map


def test_label_density_map_preserves_shape():
    labels = np.zeros((8, 10), dtype=np.int32)
    labels[3, 4] = 1

    density = label_density_map(labels, sigma=1.0)

    assert density.shape == labels.shape
    assert density.dtype == np.float32
    assert density.max() > 0
