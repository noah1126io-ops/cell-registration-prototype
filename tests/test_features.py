import numpy as np

from src.features import extract_basic_region_features


def test_extract_basic_region_features_returns_labels_and_area():
    labels = np.array(
        [
            [0, 1, 1],
            [0, 2, 0],
            [0, 2, 2],
        ],
        dtype=np.int32,
    )

    features = extract_basic_region_features(labels)

    assert list(features["label"]) == [1, 2]
    assert list(features["area"]) == [2.0, 3.0]
