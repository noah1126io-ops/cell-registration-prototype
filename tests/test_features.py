import numpy as np

from src.features import extract_basic_region_features, extract_cell_features


def test_extract_cell_features_returns_expected_columns_and_values():
    labels = np.array(
        [
            [0, 1, 1],
            [0, 2, 0],
            [0, 2, 2],
        ],
        dtype=np.int32,
    )

    features = extract_cell_features(labels)

    assert list(features["cell_id"]) == [1, 2]
    assert list(features["area"]) == [2.0, 3.0]
    assert features.loc[0, "centroid_x"] == 1.5
    assert features.loc[0, "centroid_y"] == 0.0
    assert features.loc[0, "bbox"] == (0, 1, 1, 3)
    assert "mean_intensity" not in features.columns


def test_extract_cell_features_adds_mean_intensity_when_image_is_provided():
    labels = np.array(
        [
            [0, 1, 1],
            [0, 2, 0],
            [0, 2, 2],
        ],
        dtype=np.int32,
    )
    image = np.array(
        [
            [0, 10, 20],
            [0, 30, 0],
            [0, 40, 50],
        ],
        dtype=np.uint8,
    )

    features = extract_cell_features(labels, image=image)

    assert list(features["cell_id"]) == [1, 2]
    assert list(features["mean_intensity"]) == [15.0, 40.0]


def test_extract_basic_region_features_uses_cell_feature_schema():
    labels = np.array([[0, 1], [2, 2]], dtype=np.int32)

    features = extract_basic_region_features(labels)

    assert list(features["cell_id"]) == [1, 2]
