import numpy as np

from src.point_io import load_npy_centers


def test_load_npy_centers_returns_point_table(tmp_path):
    path = tmp_path / "centers.npy"
    np.save(path, np.array([[10.0, 20.0], [30.5, 40.5]], dtype=np.float32))

    points = load_npy_centers(path, point_source="stardist_npy")

    assert list(points.columns) == ["point_id", "centroid_x", "centroid_y", "source"]
    assert list(points["point_id"]) == [1, 2]
    assert list(points["centroid_x"]) == [10.0, 30.5]
    assert list(points["centroid_y"]) == [20.0, 40.5]
    assert list(points["source"]) == ["stardist_npy", "stardist_npy"]
