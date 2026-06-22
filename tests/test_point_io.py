import numpy as np
import pytest

from src.point_io import load_csv_points, load_npy_centers


def test_load_npy_centers_returns_point_table_for_xy_order(tmp_path):
    path = tmp_path / "centers.npy"
    np.save(path, np.array([[10.0, 20.0], [30.5, 40.5]], dtype=np.float32))

    points = load_npy_centers(path, point_source="stardist_npy", coordinate_order="xy")

    assert list(points.columns) == ["point_id", "centroid_x", "centroid_y", "source"]
    assert list(points["point_id"]) == [1, 2]
    assert list(points["centroid_x"]) == [10.0, 30.5]
    assert list(points["centroid_y"]) == [20.0, 40.5]
    assert list(points["source"]) == ["stardist_npy", "stardist_npy"]


def test_load_npy_centers_returns_point_table_for_yx_order(tmp_path):
    path = tmp_path / "centers_yx.npy"
    np.save(path, np.array([[20.0, 10.0], [40.5, 30.5]], dtype=np.float32))

    points = load_npy_centers(path, point_source="stardist_npy", coordinate_order="yx")

    assert list(points["centroid_x"]) == [10.0, 30.5]
    assert list(points["centroid_y"]) == [20.0, 40.5]


@pytest.mark.parametrize(
    "centers",
    [
        np.empty((0, 2), dtype=np.float32),
        np.array([[np.nan, 1.0]], dtype=np.float32),
        np.array([[np.inf, 1.0]], dtype=np.float32),
        np.array([["x", "y"]]),
    ],
)
def test_load_npy_centers_rejects_invalid_arrays(tmp_path, centers):
    path = tmp_path / "invalid.npy"
    np.save(path, centers)

    with pytest.raises(ValueError):
        load_npy_centers(path)


def test_load_csv_points_accepts_xy_columns(tmp_path):
    path = tmp_path / "points_xy.csv"
    path.write_text("x,y\n10,20\n30.5,40.5\n", encoding="utf-8")

    points = load_csv_points(path, point_source="csv_points")

    assert list(points.columns) == ["point_id", "centroid_x", "centroid_y", "source"]
    assert list(points["point_id"]) == [1, 2]
    assert list(points["centroid_x"]) == [10.0, 30.5]
    assert list(points["centroid_y"]) == [20.0, 40.5]


def test_load_csv_points_accepts_centroid_columns_with_point_id(tmp_path):
    path = tmp_path / "points_centroid.csv"
    path.write_text("point_id,centroid_x,centroid_y\n101,10,20\n102,30.5,40.5\n", encoding="utf-8")

    points = load_csv_points(path, point_source="csv_points")

    assert list(points["point_id"]) == [101, 102]
    assert list(points["centroid_x"]) == [10.0, 30.5]
    assert list(points["centroid_y"]) == [20.0, 40.5]
