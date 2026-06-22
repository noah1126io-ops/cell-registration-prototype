import json

import numpy as np

from src.geojson_utils import load_geojson_centroids, load_geojson_polygons


def test_load_geojson_centroids_uses_properties(tmp_path):
    path = tmp_path / "nuclei.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"segment_id": 7, "centroid_x": 10.5, "centroid_y": 20.5},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    centroids = load_geojson_centroids(path, point_source="fluorescence_geojson")

    assert list(centroids.columns) == ["point_id", "centroid_x", "centroid_y", "source"]
    assert centroids.loc[0, "point_id"] == 7
    assert centroids.loc[0, "centroid_x"] == 10.5
    assert centroids.loc[0, "centroid_y"] == 20.5
    assert centroids.loc[0, "source"] == "fluorescence_geojson"


def test_load_geojson_centroids_falls_back_to_polygon_mean(tmp_path):
    path = tmp_path / "nuclei.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    centroids = load_geojson_centroids(path)

    assert centroids.loc[0, "point_id"] == 1
    assert centroids.loc[0, "centroid_x"] == 1.0
    assert centroids.loc[0, "centroid_y"] == 1.0


def test_load_geojson_polygons_returns_exterior_coordinates(tmp_path):
    path = tmp_path / "nuclei.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"segment_id": "a", "centroid_x": 1.0, "centroid_y": 1.0},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 0]]],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    polygons = load_geojson_polygons(path)

    assert list(polygons.columns) == ["polygon_id", "point_id", "centroid_x", "centroid_y", "coordinates", "source"]
    assert polygons.loc[0, "polygon_id"] == "a:1"
    np.testing.assert_allclose(polygons.loc[0, "coordinates"], np.array([[0, 0], [2, 0], [2, 2], [0, 0]]))
