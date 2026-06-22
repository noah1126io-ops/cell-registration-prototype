from __future__ import annotations

import json
from os import PathLike
from typing import BinaryIO

import numpy as np
import pandas as pd


CENTROID_COLUMNS = ["point_id", "centroid_x", "centroid_y", "source"]
POLYGON_COLUMNS = ["polygon_id", "point_id", "centroid_x", "centroid_y", "coordinates", "source"]


def _read_geojson(source: str | PathLike | BinaryIO) -> dict:
    if hasattr(source, "seek"):
        source.seek(0)
        data = source.read()
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return json.loads(data)

    with open(source, encoding="utf-8") as file:
        return json.load(file)


def _feature_id(feature: dict, fallback: int):
    properties = feature.get("properties", {}) or {}
    for key in ("segment_id", "id", "label", "cell_id", "nucleus_id"):
        if key in properties:
            return properties[key]
    return fallback


def _polygon_exteriors(geometry: dict) -> list[np.ndarray]:
    if not geometry:
        return []

    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])

    if geometry_type == "Polygon":
        if not coordinates:
            return []
        return [np.asarray(coordinates[0], dtype=float)]

    if geometry_type == "MultiPolygon":
        exteriors = []
        for polygon in coordinates:
            if polygon:
                exteriors.append(np.asarray(polygon[0], dtype=float))
        return exteriors

    return []


def _centroid_from_feature(feature: dict) -> tuple[float, float] | None:
    properties = feature.get("properties", {}) or {}
    if "centroid_x" in properties and "centroid_y" in properties:
        return float(properties["centroid_x"]), float(properties["centroid_y"])

    exteriors = _polygon_exteriors(feature.get("geometry", {}) or {})
    if not exteriors:
        return None

    exterior = exteriors[0]
    if len(exterior) > 1 and np.allclose(exterior[0], exterior[-1]):
        exterior = exterior[:-1]
    if exterior.size == 0:
        return None
    centroid = exterior.mean(axis=0)
    return float(centroid[0]), float(centroid[1])


def load_geojson_centroids(source: str | PathLike | BinaryIO, *, point_source: str = "geojson") -> pd.DataFrame:
    """Load nuclei centroids from a GeoJSON FeatureCollection."""
    geojson = _read_geojson(source)
    rows = []

    for index, feature in enumerate(geojson.get("features", []), start=1):
        centroid = _centroid_from_feature(feature)
        if centroid is None:
            continue
        centroid_x, centroid_y = centroid
        rows.append(
            {
                "point_id": _feature_id(feature, index),
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "source": point_source,
            }
        )

    if not rows:
        raise ValueError("No GeoJSON nuclei centroids could be loaded.")

    centroids = pd.DataFrame(rows, columns=CENTROID_COLUMNS)
    if not np.isfinite(centroids[["centroid_x", "centroid_y"]].to_numpy(dtype=float)).all():
        raise ValueError("GeoJSON centroids must not contain NaN or infinite coordinates.")
    return centroids


def load_geojson_polygons(source: str | PathLike | BinaryIO, *, polygon_source: str = "geojson") -> pd.DataFrame:
    """Load nuclei polygon exterior rings from a GeoJSON FeatureCollection."""
    geojson = _read_geojson(source)
    rows = []

    for feature_index, feature in enumerate(geojson.get("features", []), start=1):
        point_id = _feature_id(feature, feature_index)
        centroid = _centroid_from_feature(feature)
        centroid_x, centroid_y = centroid if centroid is not None else (np.nan, np.nan)

        for polygon_index, exterior in enumerate(_polygon_exteriors(feature.get("geometry", {}) or {}), start=1):
            if exterior.ndim != 2 or exterior.shape[1] != 2:
                raise ValueError("GeoJSON polygon coordinates must have shape (n_vertices, 2).")
            if not np.isfinite(exterior).all():
                raise ValueError("GeoJSON polygon coordinates must not contain NaN or infinite values.")

            rows.append(
                {
                    "polygon_id": f"{point_id}:{polygon_index}",
                    "point_id": point_id,
                    "centroid_x": centroid_x,
                    "centroid_y": centroid_y,
                    "coordinates": exterior,
                    "source": polygon_source,
                }
            )

    if not rows:
        raise ValueError("No GeoJSON nuclei polygons could be loaded.")

    return pd.DataFrame(rows, columns=POLYGON_COLUMNS)


# TODO: Preserve GeoJSON metadata needed for image bounds, scale, and Y-flip handling.
