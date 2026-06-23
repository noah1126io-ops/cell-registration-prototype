import pandas as pd
import numpy as np

from src.visualization import visualize_point_sets, visualize_warped_he_point_overlay


def test_visualize_point_sets_without_image_creates_figure():
    fixed = pd.DataFrame({"cell_id": [1], "centroid_x": [10.0], "centroid_y": [20.0]})
    moving = pd.DataFrame({"cell_id": [2], "centroid_x": [12.0], "centroid_y": [21.0]})

    figure = visualize_point_sets(fixed, moving, title="No image")

    assert figure is not None
    assert len(figure.axes) == 1


def test_visualize_warped_he_point_overlay_creates_figure():
    image = np.zeros((10, 10), dtype=np.uint8)
    geojson_pixels = np.array([[2.0, 3.0]])
    he_pixels = np.array([[2.5, 3.5]])

    figure = visualize_warped_he_point_overlay(
        image,
        geojson_pixels,
        he_pixels,
        title="Overlay",
    )

    assert figure is not None
    assert len(figure.axes) == 1
