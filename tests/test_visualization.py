import pandas as pd

from src.visualization import visualize_point_sets


def test_visualize_point_sets_without_image_creates_figure():
    fixed = pd.DataFrame({"cell_id": [1], "centroid_x": [10.0], "centroid_y": [20.0]})
    moving = pd.DataFrame({"cell_id": [2], "centroid_x": [12.0], "centroid_y": [21.0]})

    figure = visualize_point_sets(fixed, moving, title="No image")

    assert figure is not None
    assert len(figure.axes) == 1
