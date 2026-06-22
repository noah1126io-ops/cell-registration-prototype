import pandas as pd

from src.matching import MATCH_COLUMNS, initialize_empty_matches, match_cells


def test_initialize_empty_matches_schema():
    matches = initialize_empty_matches()

    assert list(matches.columns) == MATCH_COLUMNS
    assert matches.empty


def test_match_cells_matches_known_positions():
    fixed = pd.DataFrame(
        {
            "cell_id": [1, 2],
            "centroid_x": [10.0, 100.0],
            "centroid_y": [10.0, 100.0],
            "area": [50.0, 80.0],
            "eccentricity": [0.2, 0.6],
        }
    )
    moving = pd.DataFrame(
        {
            "cell_id": [11, 22],
            "centroid_x": [12.0, 98.0],
            "centroid_y": [11.0, 101.0],
            "area": [55.0, 72.0],
            "eccentricity": [0.25, 0.55],
        }
    )

    matches = match_cells(
        fixed,
        moving,
        max_distance=10.0,
        min_area_ratio=0.5,
        max_area_ratio=2.0,
        max_score=1.0,
    )

    assert list(matches["fixed_cell_id"]) == [1, 2]
    assert list(matches["moving_cell_id"]) == [11, 22]
    assert list(matches["matched_status"]) == ["matched", "matched"]
    assert (matches["confidence"] > 0).all()


def test_match_cells_marks_threshold_exceeding_cell_unmatched():
    fixed = pd.DataFrame(
        {
            "cell_id": [1],
            "centroid_x": [10.0],
            "centroid_y": [10.0],
            "area": [50.0],
            "eccentricity": [0.2],
        }
    )
    moving = pd.DataFrame(
        {
            "cell_id": [11],
            "centroid_x": [100.0],
            "centroid_y": [100.0],
            "area": [50.0],
            "eccentricity": [0.2],
        }
    )

    matches = match_cells(
        fixed,
        moving,
        max_distance=20.0,
        min_area_ratio=0.5,
        max_area_ratio=2.0,
        max_score=1.0,
    )

    assert matches.loc[0, "fixed_cell_id"] == 1
    assert pd.isna(matches.loc[0, "moving_cell_id"])
    assert matches.loc[0, "matched_status"] == "unmatched_fixed"
    assert matches.loc[0, "confidence"] == 0.0


def test_match_cells_marks_high_score_assignment_low_confidence():
    fixed = pd.DataFrame(
        {
            "cell_id": [1],
            "centroid_x": [10.0],
            "centroid_y": [10.0],
            "area": [50.0],
            "eccentricity": [0.1],
        }
    )
    moving = pd.DataFrame(
        {
            "cell_id": [11],
            "centroid_x": [12.0],
            "centroid_y": [10.0],
            "area": [75.0],
            "eccentricity": [0.8],
        }
    )

    matches = match_cells(
        fixed,
        moving,
        max_distance=10.0,
        min_area_ratio=0.5,
        max_area_ratio=2.0,
        max_score=0.1,
    )

    assert matches.loc[0, "moving_cell_id"] == 11
    assert matches.loc[0, "matched_status"] == "low_confidence"
    assert matches.loc[0, "score"] > 0.1
    assert 0.0 < matches.loc[0, "confidence"] < 1.0


def test_match_cells_uses_position_only_when_area_and_shape_are_missing():
    fixed = pd.DataFrame(
        {
            "cell_id": [1],
            "centroid_x": [10.0],
            "centroid_y": [10.0],
        }
    )
    moving = pd.DataFrame(
        {
            "cell_id": [11],
            "centroid_x": [12.0],
            "centroid_y": [10.0],
        }
    )

    matches = match_cells(
        fixed,
        moving,
        max_distance=10.0,
        min_area_ratio=0.5,
        max_area_ratio=2.0,
        max_score=1.0,
    )

    assert matches.loc[0, "moving_cell_id"] == 11
    assert matches.loc[0, "matched_status"] == "matched"
    assert pd.isna(matches.loc[0, "area_ratio"])
