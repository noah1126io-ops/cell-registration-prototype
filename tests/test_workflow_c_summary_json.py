import io
import json
from pathlib import Path

import numpy as np

from app import _displacement_field_npz_bytes, _he_geojson_summary_to_json, _json_summary_safe
from src.pointset_registration import AffineICPResult, FineWarpResult


def _results(metrics):
    affine = AffineICPResult(
        affine_matrix=np.eye(2),
        translation=np.array([1.0, -2.0]),
        transformed_points=np.array([[2.0, 3.0]]),
        flip_x=False,
        flip_y=True,
        image_width=np.float64(100.0),
        image_height=np.float32(80.0),
        mean_residual=np.float32(1.25),
        median_residual=np.float64(1.0),
        n_pairs=np.int64(7),
        success=True,
        message="ok",
    )
    zeros = np.zeros((2, 3), dtype=np.float32)
    fine = FineWarpResult(
        transformed_points=np.array([[2.0, 3.0]]),
        grid_x=np.zeros((2, 3)),
        grid_y=np.zeros((2, 3)),
        displacement_x=zeros,
        displacement_y=zeros,
        bounds=(0.0, 0.0, 3.0, 2.0),
        grid_spacing=np.float32(1.0),
        jacobian_min=np.float64(0.9),
        jacobian_max=np.float32(1.1),
        max_displacement=np.float64(2.0),
        n_candidate_pairs=np.int32(9),
        n_pairs=np.int64(7),
        n_filtered_pairs=np.int32(2),
        median_pair_distance_before=np.float64(3.0),
        median_pair_distance_after=np.float32(2.0),
        success=True,
        message="applied",
        attempted_metrics={"median": np.float32(2.1)},
        applied_metrics={"median": np.float64(2.0)},
        applied=True,
        metrics=metrics,
    )
    return affine, fine


def test_joint_flow_arrays_are_compact_metadata_and_scalars_are_preserved():
    displacement = np.arange(12, dtype=np.float32).reshape(3, 4)
    metrics = {
        "joint_flow": {
            "stage_a": {"objective": np.float32(1.5), "iterations": np.int32(4)},
            "stage_b": {"objective": np.float64(1.0)},
            "final": {"jacobian_min": np.float64(0.8)},
            "objective_history": [{"iteration": np.int64(1), "objective": np.float32(2.0)}],
            "selected_checkpoint": np.int32(1),
            "fixed_points_moved": False,
            "stage_a_displacement_x": displacement,
            "stage_a_displacement_y": -displacement,
            "stage_b_incremental_x": displacement / 2,
            "stage_b_incremental_y": displacement / 3,
        }
    }
    affine, fine = _results(metrics)

    decoded = json.loads(_he_geojson_summary_to_json(
        affine,
        fine,
        parameters={"weight": np.float32(0.2), "output": Path("result")},
        warp_metadata={"pixel_size": np.float64(2.0)},
    ))
    joint = decoded["fine_center_snap"]["distance_metrics"]["joint_flow"]

    assert joint["stage_a"] == {"objective": 1.5, "iterations": 4}
    assert joint["stage_b"]["objective"] == 1.0
    assert joint["final"]["jacobian_min"] == 0.8
    assert joint["objective_history"][0]["iteration"] == 1
    assert joint["selected_checkpoint"] == 1
    assert joint["fixed_points_moved"] is False
    assert joint["stage_a_displacement_x"] == {
        "type": "ndarray",
        "shape": [3, 4],
        "dtype": "float32",
        "omitted_from_summary_json": True,
    }
    for field_name in (
        "stage_a_displacement_x",
        "stage_a_displacement_y",
        "stage_b_incremental_x",
        "stage_b_incremental_y",
    ):
        assert isinstance(joint[field_name], dict)
        assert joint[field_name]["omitted_from_summary_json"] is True
        assert set(joint[field_name]) == {
            "type", "shape", "dtype", "omitted_from_summary_json"
        }


def test_summary_sanitizer_recurses_and_converts_nonfinite_values_to_null():
    safe = _json_summary_safe({
        "values": [np.float32(1.25), np.float64(np.nan), np.float32(np.inf)],
        "integers": (np.int32(3), np.int64(4)),
        "nested": {"path": Path("output/file.json")},
    })
    encoded = json.dumps(safe, allow_nan=False)
    decoded = json.loads(encoded)

    assert decoded["values"] == [1.25, None, None]
    assert decoded["integers"] == [3, 4]
    assert decoded["nested"]["path"] == str(Path("output/file.json"))


def test_displacement_npz_export_still_contains_full_array_values():
    displacement_x = np.arange(12, dtype=np.float32).reshape(3, 4)
    displacement_y = -displacement_x
    payload = _displacement_field_npz_bytes(
        displacement_x,
        displacement_y,
        bounds=(0.0, 0.0, 4.0, 3.0),
        grid_spacing=1.0,
    )

    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["displacement_x"], displacement_x)
        np.testing.assert_array_equal(archive["displacement_y"], displacement_y)
        np.testing.assert_array_equal(archive["bounds"], [0.0, 0.0, 4.0, 3.0])
        assert archive["grid_spacing"].item() == 1.0
