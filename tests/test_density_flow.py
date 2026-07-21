import inspect
from pathlib import Path

import numpy as np

from app import show_he_geojson_preparation, show_mask_to_mask_workflow, show_point_registration_workflow
from src.density_flow import detect_xy_reversal, tissue_aware_density_flow_registration
from src.pointset_registration import cluster_anchor_fine_warp


def _grid_points(step: float = 10.0) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(10.0, 71.0, step), np.arange(10.0, 71.0, step))
    return np.column_stack([xs.ravel(), ys.ravel()])


def _run(fixed: np.ndarray, moving: np.ndarray, **kwargs):
    parameters = {
        "density_pixel_size": 2.0,
        "iterations_per_level": 4,
        "max_displacement": 20.0,
        "displacement_p95_limit": 20.0,
        "detect_axis_reversal": False,
    }
    parameters.update(kwargs)
    return tissue_aware_density_flow_registration(fixed, moving, **parameters)


def test_density_flow_identity_transformation():
    points = _grid_points()
    result = _run(points, points)

    assert result.success is True
    assert result.applied is True
    np.testing.assert_allclose(result.transformed_points, points, atol=1e-8)
    np.testing.assert_allclose(result.attempted_displacement_x, 0.0, atol=1e-8)
    assert result.jacobian_min == 1.0
    assert result.jacobian_max == 1.0


def test_density_flow_recovers_known_translation():
    fixed = _grid_points()
    moving = fixed + np.array([6.0, -4.0])
    result = _run(fixed, moving)

    assert result.success is True
    assert result.metrics["attempted"]["symmetric_median_distance"] < 0.05
    point_errors = np.linalg.norm(result.transformed_points - fixed, axis=1)
    assert float(np.median(point_errors)) < 0.05
    assert float(np.max(point_errors)) < 0.3
    assert result.jacobian_min > 0.0


def test_density_flow_improves_known_smooth_deformation():
    xs, ys = np.meshgrid(np.arange(10.0, 91.0, 8.0), np.arange(10.0, 91.0, 8.0))
    fixed = np.column_stack([xs.ravel(), ys.ravel()])
    deformation = np.column_stack(
        [2.0 * np.sin(fixed[:, 1] / 25.0), 1.5 * np.cos(fixed[:, 0] / 30.0)]
    )
    moving = fixed - deformation
    result = _run(
        fixed,
        moving,
        iterations_per_level=8,
        max_displacement=15.0,
        displacement_p95_limit=15.0,
    )

    assert result.success is True
    assert (
        result.metrics["attempted"]["symmetric_median_distance"]
        < result.metrics["before"]["symmetric_median_distance"]
    )
    assert result.jacobian_min > 0.0
    assert result.jacobian_max < 4.0


def test_density_flow_supports_unequal_point_counts():
    rng = np.random.default_rng(2)
    fixed = rng.uniform(10.0, 90.0, size=(100, 2))
    moving = fixed[:70] + np.array([4.0, -3.0])
    result = _run(fixed, moving)

    assert result.success is True
    assert result.transformed_points.shape == moving.shape
    assert np.isfinite(result.transformed_points).all()
    assert result.metrics["attempted"]["symmetric_median_distance"] < result.metrics["before"]["symmetric_median_distance"]


def test_density_flow_reports_points_outside_missing_tissue_region():
    fixed = _grid_points()
    moving_inside = fixed + np.array([2.0, -1.0])
    moving_outside = np.array([[180.0, 180.0], [190.0, 185.0], [185.0, 195.0]])
    moving = np.vstack([moving_inside, moving_outside])
    result = _run(fixed, moving, density_pixel_size=4.0)

    safety = result.metrics["safety"]
    assert safety["points_outside_tissue_before_fraction"] > 0.0
    assert safety["points_outside_tissue_attempted_fraction"] >= 0.0
    assert np.isfinite(result.attempted_transformed_points).all()


def test_density_flow_detects_xy_reversal():
    x = np.linspace(0.0, 100.0, 30)
    fixed = np.column_stack([x, 10.0 * np.sin(x / 20.0)])
    reversed_moving = fixed[:, ::-1]

    diagnostic = detect_xy_reversal(fixed, reversed_moving)
    result = tissue_aware_density_flow_registration(
        fixed,
        reversed_moving,
        density_pixel_size=4.0,
        iterations_per_level=2,
        max_displacement=30.0,
        displacement_p95_limit=30.0,
        detect_axis_reversal=True,
    )

    assert diagnostic["detected"] is True
    assert result.success is False
    assert result.rejection_reason == "possible_xy_reversal"
    np.testing.assert_allclose(result.transformed_points, reversed_moving)


def test_density_flow_finite_output_and_positive_jacobian():
    fixed = _grid_points()
    moving = fixed + np.array([3.0, 2.0])
    result = _run(fixed, moving)

    assert result.metrics["safety"]["finite_output"] is True
    assert np.isfinite(result.attempted_displacement_x).all()
    assert np.isfinite(result.attempted_displacement_y).all()
    assert result.jacobian_min > 0.0


def test_density_flow_rejection_falls_back_to_affine_points():
    fixed = _grid_points()
    moving = fixed + np.array([6.0, -4.0])
    result = _run(
        fixed,
        moving,
        max_displacement=2.0,
        displacement_p95_limit=None,
    )

    assert result.success is False
    assert result.applied is False
    assert result.rejection_reason == "max_displacement_too_large"
    np.testing.assert_allclose(result.transformed_points, moving)
    assert not np.allclose(result.attempted_transformed_points, moving)
    np.testing.assert_allclose(result.displacement_x, 0.0)
    np.testing.assert_allclose(result.displacement_y, 0.0)


def test_density_flow_does_not_regress_cluster_anchor():
    fixed = _grid_points()
    moving = fixed + np.array([4.0, -2.0])
    result = cluster_anchor_fine_warp(
        fixed,
        moving,
        bounds=(0.0, 0.0, 80.0, 80.0),
        grid_spacing=20.0,
        patch_radius=18.0,
        search_radius=8.0,
        search_step=2.0,
        min_points_per_cluster=3,
        match_threshold=4.0,
        min_improvement=0.5,
        max_shift=10.0,
        min_accepted_anchors=3,
        smoothing=0.1,
        neighbors=0,
    )

    assert result.success is True
    assert result.metrics["attempted"]["median_distance"] < result.metrics["before"]["median_distance"]


def test_density_flow_is_not_added_to_workflow_a_or_b():
    assert "tissue_aware_density_flow_registration" not in inspect.getsource(show_point_registration_workflow)
    assert "tissue_aware_density_flow_registration" not in inspect.getsource(show_mask_to_mask_workflow)


def test_density_flow_first_milestone_keeps_he_raster_affine_only():
    source = inspect.getsource(show_he_geojson_preparation)

    assert 'density_flow_points_only = fine_alignment_method == "tissue-aware density flow"' in source
    assert "image_fine_applied = fine_applied and not density_flow_points_only" in source
    assert "and not density_flow_points_only" in source


def test_density_flow_has_no_stalign_runtime_or_source_dependency():
    implementation = Path("src/density_flow.py").read_text(encoding="utf-8").lower()
    requirements = Path("requirements.txt").read_text(encoding="utf-8").lower()

    assert "stalign" not in implementation
    assert "stalign" not in requirements
