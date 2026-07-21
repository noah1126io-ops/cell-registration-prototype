import inspect
from pathlib import Path

import numpy as np

from app import show_he_geojson_preparation, show_mask_to_mask_workflow, show_point_registration_workflow
from src.density_flow import (
    density_flow_image_outputs,
    detect_xy_reversal,
    tissue_aware_density_flow_registration,
    warp_affine_image_with_density_flow,
)
from src.pointset_registration import FineWarpResult, cluster_anchor_fine_warp


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


def _image_metadata(height: int, width: int) -> dict:
    return {
        "bounds_um": [0.0, 0.0, float(width), float(height)],
        "output_pixel_size_um": 1.0,
        "output_origin": "upper-left",
        "width": width,
        "height": height,
        "row0_world_y": 0.5,
        "col0_world_x": 0.5,
    }


def _image_fine_result(
    height: int,
    width: int,
    *,
    dx: float,
    dy: float,
    applied: bool,
) -> FineWarpResult:
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=float) + 0.5,
        np.arange(height, dtype=float) + 0.5,
    )
    attempted_x = np.full((height, width), dx, dtype=float)
    attempted_y = np.full((height, width), dy, dtype=float)
    zeros = np.zeros((height, width), dtype=float)
    return FineWarpResult(
        transformed_points=np.array([[0.0, 0.0]]),
        grid_x=grid_x,
        grid_y=grid_y,
        displacement_x=attempted_x if applied else zeros,
        displacement_y=attempted_y if applied else zeros,
        bounds=(0.5, 0.5, width - 0.5, height - 0.5),
        grid_spacing=1.0,
        jacobian_min=1.0,
        jacobian_max=1.0,
        max_displacement=float(np.hypot(dx, dy)),
        n_candidate_pairs=1,
        n_pairs=1 if applied else 0,
        n_filtered_pairs=0 if applied else 1,
        median_pair_distance_before=1.0,
        median_pair_distance_after=0.0 if applied else 1.0,
        success=applied,
        message="test",
        attempted_transformed_points=np.array([[dx, dy]]),
        attempted_displacement_x=attempted_x,
        attempted_displacement_y=attempted_y,
        rejection_reason=None if applied else "test_rejection",
        applied=applied,
    )


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


def test_density_flow_workflow_uses_safety_gated_raster_outputs():
    source = inspect.getsource(show_he_geojson_preparation)

    assert 'density_flow_mode = fine_alignment_method == "tissue-aware density flow"' in source
    assert "density_flow_image_outputs(" in source
    assert 'attempted_warped_he_image = density_flow_images["attempted"]' in source
    assert 'warped_he_image = density_flow_images["final"]' in source


def test_density_flow_zero_field_returns_identical_affine_image():
    image = np.arange(63, dtype=np.uint8).reshape(7, 9)
    zeros = np.zeros_like(image, dtype=float)

    warped = warp_affine_image_with_density_flow(
        image,
        _image_metadata(*image.shape),
        zeros,
        zeros,
        field_bounds=(0.5, 0.5, 8.5, 6.5),
        field_spacing=1.0,
    )

    np.testing.assert_array_equal(warped, image)


def test_density_flow_constant_translation_shifts_image_by_inverse_mapping():
    image = np.arange(63, dtype=float).reshape(7, 9) + 1.0
    field_x = np.full(image.shape, 2.0)
    field_y = np.zeros(image.shape)

    warped = warp_affine_image_with_density_flow(
        image,
        _image_metadata(*image.shape),
        field_x,
        field_y,
        field_bounds=(0.5, 0.5, 8.5, 6.5),
        field_spacing=1.0,
    )

    np.testing.assert_allclose(warped[:, 2:], image[:, :-2])
    np.testing.assert_allclose(warped[:, :2], 0.0)


def test_density_flow_asymmetric_image_keeps_x_and_y_axes_explicit():
    image = np.zeros((7, 9), dtype=np.uint8)
    image[1, 2] = 255
    field_x = np.full(image.shape, 2.0)
    field_y = np.zeros(image.shape)

    warped = warp_affine_image_with_density_flow(
        image,
        _image_metadata(*image.shape),
        field_x,
        field_y,
        field_bounds=(0.5, 0.5, 8.5, 6.5),
        field_spacing=1.0,
    )

    assert warped[1, 4] == 255
    assert warped[3, 2] == 0


def test_rejected_density_flow_keeps_final_image_affine_only():
    image = np.arange(63, dtype=np.uint8).reshape(7, 9)
    result = _image_fine_result(*image.shape, dx=2.0, dy=0.0, applied=False)

    outputs = density_flow_image_outputs(image, _image_metadata(*image.shape), result)

    np.testing.assert_array_equal(outputs["affine"], image)
    np.testing.assert_array_equal(outputs["final"], image)
    assert not np.array_equal(outputs["attempted"], image)


def test_accepted_density_flow_changes_final_image():
    image = np.arange(63, dtype=np.uint8).reshape(7, 9)
    result = _image_fine_result(*image.shape, dx=2.0, dy=0.0, applied=True)

    outputs = density_flow_image_outputs(image, _image_metadata(*image.shape), result)

    assert not np.array_equal(outputs["final"], image)
    np.testing.assert_array_equal(outputs["final"], outputs["attempted"])


def test_density_flow_inverse_mapping_has_no_forward_splat_holes():
    image = np.ones((12, 14), dtype=float)
    field_x = np.full(image.shape, 0.5)
    field_y = np.full(image.shape, 0.25)

    warped = warp_affine_image_with_density_flow(
        image,
        _image_metadata(*image.shape),
        field_x,
        field_y,
        field_bounds=(0.5, 0.5, 13.5, 11.5),
        field_spacing=1.0,
    )

    assert np.all(warped[1:, 1:] > 0.99)


def test_density_flow_has_no_stalign_runtime_or_source_dependency():
    implementation = Path("src/density_flow.py").read_text(encoding="utf-8").lower()
    requirements = Path("requirements.txt").read_text(encoding="utf-8").lower()

    assert "stalign" not in implementation
    assert "stalign" not in requirements
