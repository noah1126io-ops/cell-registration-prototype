import numpy as np
import pytest

from src.density_flow import (
    _checkpoint_improves_affine,
    _sample_field,
    density_flow_image_outputs,
    joint_density_tissue_structure_registration,
    warp_affine_image_with_density_flow,
)
from src.raster_deformation_qc import (
    checkerboard_comparison,
    local_region_metrics,
    point_displacement_pixel_summary,
    raster_difference_metrics,
    soft_jacobian_log_penalty,
    tissue_support_mismatch,
)
from tests.test_density_flow import _image_fine_result, _image_metadata


MARKERS_RC = np.array([[12, 17], [22, 73], [48, 31], [69, 91], [78, 52]])


def _marker_image(shape=(96, 112)):
    image = np.zeros(shape, dtype=float)
    for index, (row, col) in enumerate(MARKERS_RC):
        image[row, col] = 100.0 + 20.0 * index
    return image


def _marker_world_points(metadata):
    pixel_size = metadata["output_pixel_size_um"]
    if metadata["output_origin"] == "upper-right":
        x = metadata["col0_world_x"] - MARKERS_RC[:, 1] * pixel_size
    else:
        x = metadata["col0_world_x"] + MARKERS_RC[:, 1] * pixel_size
    if metadata["output_origin"] in {"upper-left", "upper-right"}:
        y = metadata["row0_world_y"] + MARKERS_RC[:, 0] * pixel_size
    else:
        y = metadata["row0_world_y"] - MARKERS_RC[:, 0] * pixel_size
    return np.column_stack([x, y])


def _world_to_pixels(points, metadata):
    pixel_size = metadata["output_pixel_size_um"]
    if metadata["output_origin"] == "upper-right":
        cols = (metadata["col0_world_x"] - points[:, 0]) / pixel_size
    else:
        cols = (points[:, 0] - metadata["col0_world_x"]) / pixel_size
    if metadata["output_origin"] in {"upper-left", "upper-right"}:
        rows = (points[:, 1] - metadata["row0_world_y"]) / pixel_size
    else:
        rows = (metadata["row0_world_y"] - points[:, 1]) / pixel_size
    return np.column_stack([rows, cols])


def _detected_centroids(image, expected_pixels, radius=4):
    detected = []
    for expected_row, expected_col in expected_pixels:
        row0 = max(0, int(np.floor(expected_row)) - radius)
        row1 = min(image.shape[0], int(np.ceil(expected_row)) + radius + 1)
        col0 = max(0, int(np.floor(expected_col)) - radius)
        col1 = min(image.shape[1], int(np.ceil(expected_col)) + radius + 1)
        patch = np.maximum(image[row0:row1, col0:col1], 0.0)
        rows, cols = np.indices(patch.shape, dtype=float)
        total = patch.sum()
        assert total > 0
        detected.append([row0 + np.sum(rows * patch) / total, col0 + np.sum(cols * patch) / total])
    return np.asarray(detected)


def _field(kind, shape):
    rows, cols = np.indices(shape, dtype=float)
    if kind == "zero":
        return np.zeros(shape), np.zeros(shape)
    if kind == "translation":
        return np.full(shape, 1.5), np.full(shape, -0.75)
    if kind == "affine":
        return 0.015 * (cols - 55.0), -0.012 * (rows - 47.0)
    if kind == "sinusoidal":
        return 0.8 * np.sin(rows / 17.0), 0.6 * np.cos(cols / 19.0)
    if kind == "asymmetric":
        return 0.5 + 0.7 * np.sin(rows / 13.0), -0.3 + 0.4 * np.cos(cols / 11.0)
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["zero", "translation", "affine", "sinusoidal", "asymmetric"])
def test_transformed_points_align_with_warped_image_markers(kind):
    image = _marker_image()
    metadata = _image_metadata(*image.shape)
    field_x, field_y = _field(kind, image.shape)
    points = _marker_world_points(metadata)
    expected_world = points + _sample_field(
        points, field_x, field_y, (0.5, 0.5, 111.5, 95.5), 1.0, mode="nearest"
    )
    expected_pixels = _world_to_pixels(expected_world, metadata)

    warped, diagnostics = warp_affine_image_with_density_flow(
        image,
        metadata,
        field_x,
        field_y,
        field_bounds=(0.5, 0.5, 111.5, 95.5),
        field_spacing=1.0,
        inverse_iterations=30,
        inverse_convergence_tolerance_pixels=0.01,
        return_diagnostics=True,
    )
    detected = _detected_centroids(warped, expected_pixels)

    assert diagnostics["converged"] is True
    assert float(np.max(np.linalg.norm(detected - expected_pixels, axis=1))) < 0.8


def test_wrong_sign_and_reversed_axes_fail_marker_consistency():
    image = _marker_image()
    metadata = _image_metadata(*image.shape)
    field_x, field_y = _field("asymmetric", image.shape)
    points = _marker_world_points(metadata)
    displacement = _sample_field(points, field_x, field_y, (0.5, 0.5, 111.5, 95.5), 1.0, mode="nearest")
    warped = warp_affine_image_with_density_flow(
        image, metadata, field_x, field_y,
        field_bounds=(0.5, 0.5, 111.5, 95.5), field_spacing=1.0,
    )
    correct_pixels = _world_to_pixels(points + displacement, metadata)
    detected = _detected_centroids(warped, correct_pixels)
    wrong_sign = _world_to_pixels(points - displacement, metadata)
    reversed_axes = _world_to_pixels(points + displacement[:, ::-1], metadata)

    correct_error = np.median(np.linalg.norm(detected - correct_pixels, axis=1))
    assert np.median(np.linalg.norm(detected - wrong_sign, axis=1)) > correct_error + 0.3
    assert np.median(np.linalg.norm(detected - reversed_axes, axis=1)) > correct_error + 0.2


def test_lower_left_origin_marker_consistency():
    image = _marker_image()
    metadata = _image_metadata(*image.shape)
    metadata.update({"output_origin": "lower-left", "row0_world_y": image.shape[0] - 0.5})
    field_x, field_y = _field("asymmetric", image.shape)
    points = _marker_world_points(metadata)
    expected = points + _sample_field(points, field_x, field_y, (0.5, 0.5, 111.5, 95.5), 1.0, mode="nearest")
    expected_pixels = _world_to_pixels(expected, metadata)
    warped = warp_affine_image_with_density_flow(
        image, metadata, field_x, field_y,
        field_bounds=(0.5, 0.5, 111.5, 95.5), field_spacing=1.0,
    )

    detected = _detected_centroids(warped, expected_pixels)
    assert float(np.max(np.linalg.norm(detected - expected_pixels, axis=1))) < 0.8


def test_field_spacing_or_bounds_mismatch_is_rejected():
    image = _marker_image()
    zeros = np.zeros_like(image)
    with pytest.raises(ValueError, match="field shape"):
        warp_affine_image_with_density_flow(
            image, _image_metadata(*image.shape), zeros, zeros,
            field_bounds=(0.5, 0.5, 111.5, 95.5), field_spacing=2.0,
        )
    with pytest.raises(ValueError, match="field shape"):
        warp_affine_image_with_density_flow(
            image, _image_metadata(*image.shape), zeros, zeros,
            field_bounds=(0.5, 0.5, 100.5, 90.5), field_spacing=1.0,
        )


def test_inverse_residual_converges_for_safe_smooth_field():
    image = _marker_image()
    field_x, field_y = _field("sinusoidal", image.shape)
    _, diagnostics = warp_affine_image_with_density_flow(
        image, _image_metadata(*image.shape), field_x, field_y,
        field_bounds=(0.5, 0.5, 111.5, 95.5), field_spacing=1.0,
        inverse_iterations=30, inverse_convergence_tolerance_pixels=0.01,
        return_diagnostics=True,
    )

    assert diagnostics["converged"] is True
    assert diagnostics["max_residual_pixels"] <= 0.01
    assert diagnostics["iterations_used"] <= 30
    assert diagnostics["history"][-1]["max_residual_pixels"] <= diagnostics["history"][0]["max_residual_pixels"]


def test_accepted_and_rejected_result_raster_gating():
    image = _marker_image((96, 112))
    accepted = _image_fine_result(*image.shape, dx=0.5, dy=0.0, applied=True)
    rejected = _image_fine_result(*image.shape, dx=0.5, dy=0.0, applied=False)

    accepted_outputs = density_flow_image_outputs(image, _image_metadata(*image.shape), accepted)
    rejected_outputs = density_flow_image_outputs(image, _image_metadata(*image.shape), rejected)

    assert accepted_outputs["raster_applied"] is True
    assert not np.array_equal(accepted_outputs["final"], image)
    assert rejected_outputs["raster_applied"] is False
    np.testing.assert_array_equal(rejected_outputs["final"], image)
    assert not np.array_equal(rejected_outputs["attempted"], image)


def test_subpixel_field_is_reported_as_visually_subtle():
    summary = point_displacement_pixel_summary(
        np.full((10, 12), 0.55), np.zeros((10, 12)), output_pixel_size_um=1.0
    )

    assert summary["visually_subpixel"] is True
    assert summary["p95_output_pixels"] == pytest.approx(0.55)


def test_qc_gain_does_not_change_true_outputs_or_metrics():
    image = _marker_image()
    field_x, field_y = _field("sinusoidal", image.shape)
    metadata = _image_metadata(*image.shape)
    true_warp = warp_affine_image_with_density_flow(
        image, metadata, field_x, field_y,
        field_bounds=(0.5, 0.5, 111.5, 95.5), field_spacing=1.0,
    )
    true_summary = point_displacement_pixel_summary(field_x, field_y, output_pixel_size_um=1.0)
    _ = warp_affine_image_with_density_flow(
        image, metadata, field_x * 5.0, field_y * 5.0,
        field_bounds=(0.5, 0.5, 111.5, 95.5), field_spacing=1.0,
    )

    repeated_true = warp_affine_image_with_density_flow(
        image, metadata, field_x, field_y,
        field_bounds=(0.5, 0.5, 111.5, 95.5), field_spacing=1.0,
    )
    np.testing.assert_array_equal(repeated_true, true_warp)
    assert point_displacement_pixel_summary(field_x, field_y, output_pixel_size_um=1.0) == true_summary


def test_raster_metrics_checkerboard_support_and_soft_jacobian():
    image = np.arange(100, dtype=float).reshape(10, 10)
    warped = np.roll(image, 1, axis=1)
    mask = np.zeros((10, 10), dtype=bool)
    mask[:, 2:8] = True
    metrics = raster_difference_metrics(image, warped, tissue_mask=mask)

    assert metrics["full_image"]["mean_absolute_difference"] > 0
    assert metrics["inside_tissue"]["p95_absolute_difference"] > 0
    assert checkerboard_comparison(image, warped, tile_size=2).shape == image.shape
    assert soft_jacobian_log_penalty(np.ones((4, 4))) == pytest.approx(0.0)
    assert soft_jacobian_log_penalty(np.full((4, 4), 1.1)) > 0.0
    assert tissue_support_mismatch(mask, mask) == 0.0
    assert tissue_support_mismatch(mask, ~mask) > 0.0


def test_local_region_degradation_is_detected():
    fixed = np.array([[10, 10], [20, 10], [10, 20], [20, 20], [110, 10], [120, 10], [110, 20], [120, 20]], dtype=float)
    before = fixed.copy()
    after = fixed.copy()
    after[4:, 0] += 8.0
    zeros = np.zeros((15, 15), dtype=float)
    jacobian = np.ones_like(zeros)

    table, summary = local_region_metrics(
        fixed, before, after, zeros, zeros, jacobian,
        bounds=(0.0, 0.0, 140.0, 140.0), field_spacing=10.0,
        block_size_um=70.0, min_points=3,
    )

    assert not table.empty
    assert summary["fraction_regions_worsened"] > 0
    assert summary["worst_degraded_region"] is not None


def test_actual_he_tissue_mask_changes_joint_support_objective():
    points = np.array(
        [[8, 8], [16, 8], [8, 16], [16, 16], [24, 24], [32, 32]],
        dtype=float,
    )
    image = np.tile(np.arange(40, dtype=float), (40, 1))
    metadata = _image_metadata(40, 40)
    full_mask = np.ones((40, 40), dtype=bool)
    central_mask = np.zeros((40, 40), dtype=bool)
    central_mask[10:30, 10:30] = True
    common = {
        "affine_he_image": image,
        "affine_he_metadata": metadata,
        "bounds": (0.0, 0.0, 40.0, 40.0),
        "density_pixel_size": 2.0,
        "optimization_levels": 1,
        "iterations_per_level": 1,
        "global_translation_initialization": "off",
        "detect_axis_reversal": False,
    }

    full = joint_density_tissue_structure_registration(
        points, points, affine_he_tissue_mask=full_mask, **common
    )
    central = joint_density_tissue_structure_registration(
        points, points, affine_he_tissue_mask=central_mask, **common
    )

    assert full.metrics["density_flow"]["uses_actual_he_tissue_mask"] is True
    assert central.metrics["density_flow"]["uses_actual_he_tissue_mask"] is True
    assert full.metrics["density_flow"]["initial_objective"] != pytest.approx(
        central.metrics["density_flow"]["initial_objective"]
    )


def test_negligible_numerical_improvement_is_not_a_checkpoint():
    affine_metrics = {
        "symmetric_within_3": 0.2,
        "symmetric_within_5": 0.4,
        "symmetric_within_10": 0.8,
    }
    candidate = {
        "symmetric_median_distance": 10.0 - 1e-12,
        "mutual_nearest_fraction": 0.5,
        **affine_metrics,
    }

    assert not _checkpoint_improves_affine(
        candidate,
        affine_median=10.0,
        affine_mutual_fraction=0.5,
        affine_metrics=affine_metrics,
    )
