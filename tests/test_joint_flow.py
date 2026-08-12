import inspect

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from app import show_mask_to_mask_workflow, show_point_registration_workflow
from src.density_flow import joint_density_tissue_structure_registration, tissue_aware_density_flow_registration
from src.joint_flow import (
    build_fixed_nuclear_structure_features,
    build_he_nuclear_structure_features,
)


def _metadata(size: int) -> dict:
    return {
        "bounds_um": [0.0, 0.0, float(size), float(size)],
        "output_pixel_size_um": 1.0,
        "output_origin": "upper-left",
        "width": size,
        "height": size,
        "row0_world_y": 0.5,
        "col0_world_x": 0.5,
    }


def _joint_case(amplitude: float, *, strict_max: float = 12.0, intensity_pattern: bool = False):
    size = 96
    yy, xx = np.indices((size, size), dtype=float)
    fixed = np.array([(x, y) for y in range(16, 81, 8) for x in range(16, 81, 8)], dtype=float)
    local_shift = amplitude * np.exp(
        -((fixed[:, 0] - 48.0) ** 2 + (fixed[:, 1] - 48.0) ** 2) / (2.0 * 22.0**2)
    )
    moving = fixed.copy()
    moving[:, 0] -= local_shift
    fixed_mask = ((xx - 48.0) ** 2 / 38.0**2 + (yy - 48.0) ** 2 / 34.0**2) < 1.0
    source_x = xx + amplitude * np.exp(
        -((xx - 48.0) ** 2 + (yy - 48.0) ** 2) / (2.0 * 22.0**2)
    )
    moving_mask = map_coordinates(
        fixed_mask.astype(float), [yy, source_x], order=1, mode="constant"
    ) >= 0.5
    image = np.full((size, size, 3), 235, dtype=np.uint8)
    image[moving_mask] = [160, 95, 145]
    if intensity_pattern:
        image[..., 0] = np.asarray((xx * 7) % 255, dtype=np.uint8)
        image[..., 1] = np.asarray((yy * 11) % 255, dtype=np.uint8)
        image[..., 2] = np.asarray(((xx + yy) * 5) % 255, dtype=np.uint8)
        image[~moving_mask] = 240
    else:
        for x, y in moving.astype(int):
            image[max(y - 1, 0): y + 2, max(x - 1, 0): x + 2] = [70, 35, 90]
    result = joint_density_tissue_structure_registration(
        fixed,
        moving,
        affine_he_image=image,
        affine_he_tissue_mask=moving_mask,
        affine_he_metadata=_metadata(size),
        bounds=(0.0, 0.0, float(size), float(size)),
        density_pixel_size=2.0,
        stage_a_scales_um=(16.0, 8.0),
        stage_b_scales_um=(8.0, 4.0),
        stage_a_iterations=6,
        stage_b_iterations=6,
        stage_a_learning_rate=0.14,
        stage_b_learning_rate=0.08,
        support_weight=1.2,
        structure_weight=0.35,
        detect_axis_reversal=False,
        max_displacement=strict_max,
        displacement_p95_limit=min(strict_max, 10.0),
        minimum_absolute_median_improvement=0.01,
        minimum_relative_median_improvement=0.001,
        minimum_jacobian_p05=0.5,
        maximum_jacobian_p95=1.5,
    )
    return fixed, moving, result


def test_joint_structure_features_are_modality_compatible_and_normalized():
    size = 48
    yy, xx = np.indices((size, size))
    points = np.array([[12, 12], [24, 12], [36, 12], [12, 24], [24, 24], [36, 24]], dtype=float)
    mask = ((xx - 24) ** 2 + (yy - 24) ** 2) < 20**2
    image = np.full((size, size, 3), 235, dtype=np.uint8)
    image[mask] = [150, 80, 145]
    fixed = build_fixed_nuclear_structure_features(
        points, shape=(25, 25), bounds=(0.0, 0.0, 48.0, 48.0), pixel_size_um=2.0
    )
    moving = build_he_nuclear_structure_features(image, mask, pixel_size_um=1.0)

    assert fixed["mode"] == "geojson_nuclear_geometry"
    assert moving["mode"] in {"hematoxylin_rgb2hed", "normalized_grayscale_fallback"}
    for features in (fixed, moving):
        for key in ("density", "gradient", "curvature", "nuclear_structure"):
            values = np.asarray(features[key])
            assert np.isfinite(values).all()
            assert values.min() >= 0.0 and values.max() <= 1.0
        signed = np.asarray(features["signed_distance"])
        assert np.isfinite(signed).all()
        assert signed.min() >= -1.0 and signed.max() <= 1.0


def test_joint_no_deformation_stays_near_zero_and_fixed_points_never_move():
    fixed, _, result = _joint_case(0.0)
    magnitude = np.hypot(result.attempted_displacement_x, result.attempted_displacement_y)

    assert np.percentile(magnitude, 95) < 0.2
    assert result.metrics["joint_flow"]["fixed_points_moved"] is False
    np.testing.assert_array_equal(fixed, fixed.copy())


@pytest.mark.parametrize("amplitude, expected_sign", [(4.0, 1.0), (-4.0, -1.0)])
def test_joint_recovers_smooth_bulge_or_compression_direction_without_foldover(amplitude, expected_sign):
    _, _, result = _joint_case(amplitude)
    center_row = result.attempted_displacement_x.shape[0] // 2
    center_col = result.attempted_displacement_x.shape[1] // 2
    center_dx = result.attempted_displacement_x[center_row, center_col]

    assert np.sign(center_dx) == expected_sign
    assert 1.0 < result.max_displacement < 6.0
    assert result.jacobian_min > 0.0
    assert result.attempted_metrics["symmetric_median_distance"] < result.median_pair_distance_before


def test_joint_density_only_still_improves_alignment():
    fixed, moving, _ = _joint_case(3.0)
    size = 96
    image = np.full((size, size), 128, dtype=np.uint8)
    mask = np.ones((size, size), dtype=bool)
    result = joint_density_tissue_structure_registration(
        fixed, moving,
        affine_he_image=image, affine_he_tissue_mask=mask, affine_he_metadata=_metadata(size),
        bounds=(0.0, 0.0, 96.0, 96.0), density_pixel_size=2.0,
        stage_a_scales_um=(16.0, 8.0), stage_b_scales_um=(8.0, 4.0),
        stage_a_iterations=4, stage_b_iterations=5,
        support_weight=0.0, structure_weight=0.0,
        detect_axis_reversal=False, max_displacement=12.0, displacement_p95_limit=10.0,
        minimum_absolute_median_improvement=0.01, minimum_relative_median_improvement=0.001,
        minimum_jacobian_p05=0.5, maximum_jacobian_p95=1.5,
    )

    assert result.attempted_metrics["symmetric_median_distance"] < result.median_pair_distance_before


def test_misleading_he_intensity_without_geometry_does_not_drive_large_warp():
    _, _, result = _joint_case(0.0, intensity_pattern=True)
    p95 = np.percentile(np.hypot(result.attempted_displacement_x, result.attempted_displacement_y), 95)

    assert p95 < 0.5


def test_unsafe_strong_attempt_is_retained_but_application_falls_back_to_affine():
    _, moving, result = _joint_case(4.0, strict_max=0.25)

    assert result.applied is False
    assert result.rejection_reason == "no_joint_checkpoint_passed_final_application_safety"
    assert result.max_displacement > 0.25
    np.testing.assert_allclose(result.transformed_points, moving)
    assert not np.allclose(result.attempted_transformed_points, moving)


def test_workflow_a_b_and_density_only_implementation_remain_separate():
    assert "joint_density_tissue_structure_registration" not in inspect.getsource(show_point_registration_workflow)
    assert "joint_density_tissue_structure_registration" not in inspect.getsource(show_mask_to_mask_workflow)
    signature = inspect.signature(tissue_aware_density_flow_registration)
    assert signature.parameters["density_channel_weight"].default == 1.0
    assert signature.parameters["tissue_support_channel_weight"].default == 0.0
    assert signature.parameters["structure_channel_weight"].default == 0.0
