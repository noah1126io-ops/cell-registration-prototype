import numpy as np

from src.synthetic_registration_benchmark import (
    evaluate_synthetic_displacement,
    generate_synthetic_registration_sample,
    ground_truth_displacement,
    marker_point_raster_consistency,
    synthetic_same_modality_raster_metrics,
)


def _grid(size=65):
    axis = np.arange(size, dtype=float)
    return np.meshgrid(axis, axis)


def test_identity_field_is_zero_and_bulge_varies_spatially():
    grid_x, grid_y = _grid()
    identity_x, identity_y = ground_truth_displacement(grid_x, grid_y, "identity", 10.0)
    bulge_x, bulge_y = ground_truth_displacement(grid_x, grid_y, "Gaussian bulge", 5.0)

    assert np.count_nonzero(identity_x) == 0
    assert np.count_nonzero(identity_y) == 0
    assert np.std(bulge_x) > 0
    assert np.std(bulge_y) > 0


def test_compression_has_opposite_direction_to_bulge():
    grid_x, grid_y = _grid()
    bulge = ground_truth_displacement(grid_x, grid_y, "Gaussian bulge", 5.0)
    compression = ground_truth_displacement(grid_x, grid_y, "Gaussian compression", 5.0)

    assert np.allclose(compression[0], -bulge[0])
    assert np.allclose(compression[1], -bulge[1])


def test_synthetic_seed_is_reproducible():
    first = generate_synthetic_registration_sample(seed=17, size=64, n_points=80)
    second = generate_synthetic_registration_sample(seed=17, size=64, n_points=80)

    assert np.array_equal(first.fixed_points, second.fixed_points)
    assert np.array_equal(first.moving_points, second.moving_points)
    assert np.array_equal(first.moving_image, second.moving_image)


def test_points_and_landmarks_follow_same_forward_field_direction():
    sample = generate_synthetic_registration_sample(
        deformation_type="smooth local translation", amplitude_um=4.0, seed=5, size=96, n_points=100
    )
    from src.registration_evaluation import sample_displacement_field

    point_displacement = sample_displacement_field(
        sample.moving_points,
        sample.ground_truth_displacement_x,
        sample.ground_truth_displacement_y,
        bounds=sample.bounds,
        spacing=sample.spacing,
    )
    landmark_displacement = sample_displacement_field(
        sample.moving_landmarks,
        sample.ground_truth_displacement_x,
        sample.ground_truth_displacement_y,
        bounds=sample.bounds,
        spacing=sample.spacing,
    )

    assert np.median(np.linalg.norm(sample.moving_points + point_displacement - sample.fixed_points, axis=1)) < 0.02
    assert np.median(np.linalg.norm(sample.moving_landmarks + landmark_displacement - sample.fixed_landmarks, axis=1)) < 0.02
    rows, cols = np.indices(sample.ground_truth_warped_mask.shape, dtype=float)
    fixed_tissue_mask = (
        ((cols - 47.5) / (0.39 * 96)) ** 2
        + ((rows - 47.5) / (0.34 * 96)) ** 2
        <= 1
    )
    intersection = np.count_nonzero(sample.ground_truth_warped_mask & fixed_tissue_mask)
    union = np.count_nonzero(sample.ground_truth_warped_mask | fixed_tissue_mask)
    assert intersection / union > 0.98


def test_exact_ground_truth_field_has_zero_epe():
    sample = generate_synthetic_registration_sample(seed=3, size=48, n_points=50)
    _, summary, epe_map = evaluate_synthetic_displacement(
        sample, sample.ground_truth_displacement_x, sample.ground_truth_displacement_y
    )

    assert summary["epe_max_um"] == 0.0
    assert np.count_nonzero(epe_map) == 0


def test_synthetic_image_against_itself_has_perfect_raster_metrics():
    sample = generate_synthetic_registration_sample(seed=9, size=64, n_points=70)
    metrics = synthetic_same_modality_raster_metrics(
        sample.ground_truth_warped_image, sample.ground_truth_warped_image
    )

    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["ssim"] == 1.0


def test_warped_marker_raster_is_consistent_with_transformed_points():
    sample = generate_synthetic_registration_sample(
        deformation_type="smooth local translation", amplitude_um=2.0, seed=11, size=96, n_points=40
    )
    errors, summary = marker_point_raster_consistency(sample.ground_truth_warped_image, sample.fixed_points)

    assert len(errors) > 20
    assert summary["point_raster_consistency_median_pixels"] < 2.5
    assert summary["point_raster_consistency_p95_pixels"] < 4.0
