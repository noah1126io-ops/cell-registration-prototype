import numpy as np

from src.pointset_registration import (
    apply_affine,
    estimate_affine_with_y_flip,
    fine_center_snap_warp,
)


def test_estimate_affine_with_y_flip_recovers_known_transform_without_flip():
    src = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [5.0, 3.0],
            [2.0, 7.0],
        ]
    )
    affine = np.array([[2.0, 0.2], [-0.1, 1.8]])
    translation = np.array([20.0, 30.0])
    dst = apply_affine(src, affine, translation)

    result = estimate_affine_with_y_flip(src, dst, image_height_px=20.0)

    assert result.flip_y is False
    assert result.median_residual < 1e-6
    assert np.allclose(result.transformed_points, dst, atol=1e-6)


def test_estimate_affine_with_y_flip_selects_flipped_orientation():
    src = np.array(
        [
            [5.0, 10.0],
            [20.0, 15.0],
            [10.0, 40.0],
            [35.0, 45.0],
            [42.0, 12.0],
        ]
    )
    image_height = 60.0
    flipped = src.copy()
    flipped[:, 1] = image_height - flipped[:, 1]
    dst = flipped * 1.5 + np.array([100.0, 25.0])

    result = estimate_affine_with_y_flip(src, dst, image_height_px=image_height)

    assert result.flip_y is True
    assert result.median_residual < 1e-6


def test_fine_center_snap_warp_moves_points_toward_targets():
    source = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [5.0, 5.0],
        ]
    )
    target = source + np.array([2.0, -1.0])

    result = fine_center_snap_warp(
        source,
        target,
        match_radius=5.0,
        grid_spacing=2.0,
        bandwidth=4.0,
        ridge=0.01,
        padding=5.0,
    )

    before = np.mean(np.linalg.norm(source - target, axis=1))
    after = np.mean(np.linalg.norm(result.transformed_points - target, axis=1))
    assert result.success is True
    assert result.n_pairs == len(source)
    assert after < before
    assert np.isfinite(result.jacobian_min)
