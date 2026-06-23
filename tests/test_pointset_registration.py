import numpy as np

from src.pointset_registration import (
    AffineICPResult,
    apply_affine,
    estimate_affine_with_y_flip,
    fine_center_snap_warp,
    local_translation_fine_warp,
    warp_he_image_to_world,
    world_points_to_warped_image_pixels,
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
    assert result.flip_x is False
    assert result.median_residual < 1e-6
    assert np.allclose(result.transformed_points, dst, atol=1e-6)


def test_estimate_affine_with_y_flip_can_use_y_flipped_orientation():
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

    result = estimate_affine_with_y_flip(
        src,
        dst,
        image_height_px=image_height,
        flip_candidates=((False, True),),
    )

    assert result.flip_y is True
    assert result.median_residual < 1e-6


def test_estimate_affine_with_y_flip_selects_x_flipped_orientation():
    src = np.array(
        [
            [5.0, 10.0],
            [20.0, 15.0],
            [10.0, 40.0],
            [35.0, 45.0],
            [42.0, 12.0],
        ]
    )
    image_width = 50.0
    flipped = src.copy()
    flipped[:, 0] = image_width - flipped[:, 0]
    dst = flipped * 1.2 + np.array([15.0, 8.0])

    result = estimate_affine_with_y_flip(
        src,
        dst,
        image_height_px=60.0,
        image_width_px=image_width,
        flip_candidates=((True, False),),
    )

    assert result.flip_x is True
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


def test_fine_center_snap_warp_filters_locally_inconsistent_pairs():
    source = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [5.0, 5.0],
        ]
    )
    target = source + np.array([1.0, 1.0])
    target[-1] = source[-1] + np.array([10.0, 0.0])

    result = fine_center_snap_warp(
        source,
        target,
        match_radius=15.0,
        grid_spacing=2.0,
        bandwidth=4.0,
        ridge=0.01,
        padding=5.0,
        coherence_radius=20.0,
        max_local_deviation=3.0,
        min_pair_confidence=0.0,
    )

    assert result.n_candidate_pairs < len(source) or result.n_filtered_pairs > 0
    assert result.n_pairs <= result.n_candidate_pairs


def test_warp_he_image_to_world_returns_corrected_image_grid():
    he_image = np.arange(25, dtype=np.uint8).reshape(5, 5)
    src = np.array(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [0.0, 4.0],
            [4.0, 4.0],
            [2.0, 2.0],
        ]
    )
    dst = src.copy()
    affine = AffineICPResult(
        affine_matrix=np.eye(2),
        translation=np.zeros(2),
        transformed_points=src,
        flip_x=False,
        flip_y=False,
        image_width=5.0,
        image_height=5.0,
        mean_residual=0.0,
        median_residual=0.0,
        n_pairs=len(src),
        success=True,
        message="identity",
    )
    fine = fine_center_snap_warp(dst, dst, match_radius=1.0, grid_spacing=1.0, bandwidth=1.0, ridge=0.1, padding=0.0)

    warped, metadata = warp_he_image_to_world(
        he_image,
        affine,
        fine,
        output_pixel_size_um=1.0,
        bounds=(0.0, 0.0, 5.0, 5.0),
    )

    assert warped.shape == (5, 5)
    assert metadata["output_origin"] == "upper-left"
    assert metadata["row0_world_y"] == 0.5
    assert metadata["col0_world_x"] == 0.5
    assert warped.dtype == he_image.dtype


def test_warp_he_image_to_world_can_export_lower_left_origin():
    he_image = np.arange(25, dtype=np.uint8).reshape(5, 5)
    src = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [4.0, 4.0], [2.0, 2.0]])
    affine = AffineICPResult(
        affine_matrix=np.eye(2),
        translation=np.zeros(2),
        transformed_points=src,
        flip_x=False,
        flip_y=False,
        image_width=5.0,
        image_height=5.0,
        mean_residual=0.0,
        median_residual=0.0,
        n_pairs=len(src),
        success=True,
        message="identity",
    )

    _, metadata = warp_he_image_to_world(
        he_image,
        affine,
        None,
        output_pixel_size_um=1.0,
        bounds=(0.0, 0.0, 5.0, 5.0),
        output_origin="lower-left",
    )

    assert metadata["output_origin"] == "lower-left"
    assert metadata["row0_world_y"] == 4.5


def test_world_points_to_warped_image_pixels_respects_upper_left_origin():
    metadata = {
        "output_pixel_size_um": 1.0,
        "output_origin": "upper-left",
        "col0_world_x": 0.5,
        "row0_world_y": 0.5,
    }
    pixels = world_points_to_warped_image_pixels(np.array([[0.5, 0.5], [2.5, 3.5]]), metadata)

    assert np.allclose(pixels, [[0.0, 0.0], [2.0, 3.0]])


def test_local_translation_fine_warp_improves_shifted_point_cloud():
    xs, ys = np.meshgrid(np.arange(20.0, 100.0, 20.0), np.arange(20.0, 100.0, 20.0))
    fixed = np.column_stack([xs.ravel(), ys.ravel()])
    moving = fixed + np.array([4.0, -3.0])

    result = local_translation_fine_warp(
        fixed,
        moving,
        bounds=(0.0, 0.0, 120.0, 120.0),
        density_sigma=2.0,
        density_pixel_size=1.0,
        grid_spacing=30.0,
        patch_radius=18.0,
        search_radius=8.0,
        min_correlation=0.1,
        max_shift=10.0,
        min_accepted_anchors=3,
        smoothing=0.1,
        neighbors=0,
    )

    assert result.success is True
    assert result.n_pairs >= 3
    assert result.median_pair_distance_after < result.median_pair_distance_before
    assert result.anchors is not None
