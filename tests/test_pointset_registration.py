import numpy as np

from src.pointset_registration import (
    AffineICPResult,
    FineWarpResult,
    apply_affine,
    cluster_anchor_fine_warp,
    estimate_affine_with_y_flip,
    fine_center_snap_warp,
    local_translation_fine_warp,
    matched_nuclei_rbf_fine_warp,
    warp_he_image_to_world,
    world_points_to_warped_image_pixels,
)


def test_fine_warp_result_legacy_constructor_has_diagnostic_defaults():
    grid_x, grid_y = np.meshgrid(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    zeros = np.zeros_like(grid_x)

    result = FineWarpResult(
        transformed_points=np.array([[0.0, 0.0]]),
        grid_x=grid_x,
        grid_y=grid_y,
        displacement_x=zeros,
        displacement_y=zeros,
        bounds=(0.0, 0.0, 1.0, 1.0),
        grid_spacing=1.0,
        jacobian_min=1.0,
        jacobian_max=1.0,
        max_displacement=0.0,
        n_candidate_pairs=0,
        n_pairs=0,
        n_filtered_pairs=0,
        median_pair_distance_before=0.0,
        median_pair_distance_after=0.0,
        success=True,
        message="legacy",
    )

    assert result.attempted_transformed_points is None
    assert result.attempted_displacement_x is None
    assert result.applied is None


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
    assert np.allclose(result.attempted_transformed_points, result.transformed_points)
    assert np.allclose(result.attempted_displacement_x, result.displacement_x)
    assert np.allclose(result.attempted_displacement_y, result.displacement_y)
    assert result.applied is True
    assert result.rejection_reason is None


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
    assert result.applied is True
    assert result.attempted_metrics is not None
    assert result.applied_metrics is not None


def test_local_translation_rejection_preserves_attempted_candidate():
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
        max_shift=1.0,
        min_accepted_anchors=3,
        jacobian_min_threshold=2.0,
        smoothing=0.1,
        neighbors=0,
    )

    assert result.success is False
    assert result.applied is False
    assert result.rejection_reason
    assert result.attempted_transformed_points is not None
    assert result.attempted_displacement_x is not None
    assert result.applied_metrics is not None


def test_matched_nuclei_rbf_fine_warp_improves_paired_shift():
    xs, ys = np.meshgrid(np.arange(10.0, 60.0, 10.0), np.arange(10.0, 60.0, 10.0))
    fixed = np.column_stack([xs.ravel(), ys.ravel()])
    moving = fixed + np.array([2.0, -3.0])

    result = matched_nuclei_rbf_fine_warp(
        moving,
        fixed,
        match_radius=6.0,
        grid_spacing=5.0,
        smoothing=0.1,
        neighbors=0,
        min_pairs=4,
        max_displacement=10.0,
    )

    before = np.mean(np.linalg.norm(moving - fixed, axis=1))
    after = np.mean(np.linalg.norm(result.transformed_points - fixed, axis=1))
    assert result.success is True
    assert result.applied is True
    assert result.n_pairs >= 4
    assert result.anchors is not None
    assert after < before
    assert np.max(np.abs(result.displacement_x)) > 0


def test_matched_nuclei_rbf_rejection_preserves_attempted_candidate():
    xs, ys = np.meshgrid(np.arange(10.0, 60.0, 10.0), np.arange(10.0, 60.0, 10.0))
    fixed = np.column_stack([xs.ravel(), ys.ravel()])
    moving = fixed + np.array([2.0, -3.0])

    result = matched_nuclei_rbf_fine_warp(
        moving,
        fixed,
        match_radius=6.0,
        grid_spacing=5.0,
        smoothing=0.1,
        neighbors=0,
        min_pairs=4,
        max_displacement=10.0,
        jacobian_min_threshold=2.0,
    )

    assert result.success is False
    assert result.applied is False
    assert result.rejection_reason
    assert np.allclose(result.transformed_points, moving)
    assert result.attempted_transformed_points is not None
    assert not np.allclose(result.attempted_transformed_points, moving)
    assert result.attempted_displacement_x is not None


def test_cluster_anchor_fine_warp_improves_local_cluster_shift():
    xs, ys = np.meshgrid(np.arange(10.0, 70.0, 10.0), np.arange(10.0, 70.0, 10.0))
    fixed = np.column_stack([xs.ravel(), ys.ravel()])
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

    before = np.mean(np.linalg.norm(moving - fixed, axis=1))
    after = np.mean(np.linalg.norm(result.transformed_points - fixed, axis=1))
    assert result.success is True
    assert result.applied is True
    assert result.n_pairs >= 3
    assert result.anchors is not None
    assert after < before
    expected_columns = {
        "anchor_x",
        "anchor_y",
        "dx",
        "dy",
        "shift_magnitude",
        "n_fixed_points",
        "n_moving_points",
        "median_distance_zero_shift",
        "median_distance_best_shift",
        "improvement",
        "fraction_within_threshold_zero_shift",
        "fraction_within_threshold_best_shift",
        "accepted",
        "rejection_reason",
        "fixed_cluster_point_count",
        "moving_cluster_point_count",
        "fixed_cluster_radius",
        "moving_cluster_radius",
        "mutual_matches_before",
        "mutual_matches_after",
        "score_before",
        "score_after",
        "score_improvement",
        "selected_dx",
        "selected_dy",
    }
    assert expected_columns.issubset(result.anchors.columns)


def test_cluster_anchor_hybrid_knn_caps_cluster_sizes_and_reports_diagnostics():
    xs, ys = np.meshgrid(np.arange(5.0, 76.0, 5.0), np.arange(5.0, 76.0, 5.0))
    fixed = np.column_stack([xs.ravel(), ys.ravel()])
    moving = fixed + np.array([4.0, -2.0])

    result = cluster_anchor_fine_warp(
        fixed,
        moving,
        bounds=(0.0, 0.0, 80.0, 80.0),
        grid_spacing=20.0,
        search_radius=8.0,
        search_step=2.0,
        cluster_selection_mode="hybrid k-nearest",
        target_points_per_cluster=20,
        min_points_per_cluster=8,
        max_cluster_radius_um=30.0,
        moving_candidate_pool_ratio=1.5,
        match_threshold=4.0,
        min_improvement=0.5,
        max_shift=10.0,
        min_accepted_anchors=3,
        smoothing=0.1,
        kernel="linear",
        neighbors=30,
    )

    cluster_rows = result.anchors[result.anchors["anchor_type"] == "cluster"]
    evaluated_rows = cluster_rows[cluster_rows["score_before"].notna()]
    assert result.success is True
    assert not evaluated_rows.empty
    assert cluster_rows["fixed_cluster_point_count"].max() <= 20
    assert cluster_rows["moving_cluster_point_count"].max() <= 30
    assert cluster_rows["fixed_cluster_radius"].max() <= 30.0 + 1e-9
    assert cluster_rows["moving_cluster_radius"].max() <= 38.0 + 1e-9
    assert (evaluated_rows["cluster_selection_mode"] == "hybrid k-nearest").all()
    accepted = evaluated_rows[evaluated_rows["accepted"]]
    assert not accepted.empty
    assert (accepted["score_after"] < accepted["score_before"]).all()
    assert np.allclose(
        accepted["score_improvement"],
        accepted["score_before"] - accepted["score_after"],
    )
    assert (accepted["mutual_matches_after"] >= accepted["mutual_matches_before"]).all()
    assert np.allclose(accepted[["selected_dx", "selected_dy"]], accepted[["dx", "dy"]])


def test_cluster_anchor_hybrid_knn_rejects_sparse_clusters():
    fixed = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    moving = fixed + np.array([2.0, 1.0])

    result = cluster_anchor_fine_warp(
        fixed,
        moving,
        bounds=(-5.0, -5.0, 15.0, 15.0),
        grid_spacing=10.0,
        cluster_selection_mode="hybrid k-nearest",
        target_points_per_cluster=20,
        min_points_per_cluster=8,
        max_cluster_radius_um=40.0,
        moving_candidate_pool_ratio=1.5,
        min_accepted_anchors=1,
    )

    cluster_rows = result.anchors[result.anchors["anchor_type"] == "cluster"]
    assert result.success is False
    assert result.rejection_reason == "too_few_accepted_anchors"
    assert (cluster_rows["rejection_reason"] == "too_few_cluster_points").all()


def test_cluster_anchor_radius_mode_matches_explicit_legacy_selection():
    xs, ys = np.meshgrid(np.arange(10.0, 70.0, 10.0), np.arange(10.0, 70.0, 10.0))
    fixed = np.column_stack([xs.ravel(), ys.ravel()])
    moving = fixed + np.array([4.0, -2.0])
    kwargs = {
        "bounds": (0.0, 0.0, 80.0, 80.0),
        "grid_spacing": 20.0,
        "patch_radius": 18.0,
        "search_radius": 8.0,
        "search_step": 2.0,
        "min_points_per_cluster": 3,
        "match_threshold": 4.0,
        "min_improvement": 0.5,
        "max_shift": 10.0,
        "min_accepted_anchors": 3,
        "smoothing": 0.1,
        "neighbors": 0,
    }

    implicit_radius = cluster_anchor_fine_warp(fixed, moving, **kwargs)
    explicit_radius = cluster_anchor_fine_warp(
        fixed,
        moving,
        cluster_selection_mode="radius",
        **kwargs,
    )

    columns = ["dx", "dy", "accepted", "rejection_reason"]
    assert implicit_radius.success == explicit_radius.success
    assert implicit_radius.anchors[columns].equals(explicit_radius.anchors[columns])
    assert np.allclose(implicit_radius.transformed_points, explicit_radius.transformed_points)


def test_cluster_anchor_rejection_preserves_attempted_candidate():
    xs, ys = np.meshgrid(np.arange(10.0, 70.0, 10.0), np.arange(10.0, 70.0, 10.0))
    fixed = np.column_stack([xs.ravel(), ys.ravel()])
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
        jacobian_min_threshold=2.0,
    )

    assert result.success is False
    assert result.applied is False
    assert result.rejection_reason
    assert np.allclose(result.transformed_points, moving)
    assert result.attempted_transformed_points is not None
    assert not np.allclose(result.attempted_transformed_points, moving)


def test_cluster_anchor_uses_valid_region_success_metrics_and_boundary_pins():
    xs, ys = np.meshgrid(np.arange(10.0, 70.0, 10.0), np.arange(10.0, 70.0, 10.0))
    valid_fixed = np.column_stack([xs.ravel(), ys.ravel()])
    edge_fixed = np.array([[200.0, 200.0], [210.0, 200.0], [200.0, 210.0]], dtype=float)
    fixed = np.vstack([valid_fixed, edge_fixed])
    moving = valid_fixed + np.array([4.0, -2.0])
    weights = np.concatenate([np.ones(len(valid_fixed)), np.full(len(edge_fixed), 0.3)])
    boundary_pins = np.array([[0.0, 0.0], [80.0, 0.0], [0.0, 80.0], [80.0, 80.0]])

    result = cluster_anchor_fine_warp(
        fixed,
        moving,
        fixed_point_weights=weights,
        success_metric_fixed_points=valid_fixed,
        boundary_anchor_points=boundary_pins,
        boundary_anchor_weight=3.0,
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
        smoothing=8.0,
        kernel="linear",
        neighbors=30,
    )

    assert result.success is True
    assert result.metrics["attempted"]["median_distance"] < result.metrics["before"]["median_distance"]
    assert result.anchors is not None
    boundary_rows = result.anchors[result.anchors["anchor_type"] == "boundary_pin"]
    assert len(boundary_rows) == len(boundary_pins)
    assert np.allclose(boundary_rows[["dx", "dy"]].to_numpy(dtype=float), 0.0)
    assert np.allclose(boundary_rows["weight"].to_numpy(dtype=float), 3.0)


def test_cluster_anchor_rejects_unsafe_final_displacement_but_keeps_attempted():
    xs, ys = np.meshgrid(np.arange(10.0, 70.0, 10.0), np.arange(10.0, 70.0, 10.0))
    fixed = np.column_stack([xs.ravel(), ys.ravel()])
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
        kernel="linear",
        neighbors=30,
        max_final_displacement=1.0,
    )

    assert result.success is False
    assert result.applied is False
    assert result.rejection_reason == "max_displacement_too_large"
    assert np.allclose(result.transformed_points, moving)
    assert result.attempted_transformed_points is not None
    assert not np.allclose(result.attempted_transformed_points, moving)
    assert result.metrics is not None
    assert result.metrics["safety"]["attempted_max_displacement"] > 1.0


def test_cluster_anchor_success_uses_valid_moving_subset_for_symmetric_metrics():
    xs, ys = np.meshgrid(np.arange(10.0, 70.0, 10.0), np.arange(10.0, 70.0, 10.0))
    fixed_valid = np.column_stack([xs.ravel(), ys.ravel()])
    moving_valid = fixed_valid + np.array([4.0, -2.0])
    moving_outside = np.array([[250.0, 250.0], [260.0, 250.0], [250.0, 260.0]], dtype=float)
    moving_all = np.vstack([moving_valid, moving_outside])

    result = cluster_anchor_fine_warp(
        fixed_valid,
        moving_all,
        success_metric_fixed_points=fixed_valid,
        success_metric_moving_points=moving_valid,
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
        kernel="linear",
        neighbors=30,
    )

    assert result.success is True
    assert result.metrics is not None
    assert result.metrics["attempted"]["symmetric_median_distance"] < result.metrics["before"]["symmetric_median_distance"]
    assert "he_to_geojson_median_distance" in result.metrics["attempted"]
    assert "geojson_to_he_median_distance" in result.metrics["attempted"]
