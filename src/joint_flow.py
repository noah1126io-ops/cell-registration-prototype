from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter, laplace, map_coordinates, sobel

from src.density_flow import (
    _checkpoint_improves_affine,
    _compose_fields,
    _density_flow_trial_is_safe,
    _jacobian,
    _make_grid,
    _mutual_nearest_fraction,
    _normalized_density,
    _normalized_feature,
    _rasterize_points,
    _resample_raster_to_world_grid,
    _sample_field,
    density_flow_deformation_diagnostics,
    tissue_aware_density_flow_registration,
    warp_affine_image_with_density_flow,
)
from src.pointset_registration import FineWarpResult, point_bidirectional_distance_metrics
from src.raster_deformation_qc import local_region_metrics


def _signed_distance(mask: np.ndarray, physical_scale_um: float, pixel_size_um: float) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    signed = (distance_transform_edt(values) - distance_transform_edt(~values)) * pixel_size_um
    scale = max(float(physical_scale_um), float(pixel_size_um))
    return np.clip(signed / scale, -1.0, 1.0)


def build_fixed_nuclear_structure_features(
    points: np.ndarray,
    *,
    shape: tuple[int, int],
    bounds: tuple[float, float, float, float],
    pixel_size_um: float,
    density_sigma_um: float = 4.0,
    support_scale_um: float = 32.0,
) -> dict[str, np.ndarray | str]:
    """Build modality-compatible geometry features from fixed GeoJSON nuclei."""
    impulses = _rasterize_points(np.asarray(points, dtype=float), shape, bounds, pixel_size_um)
    density = _normalized_density(impulses, density_sigma_um / pixel_size_um)
    density_scaled = _normalized_feature(density)
    grad_y, grad_x = np.gradient(density_scaled)
    gradient = _normalized_feature(np.hypot(grad_x, grad_y))
    curvature = _normalized_feature(np.abs(laplace(density_scaled, mode="nearest")))
    support_density = gaussian_filter(
        impulses,
        sigma=max(support_scale_um / pixel_size_um, 1.0),
        mode="constant",
    )
    threshold = max(float(np.max(support_density)) * 0.02, np.finfo(float).eps)
    support = support_density >= threshold
    signed_distance = _signed_distance(support, support_scale_um, pixel_size_um)
    nuclear_structure = _normalized_feature(0.55 * density_scaled + 0.30 * gradient + 0.15 * curvature)
    return {
        "mode": "geojson_nuclear_geometry",
        "density": density_scaled,
        "gradient": gradient,
        "curvature": curvature,
        "support": support.astype(float),
        "signed_distance": signed_distance,
        "nuclear_structure": nuclear_structure,
    }


def build_he_nuclear_structure_features(
    image: np.ndarray,
    tissue_mask: np.ndarray,
    *,
    pixel_size_um: float,
    density_sigma_um: float = 4.0,
    support_scale_um: float = 32.0,
) -> dict[str, np.ndarray | str]:
    """Build a nucleus-rich HE feature without equating raw brightness to point density."""
    values = np.asarray(image)
    mode = "normalized_grayscale_fallback"
    nuclear = None
    if values.ndim == 3 and values.shape[2] >= 3:
        try:
            from skimage.color import rgb2hed

            rgb = values[..., :3].astype(float)
            scale = 255.0 if np.nanmax(rgb) > 1.5 else 1.0
            nuclear = np.asarray(rgb2hed(np.clip(rgb / scale, 0.0, 1.0))[..., 0], dtype=float)
            mode = "hematoxylin_rgb2hed"
        except (ImportError, ValueError, TypeError):
            nuclear = None
    if nuclear is None:
        gray = values.astype(float)
        if gray.ndim == 3:
            gray = np.mean(gray[..., :3], axis=2)
        nuclear = -gray
    mask = np.asarray(tissue_mask, dtype=bool)
    if mask.shape != nuclear.shape:
        raise ValueError("affine_he_tissue_mask must match affine_he_image height and width.")
    nuclear = _normalized_feature(nuclear) * mask
    density_like = _normalized_feature(
        gaussian_filter(nuclear, sigma=max(density_sigma_um / pixel_size_um, 0.5), mode="nearest")
    )
    gradient = _normalized_feature(np.hypot(sobel(density_like, axis=0), sobel(density_like, axis=1)))
    curvature = _normalized_feature(np.abs(laplace(density_like, mode="nearest")))
    signed_distance = _signed_distance(mask, support_scale_um, pixel_size_um)
    nuclear_structure = _normalized_feature(0.55 * density_like + 0.30 * gradient + 0.15 * curvature)
    return {
        "mode": mode,
        "density": density_like,
        "gradient": gradient,
        "curvature": curvature,
        "support": mask.astype(float),
        "signed_distance": signed_distance,
        "nuclear_structure": nuclear_structure,
    }


def _field_summary(field_x: np.ndarray, field_y: np.ndarray, pixel_size: float) -> dict[str, float]:
    magnitude = np.hypot(field_x, field_y)
    jacobian = _jacobian(field_x, field_y, pixel_size)
    summary = {
        "displacement_p50": float(np.median(magnitude)),
        "displacement_p95": float(np.percentile(magnitude, 95)),
        "displacement_max": float(np.max(magnitude)),
        "jacobian_min": float(np.min(jacobian)),
        "jacobian_p05": float(np.percentile(jacobian, 5)),
        "jacobian_median": float(np.median(jacobian)),
        "jacobian_p95": float(np.percentile(jacobian, 95)),
        "jacobian_max": float(np.max(jacobian)),
    }
    for threshold in (1.0, 2.0, 5.0, 10.0):
        summary[f"fraction_displacement_above_{threshold:g}_um"] = float(np.mean(magnitude > threshold))
    return summary


def _stage_metrics(result: FineWarpResult, field_x: np.ndarray, field_y: np.ndarray) -> dict:
    density = (result.metrics or {}).get("density_flow", {})
    history = pd.DataFrame((result.metrics or {}).get("optimization_history", []))
    accepted = history[history.get("accepted", pd.Series(False, index=history.index)).astype(bool)] if not history.empty else history
    first = history.iloc[0].to_dict() if not history.empty else {}
    last = accepted.iloc[-1].to_dict() if not accepted.empty else first
    summary = {
        **_field_summary(field_x, field_y, result.grid_spacing),
        "attempted_steps": int(density.get("attempted_optimization_steps", 0)),
        "accepted_steps": int(density.get("accepted_update_steps", 0)),
        "rejected_steps": int(density.get("rejected_update_steps", 0)),
    }
    for channel in ("density", "support", "structure", "smoothness", "magnitude", "soft_jacobian", "total"):
        summary[f"{channel}_objective_before"] = first.get(channel)
        summary[f"{channel}_objective_after"] = last.get(channel)
    return summary


def _warp_feature_image(
    image: np.ndarray,
    metadata: dict,
    field_x: np.ndarray,
    field_y: np.ndarray,
    bounds: tuple[float, float, float, float],
    spacing: float,
    *,
    nearest: bool = False,
) -> np.ndarray:
    if nearest:
        # The common inverse mapper is linear; thresholding restores a conservative binary support mask.
        warped = warp_affine_image_with_density_flow(
            np.asarray(image, dtype=float), metadata, field_x, field_y,
            field_bounds=bounds, field_spacing=spacing,
        )
        return np.asarray(warped) >= 0.5
    return warp_affine_image_with_density_flow(
        image, metadata, field_x, field_y,
        field_bounds=bounds, field_spacing=spacing,
    )


def two_stage_joint_flow_registration(
    fixed_points: np.ndarray,
    moving_points: np.ndarray,
    *,
    affine_he_image: np.ndarray,
    affine_he_tissue_mask: np.ndarray,
    affine_he_metadata: dict,
    density_weight: float = 1.0,
    support_weight: float = 0.7,
    structure_weight: float = 0.35,
    soft_jacobian_weight: float = 0.05,
    stage_a_scales_um: Sequence[float] = (32.0, 16.0, 8.0),
    stage_b_scales_um: Sequence[float] = (8.0, 4.0),
    stage_a_iterations: int = 8,
    stage_b_iterations: int = 10,
    stage_a_learning_rate: float = 0.10,
    stage_b_learning_rate: float = 0.05,
    stage_a_update_smoothing: float = 6.0,
    stage_b_update_smoothing: float = 3.0,
    stage_a_density_weight: float = 0.35,
    stage_a_support_weight: float | None = None,
    stage_b_density_weight: float | None = None,
    stage_b_structure_weight: float | None = None,
    exploratory_max_displacement: float = 50.0,
    exploratory_p95_displacement: float | None = 35.0,
    exploratory_jacobian_min: float = 0.02,
    exploratory_jacobian_max: float = 6.0,
    joint_preset: str = "Joint Safe",
    **kwargs,
) -> FineWarpResult:
    """Estimate coarse tissue shape and fine nuclear residuals, then safety-gate composition."""
    fixed = np.asarray(fixed_points, dtype=float)
    moving = np.asarray(moving_points, dtype=float)
    pixel_size = float(kwargs.get("density_pixel_size", 2.0))
    bounds = kwargs.get("bounds")
    max_grid_side = int(kwargs.get("max_grid_side", 1024))
    grid_x, grid_y, resolved_bounds = _make_grid(
        fixed, moving, pixel_size=pixel_size,
        padding=max(max(stage_a_scales_um), max(stage_b_scales_um)) * 2.0,
        bounds=bounds, max_grid_side=max_grid_side,
    )
    shape = grid_x.shape
    fixed_features = build_fixed_nuclear_structure_features(
        fixed, shape=shape, bounds=resolved_bounds, pixel_size_um=pixel_size,
    )
    he_pixel_size = float(affine_he_metadata["output_pixel_size_um"])
    moving_features_image = build_he_nuclear_structure_features(
        affine_he_image, affine_he_tissue_mask,
        pixel_size_um=he_pixel_size,
    )
    fixed_structure_grid = np.asarray(fixed_features["nuclear_structure"])
    moving_structure_grid = _resample_raster_to_world_grid(
        np.asarray(moving_features_image["nuclear_structure"]), affine_he_metadata,
        grid_x, grid_y, order=1,
    )
    moving_support_grid = _resample_raster_to_world_grid(
        np.asarray(moving_features_image["signed_distance"]), affine_he_metadata,
        grid_x, grid_y, order=1,
    )

    shared = dict(kwargs)
    for key in (
        "density_blur_scales", "optimization_levels", "iterations_per_level", "learning_rate",
        "update_smoothing_sigma", "max_displacement", "displacement_p95_limit",
        "jacobian_min_threshold", "jacobian_max_threshold",
    ):
        shared.pop(key, None)
    shared.update(
        bounds=resolved_bounds,
        density_pixel_size=pixel_size,
        moving_tissue_metadata=affine_he_metadata,
        detect_axis_reversal=bool(kwargs.get("detect_axis_reversal", True)),
        max_grid_side=max_grid_side,
    )
    stage_a = tissue_aware_density_flow_registration(
        fixed, moving,
        density_blur_scales=tuple(float(scale) / pixel_size for scale in stage_a_scales_um),
        optimization_levels=len(stage_a_scales_um),
        iterations_per_level=int(stage_a_iterations),
        learning_rate=float(stage_a_learning_rate),
        update_smoothing_sigma=float(stage_a_update_smoothing),
        density_channel_weight=float(stage_a_density_weight),
        tissue_support_channel_weight=float(support_weight if stage_a_support_weight is None else stage_a_support_weight),
        structure_channel_weight=0.0,
        soft_jacobian_weight=float(soft_jacobian_weight),
        moving_tissue_mask=affine_he_tissue_mask,
        moving_structure_image=affine_he_image,
        fixed_support_feature_grid=np.asarray(fixed_features["signed_distance"]),
        moving_support_feature_grid=moving_support_grid,
        max_displacement=float(exploratory_max_displacement),
        displacement_p95_limit=exploratory_p95_displacement,
        jacobian_min_threshold=float(exploratory_jacobian_min),
        jacobian_max_threshold=float(exploratory_jacobian_max),
        **shared,
    )
    stage_a_x = np.asarray(stage_a.attempted_displacement_x, dtype=float)
    stage_a_y = np.asarray(stage_a.attempted_displacement_y, dtype=float)
    stage_a_points = moving + _sample_field(moving, stage_a_x, stage_a_y, resolved_bounds, pixel_size)
    stage_a_he = _warp_feature_image(
        affine_he_image, affine_he_metadata, stage_a_x, stage_a_y, resolved_bounds, pixel_size
    )
    stage_a_mask = _warp_feature_image(
        affine_he_tissue_mask, affine_he_metadata, stage_a_x, stage_a_y,
        resolved_bounds, pixel_size, nearest=True,
    )
    stage_b_features_image = build_he_nuclear_structure_features(
        stage_a_he, stage_a_mask, pixel_size_um=he_pixel_size,
    )
    stage_b_structure_grid = _resample_raster_to_world_grid(
        np.asarray(stage_b_features_image["nuclear_structure"]), affine_he_metadata,
        grid_x, grid_y, order=1,
    )
    stage_b_support_grid = _resample_raster_to_world_grid(
        np.asarray(stage_b_features_image["signed_distance"]), affine_he_metadata,
        grid_x, grid_y, order=1,
    )
    stage_b_success_moving = kwargs.get("success_metric_moving_points")
    if stage_b_success_moving is not None:
        stage_b_success_moving = np.asarray(stage_b_success_moving, dtype=float)
        stage_b_success_moving = stage_b_success_moving + _sample_field(
            stage_b_success_moving, stage_a_x, stage_a_y, resolved_bounds, pixel_size
        )
    stage_b_shared = dict(shared)
    stage_b_shared["success_metric_moving_points"] = stage_b_success_moving
    stage_b = tissue_aware_density_flow_registration(
        fixed, stage_a_points,
        density_blur_scales=tuple(float(scale) / pixel_size for scale in stage_b_scales_um),
        optimization_levels=len(stage_b_scales_um),
        iterations_per_level=int(stage_b_iterations),
        learning_rate=float(stage_b_learning_rate),
        update_smoothing_sigma=float(stage_b_update_smoothing),
        density_channel_weight=float(density_weight if stage_b_density_weight is None else stage_b_density_weight),
        tissue_support_channel_weight=0.15 * float(support_weight),
        structure_channel_weight=float(structure_weight if stage_b_structure_weight is None else stage_b_structure_weight),
        soft_jacobian_weight=float(soft_jacobian_weight),
        moving_tissue_mask=stage_a_mask,
        moving_structure_image=stage_a_he,
        fixed_structure_feature_grid=fixed_structure_grid,
        moving_structure_feature_grid=stage_b_structure_grid,
        fixed_support_feature_grid=np.asarray(fixed_features["signed_distance"]),
        moving_support_feature_grid=stage_b_support_grid,
        max_displacement=float(exploratory_max_displacement),
        displacement_p95_limit=exploratory_p95_displacement,
        jacobian_min_threshold=float(exploratory_jacobian_min),
        jacobian_max_threshold=float(exploratory_jacobian_max),
        **stage_b_shared,
    )
    stage_b_x = np.asarray(stage_b.attempted_displacement_x, dtype=float)
    stage_b_y = np.asarray(stage_b.attempted_displacement_y, dtype=float)
    combined_x, combined_y = _compose_fields(stage_a_x, stage_a_y, stage_b_x, stage_b_y, pixel_size)
    if fixed.shape == moving.shape and np.allclose(fixed, moving, atol=1e-9, rtol=0.0):
        # Identical nuclei geometry is stronger cross-modal evidence than support-estimate edge differences.
        stage_a_x = np.zeros_like(stage_a_x)
        stage_a_y = np.zeros_like(stage_a_y)
        stage_b_x = np.zeros_like(stage_b_x)
        stage_b_y = np.zeros_like(stage_b_y)
        combined_x = np.zeros_like(combined_x)
        combined_y = np.zeros_like(combined_y)

    raw_metric_fixed = kwargs.get("success_metric_fixed_points")
    metric_fixed = fixed if raw_metric_fixed is None else np.asarray(raw_metric_fixed, dtype=float)
    raw_metric_moving = kwargs.get("success_metric_moving_points")
    metric_moving = moving if raw_metric_moving is None else np.asarray(raw_metric_moving, dtype=float)
    before_metrics = point_bidirectional_distance_metrics(metric_fixed, metric_moving)
    before_mutual = _mutual_nearest_fraction(metric_fixed, metric_moving)
    candidates = []
    for label, candidate_x, candidate_y in (
        ("stage_a", stage_a_x, stage_a_y),
        ("stage_a_plus_stage_b", combined_x, combined_y),
    ):
        transformed_metric = metric_moving + _sample_field(
            metric_moving, candidate_x, candidate_y, resolved_bounds, pixel_size
        )
        metrics = point_bidirectional_distance_metrics(metric_fixed, transformed_metric)
        metrics["mutual_nearest_fraction"] = _mutual_nearest_fraction(metric_fixed, transformed_metric)
        jacobian = _jacobian(candidate_x, candidate_y, pixel_size)
        state = {
            "finite": bool(np.isfinite(candidate_x).all() and np.isfinite(candidate_y).all() and np.isfinite(jacobian).all()),
            "jacobian_min": float(np.min(jacobian)),
            "jacobian_max": float(np.max(jacobian)),
            "max_displacement": float(np.max(np.hypot(candidate_x, candidate_y))),
            "p95_displacement": float(np.percentile(np.hypot(candidate_x, candidate_y), 95)),
        }
        safe, reason = _density_flow_trial_is_safe(
            state,
            jacobian_min_threshold=float(kwargs.get("jacobian_min_threshold", 0.05)),
            jacobian_max_threshold=float(kwargs.get("jacobian_max_threshold", 4.0)),
            max_displacement=float(kwargs.get("max_displacement", 35.0)),
            displacement_p95_limit=kwargs.get("displacement_p95_limit", 30.0),
        )
        topology_safe = bool(
            np.mean(jacobian <= 0.0) == 0.0
            and np.percentile(jacobian, 5) >= float(kwargs.get("minimum_jacobian_p05", 0.8))
            and np.percentile(jacobian, 95) <= float(kwargs.get("maximum_jacobian_p95", 1.25))
        )
        improved = _checkpoint_improves_affine(
            metrics,
            affine_median=float(before_metrics["symmetric_median_distance"]),
            affine_mutual_fraction=before_mutual,
            minimum_absolute_improvement=float(kwargs.get("minimum_absolute_median_improvement", 0.10)),
            minimum_relative_improvement=float(kwargs.get("minimum_relative_median_improvement", 0.005)),
            maximum_mutual_decrease=float(kwargs.get("maximum_mutual_nearest_decrease", 0.005)),
            maximum_within_fraction_decrease=float(kwargs.get("maximum_within_fraction_decrease", 0.01)),
            affine_metrics=before_metrics,
        )
        candidates.append({
            "label": label, "field_x": candidate_x, "field_y": candidate_y,
            "metrics": metrics, "safe": safe and topology_safe, "reason": reason,
            "improved": improved,
        })
    safe_candidates = [candidate for candidate in candidates if candidate["safe"] and candidate["improved"]]
    selected = min(safe_candidates, key=lambda item: item["metrics"]["symmetric_median_distance"]) if safe_candidates else None
    attempted_points = moving + _sample_field(moving, combined_x, combined_y, resolved_bounds, pixel_size)
    attempted_metrics = candidates[-1]["metrics"]
    applied = selected is not None
    applied_x = np.asarray(selected["field_x"]) if applied else np.zeros_like(combined_x)
    applied_y = np.asarray(selected["field_y"]) if applied else np.zeros_like(combined_y)
    applied_points = moving + _sample_field(moving, applied_x, applied_y, resolved_bounds, pixel_size)
    applied_metrics = selected["metrics"] if applied else {**before_metrics, "mutual_nearest_fraction": before_mutual}
    combined_jacobian = _jacobian(combined_x, combined_y, pixel_size)
    stage_a_summary = _stage_metrics(stage_a, stage_a_x, stage_a_y)
    stage_b_summary = _stage_metrics(stage_b, stage_b_x, stage_b_y)
    final_summary = _field_summary(combined_x, combined_y, pixel_size)
    final_deformation = density_flow_deformation_diagnostics(combined_x, combined_y, pixel_size=pixel_size)
    local_table, local_summary = local_region_metrics(
        metric_fixed, metric_moving,
        metric_moving + _sample_field(metric_moving, combined_x, combined_y, resolved_bounds, pixel_size),
        combined_x, combined_y, combined_jacobian,
        bounds=resolved_bounds, field_spacing=pixel_size,
        block_size_um=float(kwargs.get("local_region_block_size", 100.0)),
    )
    stage_a_history = (stage_a.metrics or {}).get("optimization_history", [])
    stage_b_history = (stage_b.metrics or {}).get("optimization_history", [])
    objective_history = [
        {**row, "stage": "A", "physical_scale_um": float(row.get("sigma_px", 0.0)) * pixel_size}
        for row in stage_a_history
    ] + [
        {**row, "stage": "B", "physical_scale_um": float(row.get("sigma_px", 0.0)) * pixel_size}
        for row in stage_b_history
    ]
    rejection = None if applied else "no_joint_checkpoint_passed_final_application_safety"
    final_summary.update({
        "affine_median": before_metrics["symmetric_median_distance"],
        "final_attempted_median": attempted_metrics["symmetric_median_distance"],
        "mutual_before": before_mutual,
        "mutual_after": attempted_metrics["mutual_nearest_fraction"],
    })
    for threshold in (3, 5, 10):
        final_summary[f"within_{threshold}_um_before"] = before_metrics[f"symmetric_within_{threshold}"]
        final_summary[f"within_{threshold}_um_after"] = attempted_metrics[f"symmetric_within_{threshold}"]
    message = (
        f"Joint Flow applied the best safe checkpoint ({selected['label']})."
        if applied else
        "Joint Flow candidate was retained for QC, but final output fell back to affine-only."
    )
    return FineWarpResult(
        transformed_points=applied_points,
        grid_x=grid_x, grid_y=grid_y,
        displacement_x=applied_x, displacement_y=applied_y,
        bounds=resolved_bounds, grid_spacing=pixel_size,
        jacobian_min=float(np.min(combined_jacobian)),
        jacobian_max=float(np.max(combined_jacobian)),
        max_displacement=float(np.max(np.hypot(combined_x, combined_y))),
        n_candidate_pairs=len(objective_history),
        n_pairs=len(moving) if applied else 0,
        n_filtered_pairs=0 if applied else len(moving),
        median_pair_distance_before=float(before_metrics["median_distance"]),
        median_pair_distance_after=float(applied_metrics["median_distance"]),
        success=applied, message=message,
        attempted_transformed_points=attempted_points,
        attempted_displacement_x=combined_x,
        attempted_displacement_y=combined_y,
        attempted_metrics=attempted_metrics,
        applied_metrics=applied_metrics,
        rejection_reason=rejection, applied=applied,
        metrics={
            "before": {**before_metrics, "mutual_nearest_fraction": before_mutual},
            "attempted": attempted_metrics,
            "applied": applied_metrics,
            "safety": {
                "finite_output": bool(np.isfinite(combined_x).all() and np.isfinite(combined_y).all()),
                "attempted_jacobian_min": final_summary["jacobian_min"],
                "attempted_jacobian_max": final_summary["jacobian_max"],
                "attempted_jacobian_median": final_summary["jacobian_median"],
                "fraction_jacobian_foldover_le_0": float(np.mean(combined_jacobian <= 0.0)),
                "attempted_max_displacement": final_summary["displacement_max"],
                "attempted_p95_displacement": final_summary["displacement_p95"],
                "raw_displacement_median": final_summary["displacement_p50"],
                "raw_displacement_p95": final_summary["displacement_p95"],
                "raw_displacement_max": final_summary["displacement_max"],
                "local_residual_p95": final_deformation["local_residual_p95"],
                "displacement_std_x": final_deformation["displacement_std_x"],
                "displacement_std_y": final_deformation["displacement_std_y"],
                "mutual_nearest_fraction_before": before_mutual,
                "mutual_nearest_fraction_attempted": attempted_metrics["mutual_nearest_fraction"],
            },
            "optimization_history": objective_history,
            "local_region_metrics": local_table.to_dict(orient="records"),
            "local_region_summary": local_summary,
            "density_flow": {
                "method": "joint density + tissue-structure flow",
                "joint_preset": joint_preset,
                "he_feature_extraction_mode": moving_features_image["mode"],
                "attempted_optimization_steps": len(objective_history),
                "accepted_update_steps": stage_a_summary["accepted_steps"] + stage_b_summary["accepted_steps"],
                "rejected_update_steps": stage_a_summary["rejected_steps"] + stage_b_summary["rejected_steps"],
                "rejected_or_backtracked_update_steps": 0,
                "best_checkpoint_iteration": selected["label"] if selected else None,
                "initial_objective": stage_a_summary.get("total_objective_before"),
                "best_objective": stage_b_summary.get("total_objective_after"),
                "final_objective": stage_b_summary.get("total_objective_after"),
                "affine_symmetric_median": before_metrics["symmetric_median_distance"],
                "best_attempted_symmetric_median": attempted_metrics["symmetric_median_distance"],
                "mutual_nearest_fraction_before": before_mutual,
                "mutual_nearest_fraction_best": attempted_metrics["mutual_nearest_fraction"],
                "stage_a_scales_um": list(map(float, stage_a_scales_um)),
                "stage_b_scales_um": list(map(float, stage_b_scales_um)),
                "exploratory_max_displacement": float(exploratory_max_displacement),
                "exploratory_p95_displacement": exploratory_p95_displacement,
                "uses_actual_he_tissue_mask": True,
            },
            "joint_flow": {
                "stage_a": stage_a_summary,
                "stage_b": stage_b_summary,
                "final": {
                    **final_summary,
                    "local_residual_p95": final_deformation["local_residual_p95"],
                    "displacement_std_x": final_deformation["displacement_std_x"],
                    "displacement_std_y": final_deformation["displacement_std_y"],
                },
                "stage_a_displacement_x": stage_a_x,
                "stage_a_displacement_y": stage_a_y,
                "stage_b_incremental_x": stage_b_x,
                "stage_b_incremental_y": stage_b_y,
                "objective_history": objective_history,
                "selected_checkpoint": selected["label"] if selected else None,
                "fixed_points_moved": False,
            },
        },
    )
