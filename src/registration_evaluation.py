from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from scipy.ndimage import sobel
from scipy.spatial import cKDTree


EVALUATION_VERSION = "workflow-c-evaluation-v1"
METRIC_DEFINITIONS_VERSION = "2026-08-12"


def _points(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2).")
    if len(array) == 0:
        raise ValueError(f"{name} must contain at least one point.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite coordinates.")
    return array


def landmark_tre_metrics(
    fixed_landmarks: np.ndarray,
    moving_landmarks_before: np.ndarray,
    moving_landmarks_after: np.ndarray,
    *,
    landmark_ids: np.ndarray | list | None = None,
) -> tuple[pd.DataFrame, dict]:
    fixed = _points(fixed_landmarks, "fixed_landmarks")
    before = _points(moving_landmarks_before, "moving_landmarks_before")
    after = _points(moving_landmarks_after, "moving_landmarks_after")
    if not (len(fixed) == len(before) == len(after)):
        raise ValueError("Fixed, before, and after landmarks must have identical length.")
    ids = np.arange(1, len(fixed) + 1) if landmark_ids is None else np.asarray(landmark_ids)
    if len(ids) != len(fixed):
        raise ValueError("landmark_ids must have the same length as landmark arrays.")
    tre_before = np.linalg.norm(fixed - before, axis=1)
    tre_after = np.linalg.norm(fixed - after, axis=1)
    delta = tre_after - tre_before
    table = pd.DataFrame({
        "landmark_id": ids,
        "fixed_x": fixed[:, 0], "fixed_y": fixed[:, 1],
        "moving_before_x": before[:, 0], "moving_before_y": before[:, 1],
        "moving_after_x": after[:, 0], "moving_after_y": after[:, 1],
        "tre_before_um": tre_before, "tre_after_um": tre_after,
        "delta_tre_um": delta, "improved": delta < 0,
    })
    before_median = float(np.median(tre_before))
    after_median = float(np.median(tre_after))
    deterioration = np.maximum(delta, 0.0)
    summary = {
        "n_landmarks": int(len(fixed)),
        "tre_median_before_um": before_median,
        "tre_median_after_um": after_median,
        "tre_p90_before_um": float(np.percentile(tre_before, 90)),
        "tre_p90_after_um": float(np.percentile(tre_after, 90)),
        "tre_p95_before_um": float(np.percentile(tre_before, 95)),
        "tre_p95_after_um": float(np.percentile(tre_after, 95)),
        "tre_max_before_um": float(np.max(tre_before)),
        "tre_max_after_um": float(np.max(tre_after)),
        "tre_mean_before_um": float(np.mean(tre_before)),
        "tre_mean_after_um": float(np.mean(tre_after)),
        "absolute_median_improvement_um": before_median - after_median,
        "relative_median_improvement": (
            (before_median - after_median) / before_median if before_median > 0 else 0.0
        ),
        "fraction_landmarks_improved": float(np.mean(delta < 0)),
        "fraction_landmarks_worsened": float(np.mean(delta > 0)),
        "worst_landmark_deterioration_um": float(np.max(deterioration)),
    }
    return table, summary


def sample_displacement_field(
    points: np.ndarray,
    displacement_x: np.ndarray,
    displacement_y: np.ndarray,
    *,
    bounds: tuple[float, float, float, float],
    spacing: float,
) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    values = _points(points, "points")
    min_x, min_y, _, _ = map(float, bounds)
    rows = (values[:, 1] - min_y) / float(spacing)
    cols = (values[:, 0] - min_x) / float(spacing)
    dx = map_coordinates(np.asarray(displacement_x, dtype=float), [rows, cols], order=1, mode="nearest")
    dy = map_coordinates(np.asarray(displacement_y, dtype=float), [rows, cols], order=1, mode="nearest")
    return np.column_stack([dx, dy])


def transform_validation_landmarks(
    moving_landmarks: np.ndarray,
    *,
    affine_matrix: np.ndarray,
    translation: np.ndarray,
    flip_x: bool = False,
    flip_y: bool = False,
    image_width: float = 0.0,
    image_height: float = 0.0,
    attempted_displacement_x: np.ndarray,
    attempted_displacement_y: np.ndarray,
    applied_displacement_x: np.ndarray,
    applied_displacement_y: np.ndarray,
    field_bounds: tuple[float, float, float, float],
    field_spacing: float,
) -> dict[str, np.ndarray]:
    raw = _points(moving_landmarks, "moving_landmarks").copy()
    oriented = raw.copy()
    if flip_x:
        oriented[:, 0] = float(image_width) - oriented[:, 0]
    if flip_y:
        oriented[:, 1] = float(image_height) - oriented[:, 1]
    matrix = np.asarray(affine_matrix, dtype=float)
    shift = np.asarray(translation, dtype=float)
    if matrix.shape != (2, 2) or shift.shape != (2,):
        raise ValueError("affine_matrix and translation must have shapes (2,2) and (2,).")
    affine = oriented @ matrix.T + shift
    attempted = affine + sample_displacement_field(
        affine, attempted_displacement_x, attempted_displacement_y,
        bounds=field_bounds, spacing=field_spacing,
    )
    applied = affine + sample_displacement_field(
        affine, applied_displacement_x, applied_displacement_y,
        bounds=field_bounds, spacing=field_spacing,
    )
    return {"raw": raw, "affine": affine, "attempted": attempted, "applied": applied}


def _mutual_fraction(fixed: np.ndarray, moving: np.ndarray) -> float:
    fixed_tree = cKDTree(fixed)
    moving_tree = cKDTree(moving)
    _, moving_to_fixed = fixed_tree.query(moving, k=1)
    _, fixed_to_moving = moving_tree.query(fixed, k=1)
    mutual = sum(
        fixed_to_moving[int(fixed_index)] == moving_index
        for moving_index, fixed_index in enumerate(moving_to_fixed)
    )
    return float(2.0 * mutual / max(len(fixed) + len(moving), 1))


def pointset_stage_metrics(fixed_points: np.ndarray, moving_points: np.ndarray) -> dict:
    fixed = _points(fixed_points, "fixed_points")
    moving = _points(moving_points, "moving_points")
    moving_to_fixed = cKDTree(fixed).query(moving, k=1)[0]
    fixed_to_moving = cKDTree(moving).query(fixed, k=1)[0]
    distances = np.concatenate([moving_to_fixed, fixed_to_moving])
    result = {
        "symmetric_nn_median_um": float(np.median(distances)),
        "bidirectional_p90_um": float(np.percentile(distances, 90)),
        "bidirectional_p95_um": float(np.percentile(distances, 95)),
        "mutual_nearest_fraction": _mutual_fraction(fixed, moving),
    }
    for threshold in (3.0, 5.0, 10.0):
        result[f"within_{threshold:g}_um_fraction"] = float(np.mean(distances <= threshold))
    return result


def pointset_scorecard(
    fixed_points: np.ndarray,
    stages: Mapping[str, np.ndarray],
    *,
    affine_stage: str = "affine",
) -> pd.DataFrame:
    rows = [{"stage": name, **pointset_stage_metrics(fixed_points, values)} for name, values in stages.items()]
    table = pd.DataFrame(rows)
    if affine_stage not in set(table["stage"]):
        raise ValueError("affine_stage must be present in stages.")
    baseline = table.loc[table["stage"] == affine_stage].iloc[0]
    for metric in (
        "symmetric_nn_median_um", "bidirectional_p90_um", "bidirectional_p95_um",
        "within_3_um_fraction", "within_5_um_fraction", "within_10_um_fraction",
        "mutual_nearest_fraction",
    ):
        table[f"delta_vs_affine_{metric}"] = table[metric] - float(baseline[metric])
    table["metric_type"] = "Internal point-set metric"
    return table


def deformation_validity_metrics(
    displacement_x: np.ndarray,
    displacement_y: np.ndarray,
    *,
    spacing: float,
) -> dict:
    dx = np.asarray(displacement_x, dtype=float)
    dy = np.asarray(displacement_y, dtype=float)
    if dx.shape != dy.shape or dx.ndim != 2 or not np.isfinite(dx).all() or not np.isfinite(dy).all():
        raise ValueError("Displacement fields must be matching finite 2D arrays.")
    dfx_dy, dfx_dx = np.gradient(dx, spacing, spacing)
    dfy_dy, dfy_dx = np.gradient(dy, spacing, spacing)
    jacobian = (1 + dfx_dx) * (1 + dfy_dy) - dfx_dy * dfy_dx
    magnitude = np.hypot(dx, dy)
    protected = np.maximum(jacobian, 1e-12)
    result = {
        "displacement_median_um": float(np.median(magnitude)),
        "displacement_p90_um": float(np.percentile(magnitude, 90)),
        "displacement_p95_um": float(np.percentile(magnitude, 95)),
        "displacement_max_um": float(np.max(magnitude)),
        "jacobian_min": float(np.min(jacobian)),
        "jacobian_p01": float(np.percentile(jacobian, 1)),
        "jacobian_p05": float(np.percentile(jacobian, 5)),
        "jacobian_median": float(np.median(jacobian)),
        "jacobian_p95": float(np.percentile(jacobian, 95)),
        "jacobian_p99": float(np.percentile(jacobian, 99)),
        "jacobian_max": float(np.max(jacobian)),
        "fold_over_fraction": float(np.mean(jacobian <= 0)),
        "fraction_jacobian_below_0_5": float(np.mean(jacobian < 0.5)),
        "fraction_jacobian_below_0_8": float(np.mean(jacobian < 0.8)),
        "fraction_jacobian_above_1_25": float(np.mean(jacobian > 1.25)),
        "fraction_jacobian_above_2_0": float(np.mean(jacobian > 2.0)),
        "mean_absolute_log_jacobian": float(np.mean(np.abs(np.log(protected)))),
        "rms_log_jacobian": float(np.sqrt(np.mean(np.log(protected) ** 2))),
    }
    for threshold in (1.0, 2.0, 5.0, 10.0):
        result[f"fraction_displacement_above_{threshold:g}_um"] = float(np.mean(magnitude > threshold))
    return result


def raster_fidelity_summary(
    inverse_solver_diagnostics: Mapping | None,
    *,
    point_raster_errors_pixels: np.ndarray | None = None,
) -> dict:
    diagnostics = inverse_solver_diagnostics or {}
    summary = {
        "inverse_solver_median_residual_pixels": diagnostics.get("median_residual_pixels"),
        "inverse_solver_p95_residual_pixels": diagnostics.get("p95_residual_pixels"),
        "inverse_solver_max_residual_pixels": diagnostics.get("max_residual_pixels"),
        "inverse_solver_converged": diagnostics.get("converged"),
        "fraction_residual_above_0_25_pixels": diagnostics.get("fraction_above_0_25_pixel"),
        "fraction_residual_above_0_5_pixels": diagnostics.get("fraction_above_0_5_pixel"),
        "fraction_residual_above_1_pixels": diagnostics.get("fraction_above_1_pixel"),
        "interpretation": "Raster implementation fidelity; pixel change does not establish registration accuracy.",
    }
    if point_raster_errors_pixels is not None:
        errors = np.asarray(point_raster_errors_pixels, dtype=float)
        if errors.size and np.isfinite(errors).all():
            summary["point_raster_consistency_median_pixels"] = float(np.median(errors))
            summary["point_raster_consistency_p95_pixels"] = float(np.percentile(errors, 95))
    return summary


def displacement_endpoint_error(
    estimated_x: np.ndarray,
    estimated_y: np.ndarray,
    true_x: np.ndarray,
    true_y: np.ndarray,
    *,
    direction_epsilon: float = 1e-6,
) -> tuple[pd.DataFrame, dict]:
    arrays = [np.asarray(value, dtype=float) for value in (estimated_x, estimated_y, true_x, true_y)]
    if len({value.shape for value in arrays}) != 1 or arrays[0].ndim != 2:
        raise ValueError("Estimated and true displacement fields must share one 2D shape.")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("Displacement fields must be finite.")
    ex, ey, tx, ty = arrays
    error = np.hypot(ex - tx, ey - ty)
    true_magnitude = np.hypot(tx, ty)
    estimated_magnitude = np.hypot(ex, ey)
    valid_direction = true_magnitude > direction_epsilon
    angle = np.full(error.shape, np.nan)
    if np.any(valid_direction):
        cosine = (ex * tx + ey * ty) / np.maximum(estimated_magnitude * true_magnitude, 1e-12)
        angle[valid_direction] = np.degrees(np.arccos(np.clip(cosine[valid_direction], -1.0, 1.0)))
    rows, cols = np.indices(error.shape)
    table = pd.DataFrame({
        "grid_row": rows.ravel(), "grid_col": cols.ravel(),
        "estimated_dx": ex.ravel(), "estimated_dy": ey.ravel(),
        "true_dx": tx.ravel(), "true_dy": ty.ravel(),
        "epe_um": error.ravel(),
        "directional_error_degrees": angle.ravel(),
        "magnitude_bias_um": (estimated_magnitude - true_magnitude).ravel(),
    })
    summary = {
        "epe_median_um": float(np.median(error)),
        "epe_mean_um": float(np.mean(error)),
        "epe_p90_um": float(np.percentile(error, 90)),
        "epe_p95_um": float(np.percentile(error, 95)),
        "epe_max_um": float(np.max(error)),
        "directional_error_median_degrees": float(np.nanmedian(angle)) if np.any(valid_direction) else None,
        "magnitude_bias_median_um": float(np.median(estimated_magnitude - true_magnitude)),
        "deformation_recovery_ratio": (
            float(np.percentile(estimated_magnitude, 95) / np.percentile(true_magnitude, 95))
            if np.percentile(true_magnitude, 95) > direction_epsilon else None
        ),
    }
    return table, summary


def synthetic_raster_metrics(reference: np.ndarray, recovered: np.ndarray) -> dict:
    first = np.asarray(reference, dtype=float)
    second = np.asarray(recovered, dtype=float)
    if first.shape != second.shape:
        raise ValueError("Synthetic reference and recovered raster must share shape.")
    difference = second - first
    mse = float(np.mean(difference**2))
    data_range = float(np.max(first) - np.min(first))
    gradient_first = np.hypot(sobel(first, axis=0), sobel(first, axis=1))
    gradient_second = np.hypot(sobel(second, axis=0), sobel(second, axis=1))
    result = {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(mse)),
        "psnr": float(20 * np.log10(data_range / np.sqrt(mse))) if mse > 0 and data_range > 0 else None,
        "edge_map_mae": float(np.mean(np.abs(gradient_second - gradient_first))),
        "ssim": None,
    }
    try:
        from skimage.metrics import structural_similarity

        if data_range > 0:
            channel_axis = -1 if first.ndim == 3 else None
            result["ssim"] = float(structural_similarity(first, second, data_range=data_range, channel_axis=channel_axis))
        elif np.array_equal(first, second):
            result["ssim"] = 1.0
    except (ImportError, ValueError):
        pass
    return result


def local_region_evaluation(
    fixed_points: np.ndarray,
    affine_points: np.ndarray,
    attempted_points: np.ndarray,
    applied_points: np.ndarray,
    attempted_displacement_x: np.ndarray,
    attempted_displacement_y: np.ndarray,
    applied_displacement_x: np.ndarray,
    applied_displacement_y: np.ndarray,
    *,
    bounds: tuple[float, float, float, float],
    spacing: float,
    block_size_um: float = 100.0,
    min_points: int = 5,
    unchanged_tolerance_um: float = 0.05,
) -> tuple[pd.DataFrame, dict]:
    fixed = _points(fixed_points, "fixed_points")
    stages = {
        "affine": _points(affine_points, "affine_points"),
        "attempted": _points(attempted_points, "attempted_points"),
        "applied": _points(applied_points, "applied_points"),
    }
    min_x, min_y, max_x, max_y = map(float, bounds)
    field_rows, field_cols = np.indices(np.asarray(attempted_displacement_x).shape)
    field_x_world = min_x + field_cols * spacing
    field_y_world = min_y + field_rows * spacing
    rows = []
    region_id = 0
    for y0 in np.arange(min_y, max_y, block_size_um):
        for x0 in np.arange(min_x, max_x, block_size_um):
            x1, y1 = min(x0 + block_size_um, max_x), min(y0 + block_size_um, max_y)
            fixed_mask = (fixed[:, 0] >= x0) & (fixed[:, 0] < x1) & (fixed[:, 1] >= y0) & (fixed[:, 1] < y1)
            stage_masks = {
                name: (values[:, 0] >= x0) & (values[:, 0] < x1) & (values[:, 1] >= y0) & (values[:, 1] < y1)
                for name, values in stages.items()
            }
            if np.count_nonzero(fixed_mask) < min_points or any(np.count_nonzero(mask) < min_points for mask in stage_masks.values()):
                continue
            local_fixed = fixed[fixed_mask]
            local_metrics = {
                name: pointset_stage_metrics(local_fixed, values[stage_masks[name]])
                for name, values in stages.items()
            }
            field_mask = (field_x_world >= x0) & (field_x_world <= x1) & (field_y_world >= y0) & (field_y_world <= y1)
            if not np.any(field_mask):
                continue
            attempted_mag = np.hypot(attempted_displacement_x, attempted_displacement_y)[field_mask]
            attempted_jac = _jacobian_array(attempted_displacement_x, attempted_displacement_y, spacing)[field_mask]
            row = {
                "region_id": region_id,
                "center_x": (x0 + x1) / 2, "center_y": (y0 + y1) / 2,
                "fixed_point_count": int(np.count_nonzero(fixed_mask)),
                "moving_point_count": int(np.count_nonzero(stage_masks["affine"])),
                "affine_symmetric_median": local_metrics["affine"]["symmetric_nn_median_um"],
                "attempted_symmetric_median": local_metrics["attempted"]["symmetric_nn_median_um"],
                "applied_symmetric_median": local_metrics["applied"]["symmetric_nn_median_um"],
                "delta_affine_to_attempted": local_metrics["attempted"]["symmetric_nn_median_um"] - local_metrics["affine"]["symmetric_nn_median_um"],
                "delta_affine_to_applied": local_metrics["applied"]["symmetric_nn_median_um"] - local_metrics["affine"]["symmetric_nn_median_um"],
                "mutual_nearest_affine": local_metrics["affine"]["mutual_nearest_fraction"],
                "mutual_nearest_attempted": local_metrics["attempted"]["mutual_nearest_fraction"],
                "mutual_nearest_applied": local_metrics["applied"]["mutual_nearest_fraction"],
                "displacement_median": float(np.median(attempted_mag)),
                "displacement_p95": float(np.percentile(attempted_mag, 95)),
                "jacobian_p05": float(np.percentile(attempted_jac, 5)),
                "jacobian_median": float(np.median(attempted_jac)),
                "jacobian_p95": float(np.percentile(attempted_jac, 95)),
            }
            for threshold in (3, 5, 10):
                row[f"within_{threshold}_um_affine"] = local_metrics["affine"][f"within_{threshold}_um_fraction"]
                row[f"within_{threshold}_um_attempted"] = local_metrics["attempted"][f"within_{threshold}_um_fraction"]
                row[f"within_{threshold}_um_applied"] = local_metrics["applied"][f"within_{threshold}_um_fraction"]
            rows.append(row)
            region_id += 1
    table = pd.DataFrame(rows)
    if table.empty:
        return table, {
            "n_valid_regions": 0, "fraction_improved": None, "fraction_unchanged": None,
            "fraction_worsened": None, "median_regional_improvement_um": None,
            "p10_regional_improvement_um": None, "p90_regional_improvement_um": None,
            "worst_degradation_um": None, "best_improvement_um": None,
        }
    delta = table["delta_affine_to_applied"].to_numpy()
    improvement = -delta
    return table, {
        "n_valid_regions": int(len(table)),
        "fraction_improved": float(np.mean(delta < -unchanged_tolerance_um)),
        "fraction_unchanged": float(np.mean(np.abs(delta) <= unchanged_tolerance_um)),
        "fraction_worsened": float(np.mean(delta > unchanged_tolerance_um)),
        "median_regional_improvement_um": float(np.median(improvement)),
        "p10_regional_improvement_um": float(np.percentile(improvement, 10)),
        "p90_regional_improvement_um": float(np.percentile(improvement, 90)),
        "worst_degradation_um": float(np.max(delta)),
        "best_improvement_um": float(np.max(improvement)),
    }


def _jacobian_array(dx: np.ndarray, dy: np.ndarray, spacing: float) -> np.ndarray:
    dfx_dy, dfx_dx = np.gradient(np.asarray(dx, dtype=float), spacing, spacing)
    dfy_dy, dfy_dx = np.gradient(np.asarray(dy, dtype=float), spacing, spacing)
    return (1 + dfx_dx) * (1 + dfy_dy) - dfx_dy * dfy_dx


def generate_method_scorecard(
    *,
    landmark_summary: Mapping | None,
    affine_internal_metrics: Mapping,
    applied_internal_metrics: Mapping,
    local_summary: Mapping,
    applied_deformation_metrics: Mapping,
    raster_fidelity: Mapping,
    fine_applied: bool,
    fine_rejected: bool = False,
    meaningful_improvement_um: float = 0.1,
) -> tuple[pd.DataFrame, dict]:
    deformation_pass = bool(
        not fine_rejected
        and
        applied_deformation_metrics.get("fold_over_fraction", 1.0) == 0.0
        and applied_deformation_metrics.get("jacobian_min", 0.0) > 0.0
    )
    internal_delta = (
        float(affine_internal_metrics["symmetric_nn_median_um"])
        - float(applied_internal_metrics["symmetric_nn_median_um"])
    )
    internal_status = "IMPROVED" if internal_delta >= meaningful_improvement_um else (
        "WORSE" if internal_delta <= -meaningful_improvement_um else "UNCHANGED"
    )
    if landmark_summary is None:
        independent_status = "NOT AVAILABLE"
        landmark_improved = False
    else:
        landmark_improved = float(landmark_summary["absolute_median_improvement_um"]) >= meaningful_improvement_um
        independent_status = "PASS" if landmark_improved and deformation_pass else "REVIEW"
    local_status = "PASS" if (
        local_summary.get("fraction_worsened") is not None
        and float(local_summary["fraction_worsened"]) <= 0.25
    ) else "REVIEW"
    raster_status = "PASS" if raster_fidelity.get("inverse_solver_converged") is True else "REVIEW"
    rows = [
        {"domain": "Independent landmark validation", "status": independent_status},
        {"domain": "Internal point-set metric", "status": internal_status},
        {"domain": "Local consistency", "status": local_status},
        {"domain": "Deformation safety", "status": "PASS" if deformation_pass else "FAIL"},
        {"domain": "Raster implementation fidelity", "status": raster_status},
    ]
    if not deformation_pass or (not fine_applied and internal_status == "WORSE"):
        overall = "UNSAFE / REJECTED"
    elif landmark_improved and deformation_pass:
        overall = "VALIDATED IMPROVEMENT"
    elif landmark_summary is None and internal_status == "IMPROVED" and deformation_pass:
        overall = "INTERNAL IMPROVEMENT ONLY"
    else:
        overall = "NO MEANINGFUL IMPROVEMENT"
    return pd.DataFrame(rows), {
        "overall_status": overall,
        "independent_validation_available": landmark_summary is not None,
        "evaluation_version": EVALUATION_VERSION,
        "metric_definitions_version": METRIC_DEFINITIONS_VERSION,
        "warning": "Internal nearest-neighbor improvement is not ground-truth accuracy validation.",
    }
