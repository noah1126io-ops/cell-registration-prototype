from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter, laplace, map_coordinates
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

from src.pointset_registration import FineWarpResult, point_bidirectional_distance_metrics


def _validate_points(points: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) == 0:
        raise ValueError(f"{name} must have shape (n, 2) and contain at least one point.")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite coordinates.")
    return values


def _centered_bidirectional_median(fixed: np.ndarray, moving: np.ndarray) -> float:
    fixed_centered = fixed - np.median(fixed, axis=0)
    moving_centered = moving - np.median(moving, axis=0)
    moving_to_fixed = cKDTree(fixed_centered).query(moving_centered, k=1)[0]
    fixed_to_moving = cKDTree(moving_centered).query(fixed_centered, k=1)[0]
    return float(np.median([np.median(moving_to_fixed), np.median(fixed_to_moving)]))


def detect_xy_reversal(
    fixed_points: np.ndarray,
    moving_points: np.ndarray,
    *,
    improvement_ratio: float = 0.5,
    minimum_direct_error: float = 1.0,
) -> dict[str, float | bool]:
    """Flag a likely x/y column reversal using centered point-cloud geometry."""
    fixed = _validate_points(fixed_points, "fixed_points")
    moving = _validate_points(moving_points, "moving_points")
    direct = _centered_bidirectional_median(fixed, moving)
    reversed_score = _centered_bidirectional_median(fixed, moving[:, ::-1])
    detected = bool(direct > minimum_direct_error and reversed_score < direct * improvement_ratio)
    return {
        "detected": detected,
        "direct_centered_median": direct,
        "reversed_centered_median": reversed_score,
    }


def _make_grid(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    pixel_size: float,
    padding: float,
    bounds: tuple[float, float, float, float] | None,
    max_grid_side: int,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    if bounds is None:
        combined = np.vstack([fixed, moving])
        minimum = np.min(combined, axis=0) - padding
        maximum = np.max(combined, axis=0) + padding
        bounds = (float(minimum[0]), float(minimum[1]), float(maximum[0]), float(maximum[1]))
    min_x, min_y, max_x, max_y = map(float, bounds)
    if not (max_x > min_x and max_y > min_y):
        raise ValueError("bounds must have positive width and height.")
    width = int(np.ceil((max_x - min_x) / pixel_size)) + 1
    height = int(np.ceil((max_y - min_y) / pixel_size)) + 1
    if width > max_grid_side or height > max_grid_side:
        raise ValueError(
            f"Density-flow grid would be {width} x {height}; increase density pixel size "
            f"or crop the coordinate range (maximum side {max_grid_side})."
        )
    xs = min_x + np.arange(width, dtype=float) * pixel_size
    ys = min_y + np.arange(height, dtype=float) * pixel_size
    grid_x, grid_y = np.meshgrid(xs, ys)
    return grid_x, grid_y, (min_x, min_y, max_x, max_y)


def _rasterize_points(
    points: np.ndarray,
    shape: tuple[int, int],
    bounds: tuple[float, float, float, float],
    pixel_size: float,
) -> np.ndarray:
    min_x, min_y, _, _ = bounds
    cols = (points[:, 0] - min_x) / pixel_size
    rows = (points[:, 1] - min_y) / pixel_size
    row0 = np.floor(rows).astype(int)
    col0 = np.floor(cols).astype(int)
    row_fraction = rows - row0
    col_fraction = cols - col0
    raster = np.zeros(shape, dtype=float)
    for row_offset, col_offset, weight in (
        (0, 0, (1.0 - row_fraction) * (1.0 - col_fraction)),
        (0, 1, (1.0 - row_fraction) * col_fraction),
        (1, 0, row_fraction * (1.0 - col_fraction)),
        (1, 1, row_fraction * col_fraction),
    ):
        target_rows = row0 + row_offset
        target_cols = col0 + col_offset
        valid = (
            (target_rows >= 0)
            & (target_rows < shape[0])
            & (target_cols >= 0)
            & (target_cols < shape[1])
        )
        np.add.at(raster, (target_rows[valid], target_cols[valid]), weight[valid])
    return raster


def _normalized_density(impulses: np.ndarray, sigma_px: float) -> np.ndarray:
    density = gaussian_filter(impulses, sigma=max(float(sigma_px), 0.01), mode="constant")
    total = float(np.sum(density))
    return density / total if total > 0 else density


def _sample_field(
    points: np.ndarray,
    field_x: np.ndarray,
    field_y: np.ndarray,
    bounds: tuple[float, float, float, float],
    pixel_size: float,
    *,
    mode: str = "constant",
) -> np.ndarray:
    min_x, min_y, _, _ = bounds
    cols = (points[:, 0] - min_x) / pixel_size
    rows = (points[:, 1] - min_y) / pixel_size
    coordinates = np.vstack([rows, cols])
    sampled_x = map_coordinates(field_x, coordinates, order=1, mode=mode, cval=0.0)
    sampled_y = map_coordinates(field_y, coordinates, order=1, mode=mode, cval=0.0)
    return np.column_stack([sampled_x, sampled_y])


def _output_world_grid_from_metadata(
    image_shape: tuple[int, ...],
    warp_metadata: dict,
) -> np.ndarray:
    height, width = image_shape[:2]
    if int(warp_metadata.get("height", height)) != height or int(warp_metadata.get("width", width)) != width:
        raise ValueError("warp_metadata width/height must match affine_image shape.")
    pixel_size = float(warp_metadata["output_pixel_size_um"])
    if pixel_size <= 0 or not np.isfinite(pixel_size):
        raise ValueError("warp_metadata output_pixel_size_um must be positive and finite.")
    origin = warp_metadata.get("output_origin", "upper-left")
    if origin not in {"upper-left", "upper-right", "lower-left"}:
        raise ValueError("Unsupported output_origin in warp_metadata.")
    col0_world_x = float(warp_metadata["col0_world_x"])
    row0_world_y = float(warp_metadata["row0_world_y"])

    columns = np.arange(width, dtype=float)
    rows = np.arange(height, dtype=float)
    world_x = (
        col0_world_x - columns * pixel_size
        if origin == "upper-right"
        else col0_world_x + columns * pixel_size
    )
    world_y = (
        row0_world_y + rows * pixel_size
        if origin in {"upper-left", "upper-right"}
        else row0_world_y - rows * pixel_size
    )
    grid_x, grid_y = np.meshgrid(world_x, world_y)
    return np.column_stack([grid_x.ravel(), grid_y.ravel()])


def _world_to_affine_image_rows_cols(world_points: np.ndarray, warp_metadata: dict) -> np.ndarray:
    pixel_size = float(warp_metadata["output_pixel_size_um"])
    origin = warp_metadata.get("output_origin", "upper-left")
    col0_world_x = float(warp_metadata["col0_world_x"])
    row0_world_y = float(warp_metadata["row0_world_y"])
    if origin == "upper-right":
        columns = (col0_world_x - world_points[:, 0]) / pixel_size
    else:
        columns = (world_points[:, 0] - col0_world_x) / pixel_size
    if origin in {"upper-left", "upper-right"}:
        rows = (world_points[:, 1] - row0_world_y) / pixel_size
    else:
        rows = (row0_world_y - world_points[:, 1]) / pixel_size
    return np.column_stack([rows, columns])


def warp_affine_image_with_density_flow(
    affine_image: np.ndarray,
    warp_metadata: dict,
    displacement_x: np.ndarray,
    displacement_y: np.ndarray,
    *,
    field_bounds: tuple[float, float, float, float],
    field_spacing: float,
    inverse_iterations: int = 12,
) -> np.ndarray:
    """Inverse-map an affine world image through a forward density-flow field."""
    image = np.asarray(affine_image)
    if image.ndim not in {2, 3}:
        raise ValueError("affine_image must be a 2D or 3D array.")
    field_x = np.asarray(displacement_x, dtype=float)
    field_y = np.asarray(displacement_y, dtype=float)
    if field_x.shape != field_y.shape or field_x.ndim != 2:
        raise ValueError("displacement_x and displacement_y must be matching 2D arrays.")
    if not np.isfinite(field_x).all() or not np.isfinite(field_y).all():
        raise ValueError("Density-flow displacement fields must be finite.")
    if field_spacing <= 0 or not np.isfinite(field_spacing):
        raise ValueError("field_spacing must be positive and finite.")
    if inverse_iterations < 1:
        raise ValueError("inverse_iterations must be at least 1.")

    output_world_points = _output_world_grid_from_metadata(image.shape, warp_metadata)
    source_world_points = output_world_points.copy()
    for _ in range(int(inverse_iterations)):
        sampled_displacement = _sample_field(
            source_world_points,
            field_x,
            field_y,
            field_bounds,
            field_spacing,
            mode="nearest",
        )
        source_world_points = output_world_points - sampled_displacement

    source_rows_cols = _world_to_affine_image_rows_cols(source_world_points, warp_metadata)
    rows = source_rows_cols[:, 0]
    columns = source_rows_cols[:, 1]
    if image.ndim == 2:
        sampled = map_coordinates(
            image.astype(float),
            [rows, columns],
            order=1,
            mode="constant",
            cval=0.0,
        ).reshape(image.shape)
    else:
        sampled_channels = [
            map_coordinates(
                image[..., channel].astype(float),
                [rows, columns],
                order=1,
                mode="constant",
                cval=0.0,
            ).reshape(image.shape[:2])
            for channel in range(image.shape[2])
        ]
        sampled = np.stack(sampled_channels, axis=2)

    if np.issubdtype(image.dtype, np.integer):
        limits = np.iinfo(image.dtype)
        return np.clip(sampled, limits.min, limits.max).astype(image.dtype)
    return sampled.astype(image.dtype, copy=False)


def density_flow_image_outputs(
    affine_image: np.ndarray,
    warp_metadata: dict,
    fine_result: FineWarpResult,
    *,
    inverse_iterations: int = 12,
) -> dict[str, np.ndarray]:
    """Return affine, attempted, and safety-gated final density-flow images."""
    affine_output = np.asarray(affine_image).copy()
    attempted_x = getattr(fine_result, "attempted_displacement_x", None)
    attempted_y = getattr(fine_result, "attempted_displacement_y", None)
    if attempted_x is None or attempted_y is None:
        attempted_x = np.zeros_like(fine_result.displacement_x, dtype=float)
        attempted_y = np.zeros_like(fine_result.displacement_y, dtype=float)
    attempted_output = warp_affine_image_with_density_flow(
        affine_output,
        warp_metadata,
        attempted_x,
        attempted_y,
        field_bounds=fine_result.bounds,
        field_spacing=fine_result.grid_spacing,
        inverse_iterations=inverse_iterations,
    )
    if bool(getattr(fine_result, "applied", False)):
        final_output = warp_affine_image_with_density_flow(
            affine_output,
            warp_metadata,
            fine_result.displacement_x,
            fine_result.displacement_y,
            field_bounds=fine_result.bounds,
            field_spacing=fine_result.grid_spacing,
            inverse_iterations=inverse_iterations,
        )
    else:
        final_output = affine_output.copy()
    return {
        "affine": affine_output,
        "attempted": attempted_output,
        "final": final_output,
    }


def _compose_fields(
    field_x: np.ndarray,
    field_y: np.ndarray,
    update_x: np.ndarray,
    update_y: np.ndarray,
    pixel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.indices(field_x.shape, dtype=float)
    mapped_rows = rows + field_y / pixel_size
    mapped_cols = cols + field_x / pixel_size
    composed_update_x = map_coordinates(update_x, [mapped_rows, mapped_cols], order=1, mode="nearest")
    composed_update_y = map_coordinates(update_y, [mapped_rows, mapped_cols], order=1, mode="nearest")
    return field_x + composed_update_x, field_y + composed_update_y


def _jacobian(field_x: np.ndarray, field_y: np.ndarray, pixel_size: float) -> np.ndarray:
    dfx_dy, dfx_dx = np.gradient(field_x, pixel_size, pixel_size)
    dfy_dy, dfy_dx = np.gradient(field_y, pixel_size, pixel_size)
    return (1.0 + dfx_dx) * (1.0 + dfy_dy) - dfx_dy * dfy_dx


def density_flow_deformation_diagnostics(
    field_x: np.ndarray,
    field_y: np.ndarray,
    *,
    pixel_size: float = 1.0,
) -> dict[str, float | bool | np.ndarray]:
    """Separate a displacement field's median translation from local variation."""
    dx = np.asarray(field_x, dtype=float)
    dy = np.asarray(field_y, dtype=float)
    if dx.shape != dy.shape or dx.ndim != 2:
        raise ValueError("field_x and field_y must be matching 2D arrays.")
    if not np.isfinite(dx).all() or not np.isfinite(dy).all():
        raise ValueError("Displacement fields must contain only finite values.")
    if pixel_size <= 0 or not np.isfinite(pixel_size):
        raise ValueError("pixel_size must be positive and finite.")

    median_dx = float(np.median(dx))
    median_dy = float(np.median(dy))
    raw_magnitude = np.hypot(dx, dy)
    local_x = dx - median_dx
    local_y = dy - median_dy
    local_magnitude = np.hypot(local_x, local_y)
    jacobian = _jacobian(dx, dy, float(pixel_size))
    raw_p95 = float(np.percentile(raw_magnitude, 95))
    local_p95 = float(np.percentile(local_magnitude, 95))
    translation_dominated = bool(
        raw_p95 > np.finfo(float).eps
        and local_p95 < max(1.0, 0.15 * raw_p95)
    )
    return {
        "raw_displacement_median": float(np.median(raw_magnitude)),
        "raw_displacement_p95": raw_p95,
        "raw_displacement_max": float(np.max(raw_magnitude)),
        "median_displacement_x": median_dx,
        "median_displacement_y": median_dy,
        "local_residual_median": float(np.median(local_magnitude)),
        "local_residual_p95": local_p95,
        "local_residual_max": float(np.max(local_magnitude)),
        "displacement_std_x": float(np.std(dx)),
        "displacement_std_y": float(np.std(dy)),
        "fraction_local_residual_above_1_um": float(np.mean(local_magnitude > 1.0)),
        "fraction_local_residual_above_2_um": float(np.mean(local_magnitude > 2.0)),
        "fraction_local_residual_above_5_um": float(np.mean(local_magnitude > 5.0)),
        "jacobian_p05": float(np.percentile(jacobian, 5)),
        "jacobian_median": float(np.median(jacobian)),
        "jacobian_p95": float(np.percentile(jacobian, 95)),
        "translation_dominated": translation_dominated,
        "raw_magnitude": raw_magnitude,
        "local_residual_x": local_x,
        "local_residual_y": local_y,
        "local_residual_magnitude": local_magnitude,
        "jacobian": jacobian,
    }


def _phase_translation(fixed_density: np.ndarray, moving_density: np.ndarray, pixel_size: float) -> np.ndarray:
    correlation = fftconvolve(fixed_density, moving_density[::-1, ::-1], mode="full")
    peak_row, peak_col = np.unravel_index(np.argmax(correlation), correlation.shape)
    lag_row = peak_row - (moving_density.shape[0] - 1)
    lag_col = peak_col - (moving_density.shape[1] - 1)
    return np.array([lag_col * pixel_size, lag_row * pixel_size], dtype=float)


def _best_global_shift(
    fixed: np.ndarray,
    moving: np.ndarray,
    fixed_density: np.ndarray,
    moving_density: np.ndarray,
    pixel_size: float,
) -> np.ndarray:
    candidates = [
        np.zeros(2, dtype=float),
        np.median(fixed, axis=0) - np.median(moving, axis=0),
        _phase_translation(fixed_density, moving_density, pixel_size),
    ]
    best_shift = candidates[0]
    best_score = np.inf
    for candidate in candidates:
        metrics = point_bidirectional_distance_metrics(fixed, moving + candidate)
        score = float(metrics["symmetric_median_distance"])
        if score < best_score:
            best_score = score
            best_shift = candidate
    return np.asarray(best_shift, dtype=float)


def _mutual_nearest_fraction(fixed: np.ndarray, moving: np.ndarray) -> float:
    fixed_tree = cKDTree(fixed)
    moving_tree = cKDTree(moving)
    _, moving_to_fixed = fixed_tree.query(moving, k=1)
    _, fixed_to_moving = moving_tree.query(fixed, k=1)
    mutual = sum(fixed_to_moving[int(fixed_index)] == moving_index for moving_index, fixed_index in enumerate(moving_to_fixed))
    return float(2.0 * mutual / max(len(fixed) + len(moving), 1))


def _field_objective(
    fixed_density: np.ndarray,
    moving_density: np.ndarray,
    field_x: np.ndarray,
    field_y: np.ndarray,
    tissue_weight: np.ndarray,
    jacobian: np.ndarray,
    *,
    smoothness_weight: float,
    magnitude_weight: float,
    jacobian_weight: float,
    boundary_weight: float,
    inverse_consistency_weight: float,
    jacobian_min_threshold: float,
    jacobian_max_threshold: float,
    pixel_size: float,
) -> dict[str, float]:
    residual = fixed_density - moving_density
    density_energy = float(np.mean(tissue_weight * fixed_density**2))
    density_term = float(
        np.mean(tissue_weight * residual**2) / max(density_energy, np.finfo(float).eps)
    )
    dfx_dy, dfx_dx = np.gradient(field_x, pixel_size, pixel_size)
    dfy_dy, dfy_dx = np.gradient(field_y, pixel_size, pixel_size)
    smoothness_term = float(np.mean(dfx_dx**2 + dfx_dy**2 + dfy_dx**2 + dfy_dy**2))
    magnitude_term = float(np.mean(field_x**2 + field_y**2))
    jacobian_term = float(
        np.mean(
            np.maximum(jacobian_min_threshold - jacobian, 0.0) ** 2
            + np.maximum(jacobian - jacobian_max_threshold, 0.0) ** 2
        )
    )
    boundary_term = float(np.mean((1.0 - tissue_weight) * (field_x**2 + field_y**2)))
    rows, cols = np.indices(field_x.shape, dtype=float)
    mapped_rows = rows + field_y / pixel_size
    mapped_cols = cols + field_x / pixel_size
    mapped_x = map_coordinates(field_x, [mapped_rows, mapped_cols], order=1, mode="nearest")
    mapped_y = map_coordinates(field_y, [mapped_rows, mapped_cols], order=1, mode="nearest")
    inverse_term = float(np.mean((field_x - mapped_x) ** 2 + (field_y - mapped_y) ** 2))
    total = (
        density_term
        + smoothness_weight * smoothness_term
        + magnitude_weight * magnitude_term
        + jacobian_weight * jacobian_term
        + boundary_weight * boundary_term
        + inverse_consistency_weight * inverse_term
    )
    return {
        "total": float(total),
        "density": density_term,
        "smoothness": smoothness_term,
        "magnitude": magnitude_term,
        "jacobian_barrier": jacobian_term,
        "tissue_boundary": boundary_term,
        "inverse_consistency": inverse_term,
    }


def _evaluate_density_flow_state(
    moving_points: np.ndarray,
    field_x: np.ndarray,
    field_y: np.ndarray,
    *,
    fixed_density: np.ndarray,
    density_sigma: float,
    shape: tuple[int, int],
    bounds: tuple[float, float, float, float],
    pixel_size: float,
    tissue_weight: np.ndarray,
    smoothness_weight: float,
    magnitude_weight: float,
    jacobian_weight: float,
    boundary_weight: float,
    inverse_consistency_weight: float,
    jacobian_min_threshold: float,
    jacobian_max_threshold: float,
) -> dict[str, object]:
    """Evaluate density and field penalties from one internally consistent field."""
    transformed_points = moving_points + _sample_field(
        moving_points, field_x, field_y, bounds, pixel_size
    )
    moving_impulses = _rasterize_points(transformed_points, shape, bounds, pixel_size)
    moving_density = _normalized_density(moving_impulses, density_sigma)
    jacobian = _jacobian(field_x, field_y, pixel_size)
    displacement = np.hypot(field_x, field_y)
    objective = _field_objective(
        fixed_density,
        moving_density,
        field_x,
        field_y,
        tissue_weight,
        jacobian,
        smoothness_weight=smoothness_weight,
        magnitude_weight=magnitude_weight,
        jacobian_weight=jacobian_weight,
        boundary_weight=boundary_weight,
        inverse_consistency_weight=inverse_consistency_weight,
        jacobian_min_threshold=jacobian_min_threshold,
        jacobian_max_threshold=jacobian_max_threshold,
        pixel_size=pixel_size,
    )
    return {
        "transformed_points": transformed_points,
        "moving_density": moving_density,
        "jacobian": jacobian,
        "displacement": displacement,
        "objective": objective,
        "finite": bool(
            np.isfinite(transformed_points).all()
            and np.isfinite(moving_density).all()
            and np.isfinite(field_x).all()
            and np.isfinite(field_y).all()
            and np.isfinite(jacobian).all()
            and np.isfinite(displacement).all()
            and np.isfinite(float(objective["total"]))
        ),
        "jacobian_min": float(np.min(jacobian)),
        "jacobian_max": float(np.max(jacobian)),
        "max_displacement": float(np.max(displacement)),
        "p95_displacement": float(np.percentile(displacement, 95)),
    }


def _density_flow_trial_is_safe(
    state: dict[str, object],
    *,
    jacobian_min_threshold: float,
    jacobian_max_threshold: float,
    max_displacement: float,
    displacement_p95_limit: float | None,
) -> tuple[bool, str | None]:
    """Apply the same hard limits used for trial and final density-flow fields."""
    if not bool(state["finite"]):
        return False, "non_finite_density_flow_output"
    if float(state["jacobian_min"]) <= 0.0:
        return False, "jacobian_foldover"
    if float(state["jacobian_min"]) < jacobian_min_threshold:
        return False, "jacobian_min_below_limit"
    if float(state["jacobian_max"]) > jacobian_max_threshold:
        return False, "jacobian_max_above_limit"
    if float(state["max_displacement"]) > max_displacement:
        return False, "max_displacement_too_large"
    if (
        displacement_p95_limit is not None
        and float(state["p95_displacement"]) > displacement_p95_limit
    ):
        return False, "displacement_p95_too_large"
    return True, None


def _density_flow_trial_is_acceptable(
    state: dict[str, object],
    *,
    current_objective: float,
    objective_tolerance: float,
    jacobian_min_threshold: float,
    jacobian_max_threshold: float,
    max_displacement: float,
    displacement_p95_limit: float | None,
) -> tuple[bool, str | None]:
    safe, reason = _density_flow_trial_is_safe(
        state,
        jacobian_min_threshold=jacobian_min_threshold,
        jacobian_max_threshold=jacobian_max_threshold,
        max_displacement=max_displacement,
        displacement_p95_limit=displacement_p95_limit,
    )
    if not safe:
        return False, reason
    required_decrease = max(float(objective_tolerance), abs(current_objective) * 1e-8)
    if float(state["objective"]["total"]) > current_objective - required_decrease:
        return False, "objective_not_decreased"
    return True, None


def _checkpoint_improves_affine(
    metrics: dict,
    *,
    affine_median: float,
    affine_mutual_fraction: float,
) -> bool:
    median = float(metrics["symmetric_median_distance"])
    mutual = float(metrics["mutual_nearest_fraction"])
    return bool(
        median <= affine_median + 1e-12
        and (median < affine_median - 1e-12 or mutual > affine_mutual_fraction + 1e-12)
    )


def _better_density_flow_checkpoint(candidate: dict, current: dict | None) -> bool:
    """Rank safe checkpoints by point median, then mutual fraction, then objective."""
    if current is None:
        return True
    candidate_metrics = candidate["metrics"]
    current_metrics = current["metrics"]
    candidate_median = float(candidate_metrics["symmetric_median_distance"])
    current_median = float(current_metrics["symmetric_median_distance"])
    if candidate_median < current_median - 1e-12:
        return True
    if not np.isclose(candidate_median, current_median):
        return False
    candidate_mutual = float(candidate_metrics["mutual_nearest_fraction"])
    current_mutual = float(current_metrics["mutual_nearest_fraction"])
    if candidate_mutual > current_mutual + 1e-12:
        return True
    if not np.isclose(candidate_mutual, current_mutual):
        return False
    return float(candidate.get("objective", np.inf)) < float(current.get("objective", np.inf))


def tissue_aware_density_flow_registration(
    fixed_points: np.ndarray,
    moving_points: np.ndarray,
    *,
    success_metric_fixed_points: np.ndarray | None = None,
    success_metric_moving_points: np.ndarray | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    density_pixel_size: float = 2.0,
    density_blur_scales: Sequence[float] = (8.0, 4.0, 2.0),
    optimization_levels: int = 3,
    iterations_per_level: int = 12,
    learning_rate: float = 0.2,
    update_smoothing_sigma: float = 3.0,
    smoothness_weight: float = 0.05,
    magnitude_weight: float = 0.0005,
    jacobian_barrier_weight: float = 1.0,
    tissue_boundary_weight: float = 0.02,
    inverse_consistency_weight: float = 0.0,
    jacobian_min_threshold: float = 0.05,
    jacobian_max_threshold: float = 4.0,
    max_displacement: float = 35.0,
    displacement_p95_limit: float | None = 30.0,
    global_translation_initialization: str = "off",
    objective_tolerance: float = 1e-12,
    early_stopping_patience: int = 3,
    point_metric_patience: int = 6,
    max_backtracking_steps: int = 8,
    detect_axis_reversal: bool = True,
    max_grid_side: int = 1024,
) -> FineWarpResult:
    """Independently estimate a tissue-weighted multiscale density-flow point warp."""
    fixed = _validate_points(fixed_points, "fixed_points")
    moving = _validate_points(moving_points, "moving_points")
    metric_fixed = fixed if success_metric_fixed_points is None else _validate_points(success_metric_fixed_points, "success_metric_fixed_points")
    metric_moving = moving if success_metric_moving_points is None else _validate_points(success_metric_moving_points, "success_metric_moving_points")
    if density_pixel_size <= 0 or not np.isfinite(density_pixel_size):
        raise ValueError("density_pixel_size must be positive and finite.")
    scales = tuple(float(scale) for scale in density_blur_scales)
    if not scales or any(scale <= 0 or not np.isfinite(scale) for scale in scales):
        raise ValueError("density_blur_scales must contain positive finite values.")
    if optimization_levels < 1 or iterations_per_level < 1:
        raise ValueError("optimization_levels and iterations_per_level must be at least 1.")
    if max_displacement <= 0 or not np.isfinite(max_displacement):
        raise ValueError("max_displacement must be positive and finite.")
    if objective_tolerance < 0 or not np.isfinite(objective_tolerance):
        raise ValueError("objective_tolerance must be finite and non-negative.")
    if early_stopping_patience < 1 or point_metric_patience < 1 or max_backtracking_steps < 1:
        raise ValueError("Density-flow patience and backtracking values must be at least 1.")
    global_initialization = str(global_translation_initialization).strip().lower()
    if global_initialization not in {"off", "auto"}:
        raise ValueError("global_translation_initialization must be 'off' or 'auto'.")

    active_scales = tuple(sorted(scales, reverse=True)[: min(int(optimization_levels), len(scales))])
    padding = max(active_scales) * density_pixel_size * 3.0
    grid_x, grid_y, resolved_bounds = _make_grid(
        fixed,
        moving,
        pixel_size=float(density_pixel_size),
        padding=padding,
        bounds=bounds,
        max_grid_side=int(max_grid_side),
    )
    shape = grid_x.shape
    zeros = np.zeros(shape, dtype=float)
    before_metrics = point_bidirectional_distance_metrics(metric_fixed, metric_moving)
    before_mutual = _mutual_nearest_fraction(metric_fixed, metric_moving)

    reversal = detect_xy_reversal(metric_fixed, metric_moving)
    if detect_axis_reversal and bool(reversal["detected"]):
        metrics = {
            **before_metrics,
            "mutual_nearest_fraction": before_mutual,
            "possible_xy_reversal": True,
            "xy_reversal_diagnostics": reversal,
        }
        return FineWarpResult(
            transformed_points=moving.copy(),
            grid_x=grid_x,
            grid_y=grid_y,
            displacement_x=zeros,
            displacement_y=zeros,
            bounds=resolved_bounds,
            grid_spacing=float(density_pixel_size),
            jacobian_min=1.0,
            jacobian_max=1.0,
            max_displacement=0.0,
            n_candidate_pairs=0,
            n_pairs=0,
            n_filtered_pairs=0,
            median_pair_distance_before=before_metrics["median_distance"],
            median_pair_distance_after=before_metrics["median_distance"],
            success=False,
            message="Density-flow registration was not attempted because x/y reversal is likely.",
            attempted_transformed_points=moving.copy(),
            attempted_displacement_x=zeros,
            attempted_displacement_y=zeros,
            attempted_metrics=metrics,
            applied_metrics=metrics,
            rejection_reason="possible_xy_reversal",
            applied=False,
            metrics={"before": metrics, "attempted": metrics, "applied": metrics},
        )

    fixed_impulses = _rasterize_points(fixed, shape, resolved_bounds, density_pixel_size)
    support_density = gaussian_filter(fixed_impulses, sigma=max(active_scales) * 1.5, mode="constant")
    support_threshold = max(float(np.max(support_density)) * 0.02, np.finfo(float).eps)
    tissue_mask = support_density >= support_threshold
    boundary_distance = distance_transform_edt(tissue_mask)
    boundary_confidence = np.clip(boundary_distance / max(max(active_scales), 1.0), 0.0, 1.0)
    tissue_weight_map = tissue_mask.astype(float) * (0.2 + 0.8 * boundary_confidence)

    field_x = zeros.copy()
    field_y = zeros.copy()
    original_moving = moving.copy()
    history: list[dict[str, float | int | bool | str | None]] = []

    coarse_fixed = _normalized_density(fixed_impulses, active_scales[0])
    coarse_moving = _normalized_density(
        _rasterize_points(moving, shape, resolved_bounds, density_pixel_size),
        active_scales[0],
    )
    global_shift = np.zeros(2, dtype=float)
    if global_initialization == "auto":
        global_shift = _best_global_shift(
            metric_fixed,
            metric_moving,
            coarse_fixed,
            coarse_moving,
            density_pixel_size,
        )
        global_magnitude = float(np.linalg.norm(global_shift))
        initial_limit = min(
            max_displacement,
            displacement_p95_limit if displacement_p95_limit is not None else max_displacement,
        )
        if global_magnitude > initial_limit:
            global_shift *= initial_limit / global_magnitude
        field_x.fill(global_shift[0])
        field_y.fill(global_shift[1])
    initial_field_x = field_x.copy()
    initial_field_y = field_y.copy()
    def evaluate_state(current_x: np.ndarray, current_y: np.ndarray, sigma_px: float) -> dict[str, object]:
        return _evaluate_density_flow_state(
            original_moving,
            current_x,
            current_y,
            fixed_density=_normalized_density(fixed_impulses, sigma_px),
            density_sigma=sigma_px,
            shape=shape,
            bounds=resolved_bounds,
            pixel_size=density_pixel_size,
            tissue_weight=tissue_weight_map,
            smoothness_weight=smoothness_weight,
            magnitude_weight=magnitude_weight,
            jacobian_weight=jacobian_barrier_weight,
            boundary_weight=tissue_boundary_weight,
            inverse_consistency_weight=inverse_consistency_weight,
            jacobian_min_threshold=jacobian_min_threshold,
            jacobian_max_threshold=jacobian_max_threshold,
        )

    def point_metrics_for_field(current_x: np.ndarray, current_y: np.ndarray) -> tuple[dict, np.ndarray]:
        transformed = metric_moving + _sample_field(
            metric_moving, current_x, current_y, resolved_bounds, density_pixel_size
        )
        values = point_bidirectional_distance_metrics(metric_fixed, transformed)
        values["mutual_nearest_fraction"] = _mutual_nearest_fraction(metric_fixed, transformed)
        return values, transformed

    def improves_affine(values: dict) -> bool:
        return _checkpoint_improves_affine(
            values,
            affine_median=float(before_metrics["symmetric_median_distance"]),
            affine_mutual_fraction=before_mutual,
        )

    attempted_steps = 0
    accepted_steps = 0
    backtracked_steps = 0
    rejected_steps = 0
    global_iteration = 0
    best_checkpoint: dict[str, object] | None = None
    best_attempted: dict[str, object] | None = None
    best_observed_median = float(before_metrics["symmetric_median_distance"])
    best_observed_mutual = before_mutual
    initial_objective: float | None = None
    consecutive_failed_updates = 0
    point_stall_count = 0

    initial_metric_values, _ = point_metrics_for_field(field_x, field_y)
    if improves_affine(initial_metric_values):
        best_checkpoint = {
            "field_x": field_x.copy(),
            "field_y": field_y.copy(),
            "metrics": initial_metric_values,
            "iteration": -1,
        }

    for level, sigma_px in enumerate(active_scales):
        fixed_density = _normalized_density(fixed_impulses, sigma_px)
        current_state = evaluate_state(field_x, field_y, sigma_px)
        if initial_objective is None:
            initial_objective = float(current_state["objective"]["total"])
        objective_stall_count = 0
        consecutive_failed_updates = 0
        for iteration in range(int(iterations_per_level)):
            attempted_steps += 1
            global_iteration += 1
            moving_density = np.asarray(current_state["moving_density"])
            residual = fixed_density - moving_density
            fixed_grad_y, fixed_grad_x = np.gradient(fixed_density)
            moving_grad_y, moving_grad_x = np.gradient(moving_density)
            grad_x = fixed_grad_x + moving_grad_x
            grad_y = fixed_grad_y + moving_grad_y
            denominator = grad_x**2 + grad_y**2 + 0.1 * residual**2 + 1e-15
            update_x = -learning_rate * density_pixel_size * residual * grad_x / denominator
            update_y = -learning_rate * density_pixel_size * residual * grad_y / denominator
            update_x *= tissue_weight_map
            update_y *= tissue_weight_map
            update_x += learning_rate * smoothness_weight * laplace(field_x, mode="nearest")
            update_y += learning_rate * smoothness_weight * laplace(field_y, mode="nearest")
            update_x -= learning_rate * magnitude_weight * field_x
            update_y -= learning_rate * magnitude_weight * field_y
            update_x -= learning_rate * tissue_boundary_weight * (1.0 - tissue_weight_map) * field_x
            update_y -= learning_rate * tissue_boundary_weight * (1.0 - tissue_weight_map) * field_y
            if inverse_consistency_weight > 0:
                grid_rows, grid_cols = np.indices(field_x.shape, dtype=float)
                mapped_rows = grid_rows + field_y / density_pixel_size
                mapped_cols = grid_cols + field_x / density_pixel_size
                mapped_field_x = map_coordinates(field_x, [mapped_rows, mapped_cols], order=1, mode="nearest")
                mapped_field_y = map_coordinates(field_y, [mapped_rows, mapped_cols], order=1, mode="nearest")
                update_x -= learning_rate * inverse_consistency_weight * (field_x - mapped_field_x)
                update_y -= learning_rate * inverse_consistency_weight * (field_y - mapped_field_y)
            update_x = gaussian_filter(update_x, sigma=max(update_smoothing_sigma, 0.01), mode="nearest")
            update_y = gaussian_filter(update_y, sigma=max(update_smoothing_sigma, 0.01), mode="nearest")
            magnitude_damping = 1.0 / (1.0 + magnitude_weight * np.hypot(field_x, field_y))
            update_x *= magnitude_damping
            update_y *= magnitude_damping
            update_magnitude = np.hypot(update_x, update_y)
            update_scale = np.minimum(
                1.0,
                (density_pixel_size * 0.25) / np.maximum(update_magnitude, 1e-12),
            )
            update_x *= update_scale
            update_y *= update_scale

            accepted_scale = 0.0
            rejection = "line_search_exhausted"
            previous_total = float(current_state["objective"]["total"])
            trial_scale = 1.0
            accepted_state: dict[str, object] | None = None
            for _ in range(int(max_backtracking_steps)):
                trial_x, trial_y = _compose_fields(
                    field_x,
                    field_y,
                    update_x * trial_scale,
                    update_y * trial_scale,
                    density_pixel_size,
                )
                trial_state = evaluate_state(trial_x, trial_y, sigma_px)
                accepted, rejection = _density_flow_trial_is_acceptable(
                    trial_state,
                    current_objective=previous_total,
                    objective_tolerance=objective_tolerance,
                    jacobian_min_threshold=jacobian_min_threshold,
                    jacobian_max_threshold=jacobian_max_threshold,
                    max_displacement=max_displacement,
                    displacement_p95_limit=displacement_p95_limit,
                )
                if accepted:
                    accepted_scale = trial_scale
                    accepted_state = trial_state
                    field_x, field_y = trial_x, trial_y
                    break
                backtracked_steps += 1
                trial_scale *= 0.5

            if accepted_state is None:
                rejected_steps += 1
                consecutive_failed_updates += 1
                history.append(
                    {
                        "level": int(level),
                        "iteration": int(iteration),
                        "global_iteration": global_iteration,
                        "sigma_px": float(sigma_px),
                        "accepted": False,
                        "accepted_update_scale": 0.0,
                        "backtracking_attempts": int(max_backtracking_steps),
                        "rejection_reason": rejection,
                        **current_state["objective"],
                    }
                )
                if consecutive_failed_updates >= early_stopping_patience:
                    break
                continue

            accepted_steps += 1
            consecutive_failed_updates = 0
            current_state = accepted_state
            objective_improvement = previous_total - float(current_state["objective"]["total"])
            objective_stall_count = (
                objective_stall_count + 1
                if objective_improvement <= max(objective_tolerance * 10.0, abs(previous_total) * 1e-6)
                else 0
            )
            metric_values, _ = point_metrics_for_field(field_x, field_y)
            metric_improved = bool(
                float(metric_values["symmetric_median_distance"]) < best_observed_median - 1e-12
                or float(metric_values["mutual_nearest_fraction"]) > best_observed_mutual + 1e-12
            )
            if metric_improved:
                best_observed_median = min(best_observed_median, float(metric_values["symmetric_median_distance"]))
                best_observed_mutual = max(best_observed_mutual, float(metric_values["mutual_nearest_fraction"]))
                point_stall_count = 0
            else:
                point_stall_count += 1

            checkpoint = {
                "field_x": field_x.copy(),
                "field_y": field_y.copy(),
                "metrics": metric_values,
                "iteration": global_iteration,
                "objective": float(current_state["objective"]["total"]),
            }
            if _better_density_flow_checkpoint(checkpoint, best_attempted):
                best_attempted = checkpoint
            if improves_affine(metric_values) and _better_density_flow_checkpoint(
                checkpoint, best_checkpoint
            ):
                best_checkpoint = checkpoint
            history.append(
                {
                    "level": int(level),
                    "iteration": int(iteration),
                    "global_iteration": global_iteration,
                    "sigma_px": float(sigma_px),
                    "accepted": True,
                    "accepted_update_scale": float(accepted_scale),
                    "backtracking_attempts": int(round(-np.log2(accepted_scale))) if accepted_scale > 0 else 0,
                    "rejection_reason": None,
                    "symmetric_median_distance": float(metric_values["symmetric_median_distance"]),
                    "mutual_nearest_fraction": float(metric_values["mutual_nearest_fraction"]),
                    "jacobian_min": float(current_state["jacobian_min"]),
                    "jacobian_max": float(current_state["jacobian_max"]),
                    "max_displacement": float(current_state["max_displacement"]),
                    "p95_displacement": float(current_state["p95_displacement"]),
                    **current_state["objective"],
                }
            )
            if objective_stall_count >= early_stopping_patience or point_stall_count >= point_metric_patience:
                break

    final_field_x = field_x.copy()
    final_field_y = field_y.copy()
    final_metric_values, _ = point_metrics_for_field(final_field_x, final_field_y)
    selected = best_checkpoint if best_checkpoint is not None else best_attempted
    if selected is None:
        selected = {
            "field_x": zeros.copy(),
            "field_y": zeros.copy(),
            "metrics": {**before_metrics, "mutual_nearest_fraction": before_mutual},
            "iteration": None,
        }
    field_x = np.asarray(selected["field_x"]).copy()
    field_y = np.asarray(selected["field_y"]).copy()
    attempted_points = original_moving + _sample_field(original_moving, field_x, field_y, resolved_bounds, density_pixel_size)
    attempted_metric_points = metric_moving + _sample_field(metric_moving, field_x, field_y, resolved_bounds, density_pixel_size)
    attempted_metrics = point_bidirectional_distance_metrics(metric_fixed, attempted_metric_points)
    attempted_metrics["mutual_nearest_fraction"] = _mutual_nearest_fraction(metric_fixed, attempted_metric_points)
    attempted_metrics["possible_xy_reversal"] = False
    attempted_metrics["xy_reversal_diagnostics"] = reversal

    canonical_sigma = active_scales[-1]
    canonical_initial_state = evaluate_state(initial_field_x, initial_field_y, canonical_sigma)
    canonical_attempted_state = evaluate_state(field_x, field_y, canonical_sigma)
    canonical_final_state = evaluate_state(final_field_x, final_field_y, canonical_sigma)
    initial_objective = float(canonical_initial_state["objective"]["total"])
    best_objective = float(canonical_attempted_state["objective"]["total"])
    final_objective = float(canonical_final_state["objective"]["total"])

    deformation = density_flow_deformation_diagnostics(
        field_x,
        field_y,
        pixel_size=density_pixel_size,
    )
    jacobian = deformation["jacobian"]
    displacement = deformation["raw_magnitude"]
    jacobian_min = float(np.min(jacobian))
    jacobian_max = float(np.max(jacobian))
    jacobian_median = float(np.median(jacobian))
    fold_fraction = float(np.mean(jacobian <= 0.0))
    maximum_displacement = float(np.max(displacement))
    p95_displacement = float(np.percentile(displacement, 95))
    before_support = _sample_field(metric_moving, tissue_mask.astype(float), tissue_mask.astype(float), resolved_bounds, density_pixel_size)[:, 0]
    after_support = _sample_field(attempted_metric_points, tissue_mask.astype(float), tissue_mask.astype(float), resolved_bounds, density_pixel_size)[:, 0]
    outside_before = float(np.mean(before_support < 0.5))
    outside_after = float(np.mean(after_support < 0.5))
    finite_output = bool(
        np.isfinite(attempted_points).all()
        and np.isfinite(field_x).all()
        and np.isfinite(field_y).all()
        and np.isfinite(jacobian).all()
    )

    candidate_safe, candidate_safety_reason = _density_flow_trial_is_safe(
        canonical_attempted_state,
        jacobian_min_threshold=jacobian_min_threshold,
        jacobian_max_threshold=jacobian_max_threshold,
        max_displacement=max_displacement,
        displacement_p95_limit=displacement_p95_limit,
    )
    success = bool(best_checkpoint is not None and candidate_safe)
    rejection_reason: str | None = None
    message = "Tissue-aware density-flow point registration completed."
    if best_checkpoint is None:
        rejection_reason = "no_improving_safe_checkpoint"
        message = "Density-flow was rejected because no safe checkpoint improved the affine point metrics."
    elif not candidate_safe:
        success = False
        rejection_reason = candidate_safety_reason
        message = f"Density-flow best checkpoint failed the strict final safety check: {candidate_safety_reason}."
    elif outside_after > outside_before + 0.05:
        success = False
        rejection_reason = "points_outside_tissue_increased"
        message = "Density-flow candidate was rejected because more HE points moved outside tissue support."

    safety = {
        "finite_output": finite_output,
        "attempted_jacobian_min": jacobian_min,
        "attempted_jacobian_max": jacobian_max,
        "attempted_jacobian_median": jacobian_median,
        "fraction_jacobian_foldover_le_0": fold_fraction,
        "attempted_max_displacement": maximum_displacement,
        "attempted_p95_displacement": p95_displacement,
        **{
            key: value
            for key, value in deformation.items()
            if not isinstance(value, np.ndarray)
        },
        "max_final_displacement_limit": float(max_displacement),
        "displacement_p95_limit": None if displacement_p95_limit is None else float(displacement_p95_limit),
        "points_outside_tissue_before_fraction": outside_before,
        "points_outside_tissue_attempted_fraction": outside_after,
        "mutual_nearest_fraction_before": before_mutual,
        "mutual_nearest_fraction_attempted": attempted_metrics["mutual_nearest_fraction"],
    }
    applied_points = attempted_points if success else moving.copy()
    applied_metrics = attempted_metrics if success else before_metrics
    if not success:
        applied_metrics = {**before_metrics, "mutual_nearest_fraction": before_mutual}

    return FineWarpResult(
        transformed_points=applied_points,
        grid_x=grid_x,
        grid_y=grid_y,
        displacement_x=field_x if success else zeros,
        displacement_y=field_y if success else zeros,
        bounds=resolved_bounds,
        grid_spacing=float(density_pixel_size),
        jacobian_min=jacobian_min,
        jacobian_max=jacobian_max,
        max_displacement=maximum_displacement,
        n_candidate_pairs=int(len(history)),
        n_pairs=int(len(moving)) if success else 0,
        n_filtered_pairs=0 if success else int(len(moving)),
        median_pair_distance_before=before_metrics["median_distance"],
        median_pair_distance_after=applied_metrics["median_distance"],
        success=success,
        message=message,
        attempted_transformed_points=attempted_points,
        attempted_displacement_x=field_x,
        attempted_displacement_y=field_y,
        attempted_metrics=attempted_metrics,
        applied_metrics=applied_metrics,
        rejection_reason=rejection_reason,
        applied=success,
        anchors=None,
        metrics={
            "before": {**before_metrics, "mutual_nearest_fraction": before_mutual},
            "attempted": attempted_metrics,
            "applied": applied_metrics,
            "safety": safety,
            "optimization_history": pd.DataFrame(history).to_dict(orient="records"),
            "density_flow": {
                "density_pixel_size": float(density_pixel_size),
                "density_blur_scales_px": list(active_scales),
                "optimization_levels": len(active_scales),
                "iterations_per_level": int(iterations_per_level),
                "attempted_optimization_steps": attempted_steps,
                "accepted_update_steps": accepted_steps,
                "rejected_update_steps": rejected_steps,
                "rejected_or_backtracked_update_steps": backtracked_steps,
                "best_checkpoint_iteration": (
                    None if best_checkpoint is None else best_checkpoint.get("iteration")
                ),
                "initial_objective": initial_objective,
                "best_objective": best_objective,
                "final_objective": final_objective,
                "affine_symmetric_median": float(before_metrics["symmetric_median_distance"]),
                "best_attempted_symmetric_median": float(attempted_metrics["symmetric_median_distance"]),
                "final_attempted_symmetric_median": float(final_metric_values["symmetric_median_distance"]),
                "mutual_nearest_fraction_before": before_mutual,
                "mutual_nearest_fraction_best": float(attempted_metrics["mutual_nearest_fraction"]),
                "mutual_nearest_fraction_final": float(final_metric_values["mutual_nearest_fraction"]),
                "objective_tolerance": float(objective_tolerance),
                "early_stopping_patience": int(early_stopping_patience),
                "point_metric_patience": int(point_metric_patience),
                "max_backtracking_steps": int(max_backtracking_steps),
                "global_density_shift_x": float(global_shift[0]),
                "global_density_shift_y": float(global_shift[1]),
                "global_translation_initialization": global_initialization,
                "global_translation_description": (
                    "additional post-affine global translation"
                    if global_initialization == "auto"
                    else "disabled; nonlinear field initialized at zero"
                ),
            },
        },
    )
