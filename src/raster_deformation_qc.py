from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates, sobel
from scipy.spatial import cKDTree


def grayscale_float(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=float)
    if values.ndim == 3:
        values = np.mean(values[..., :3], axis=2)
    if values.ndim != 2:
        raise ValueError("image must be grayscale or RGB/RGBA.")
    return values


def raster_difference_metrics(
    affine_image: np.ndarray,
    warped_image: np.ndarray,
    *,
    tissue_mask: np.ndarray | None = None,
) -> dict:
    affine = grayscale_float(affine_image)
    warped = grayscale_float(warped_image)
    if affine.shape != warped.shape:
        raise ValueError("affine_image and warped_image must have matching shapes.")
    mask = np.ones(affine.shape, dtype=bool) if tissue_mask is None else np.asarray(tissue_mask, dtype=bool)
    if mask.shape != affine.shape:
        raise ValueError("tissue_mask must match the image shape.")

    def summarize(region: np.ndarray) -> dict:
        if not np.any(region):
            return {key: None for key in (
                "mean_absolute_difference", "p95_absolute_difference", "max_absolute_difference",
                "fraction_changed_above_1", "fraction_changed_above_5", "structural_similarity",
                "gradient_image_difference", "edge_symmetric_median_displacement_pixels",
                "edge_p95_displacement_pixels",
            )}
        difference = np.abs(warped - affine)
        selected = difference[region]
        gradient_affine = np.hypot(sobel(affine, axis=0), sobel(affine, axis=1))
        gradient_warped = np.hypot(sobel(warped, axis=0), sobel(warped, axis=1))
        gradient_difference = np.abs(gradient_warped - gradient_affine)[region]
        threshold_affine = np.percentile(gradient_affine[region], 85)
        threshold_warped = np.percentile(gradient_warped[region], 85)
        edge_affine = (gradient_affine >= threshold_affine) & region
        edge_warped = (gradient_warped >= threshold_warped) & region
        edge_distances = np.array([], dtype=float)
        if np.any(edge_affine) and np.any(edge_warped):
            to_warped = distance_transform_edt(~edge_warped)[edge_affine]
            to_affine = distance_transform_edt(~edge_affine)[edge_warped]
            edge_distances = np.concatenate([to_warped, to_affine])
        ssim_value = None
        try:
            from skimage.metrics import structural_similarity

            data_range = float(max(affine[region].max(), warped[region].max()) - min(affine[region].min(), warped[region].min()))
            if data_range > 0:
                # SSIM is calculated on the full common canvas; the mask-specific pixel differences remain separate.
                ssim_value = float(structural_similarity(affine, warped, data_range=data_range))
        except (ImportError, ValueError):
            pass
        return {
            "mean_absolute_difference": float(np.mean(selected)),
            "p95_absolute_difference": float(np.percentile(selected, 95)),
            "max_absolute_difference": float(np.max(selected)),
            "fraction_changed_above_1": float(np.mean(selected > 1.0)),
            "fraction_changed_above_5": float(np.mean(selected > 5.0)),
            "structural_similarity": ssim_value,
            "gradient_image_difference": float(np.mean(gradient_difference)),
            "edge_symmetric_median_displacement_pixels": (
                float(np.median(edge_distances)) if edge_distances.size else None
            ),
            "edge_p95_displacement_pixels": (
                float(np.percentile(edge_distances, 95)) if edge_distances.size else None
            ),
        }

    return {"full_image": summarize(np.ones_like(mask)), "inside_tissue": summarize(mask)}


def point_displacement_pixel_summary(
    displacement_x: np.ndarray,
    displacement_y: np.ndarray,
    *,
    output_pixel_size_um: float,
) -> dict:
    if output_pixel_size_um <= 0 or not np.isfinite(output_pixel_size_um):
        raise ValueError("output_pixel_size_um must be positive and finite.")
    magnitude_um = np.hypot(np.asarray(displacement_x, dtype=float), np.asarray(displacement_y, dtype=float))
    magnitude_pixels = magnitude_um / float(output_pixel_size_um)
    result = {
        "median_um": float(np.median(magnitude_um)),
        "p95_um": float(np.percentile(magnitude_um, 95)),
        "max_um": float(np.max(magnitude_um)),
        "median_output_pixels": float(np.median(magnitude_pixels)),
        "p95_output_pixels": float(np.percentile(magnitude_pixels, 95)),
        "max_output_pixels": float(np.max(magnitude_pixels)),
    }
    for threshold in (0.25, 0.5, 1.0, 2.0, 5.0):
        result[f"fraction_above_{threshold:g}_output_pixels"] = float(np.mean(magnitude_pixels > threshold))
    result["visually_subpixel"] = bool(result["p95_output_pixels"] < 1.0)
    return result


def checkerboard_comparison(first: np.ndarray, second: np.ndarray, *, tile_size: int = 32) -> np.ndarray:
    a = np.asarray(first)
    b = np.asarray(second)
    if a.shape != b.shape:
        raise ValueError("Images must have matching shapes.")
    rows, cols = np.indices(a.shape[:2])
    choose_second = ((rows // max(1, tile_size)) + (cols // max(1, tile_size))) % 2 == 1
    output = a.copy()
    output[choose_second] = b[choose_second]
    return output


def edge_overlay(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    a = grayscale_float(first)
    b = grayscale_float(second)
    grad_a = np.hypot(sobel(a, axis=0), sobel(a, axis=1))
    grad_b = np.hypot(sobel(b, axis=0), sobel(b, axis=1))
    edge_a = grad_a >= np.percentile(grad_a, 88)
    edge_b = grad_b >= np.percentile(grad_b, 88)
    overlay = np.zeros((*a.shape, 3), dtype=np.uint8)
    overlay[..., 0] = edge_a.astype(np.uint8) * 255
    overlay[..., 1] = edge_b.astype(np.uint8) * 255
    overlay[..., 2] = (edge_a & edge_b).astype(np.uint8) * 255
    return overlay


def soft_jacobian_log_penalty(jacobian: np.ndarray, *, epsilon: float = 1e-6) -> float:
    values = np.asarray(jacobian, dtype=float)
    protected = np.maximum(values, epsilon)
    return float(np.mean(np.log(protected) ** 2))


def tissue_support_mismatch(fixed_support: np.ndarray, moving_support: np.ndarray) -> float:
    fixed = np.asarray(fixed_support, dtype=float)
    moving = np.asarray(moving_support, dtype=float)
    if fixed.shape != moving.shape:
        raise ValueError("Support maps must have matching shapes.")
    return float(np.mean((fixed - moving) ** 2))


def _sample_grid_field(
    points: np.ndarray,
    values: np.ndarray,
    bounds: tuple[float, float, float, float],
    spacing: float,
) -> np.ndarray:
    min_x, min_y, _, _ = bounds
    cols = (points[:, 0] - min_x) / spacing
    rows = (points[:, 1] - min_y) / spacing
    return map_coordinates(values, [rows, cols], order=1, mode="nearest")


def local_region_metrics(
    fixed_points: np.ndarray,
    affine_moving_points: np.ndarray,
    warped_moving_points: np.ndarray,
    displacement_x: np.ndarray,
    displacement_y: np.ndarray,
    jacobian: np.ndarray,
    *,
    bounds: tuple[float, float, float, float],
    field_spacing: float,
    block_size_um: float = 100.0,
    min_points: int = 3,
) -> tuple[pd.DataFrame, dict]:
    fixed = np.asarray(fixed_points, dtype=float)
    before = np.asarray(affine_moving_points, dtype=float)
    after = np.asarray(warped_moving_points, dtype=float)
    min_x, min_y, max_x, max_y = bounds
    rows = []
    region_id = 0
    for y0 in np.arange(min_y, max_y, block_size_um):
        for x0 in np.arange(min_x, max_x, block_size_um):
            x1, y1 = min(x0 + block_size_um, max_x), min(y0 + block_size_um, max_y)
            fixed_mask = (fixed[:, 0] >= x0) & (fixed[:, 0] < x1) & (fixed[:, 1] >= y0) & (fixed[:, 1] < y1)
            before_mask = (before[:, 0] >= x0) & (before[:, 0] < x1) & (before[:, 1] >= y0) & (before[:, 1] < y1)
            after_mask = (after[:, 0] >= x0) & (after[:, 0] < x1) & (after[:, 1] >= y0) & (after[:, 1] < y1)
            if np.count_nonzero(fixed_mask) < min_points or np.count_nonzero(before_mask) < min_points or np.count_nonzero(after_mask) < min_points:
                continue
            fixed_local = fixed[fixed_mask]
            before_local = before[before_mask]
            after_local = after[after_mask]
            before_dist = cKDTree(fixed_local).query(before_local, k=1)[0]
            after_dist = cKDTree(fixed_local).query(after_local, k=1)[0]
            center = np.array([[(x0 + x1) / 2.0, (y0 + y1) / 2.0]])
            grid_rows, grid_cols = np.indices(np.asarray(displacement_x).shape)
            grid_world_x = min_x + grid_cols * field_spacing
            grid_world_y = min_y + grid_rows * field_spacing
            grid_mask = (
                (grid_world_x >= x0) & (grid_world_x <= x1)
                & (grid_world_y >= y0) & (grid_world_y <= y1)
            )
            if np.any(grid_mask):
                local_dx = np.asarray(displacement_x, dtype=float)[grid_mask]
                local_dy = np.asarray(displacement_y, dtype=float)[grid_mask]
                local_jac = np.asarray(jacobian, dtype=float)[grid_mask]
            else:
                local_dx = _sample_grid_field(center, displacement_x, bounds, field_spacing)
                local_dy = _sample_grid_field(center, displacement_y, bounds, field_spacing)
                local_jac = _sample_grid_field(center, jacobian, bounds, field_spacing)
            local_displacement = np.hypot(local_dx, local_dy)
            before_median = float(np.median(before_dist))
            after_median = float(np.median(after_dist))
            row = {
                "region_id": region_id,
                "center_x": center[0, 0],
                "center_y": center[0, 1],
                "n_fixed": int(np.count_nonzero(fixed_mask)),
                "n_moving_before": int(np.count_nonzero(before_mask)),
                "n_moving_after": int(np.count_nonzero(after_mask)),
                "affine_median_distance": before_median,
                "warped_median_distance": after_median,
                "delta_median": after_median - before_median,
                "displacement_median": float(np.median(local_displacement)),
                "displacement_p95": float(np.percentile(local_displacement, 95)),
                "jacobian_median": float(np.median(local_jac)),
                "jacobian_p05": float(np.percentile(local_jac, 5)),
                "jacobian_p95": float(np.percentile(local_jac, 95)),
                "density_mismatch_before": float(np.mean(before_dist)),
                "density_mismatch_after": float(np.mean(after_dist)),
            }
            for threshold in (3.0, 5.0, 10.0):
                row[f"within_{threshold:g}_before"] = float(np.mean(before_dist <= threshold))
                row[f"within_{threshold:g}_after"] = float(np.mean(after_dist <= threshold))
            rows.append(row)
            region_id += 1
    table = pd.DataFrame(rows)
    if table.empty:
        return table, {
            "fraction_regions_improved": None,
            "fraction_regions_worsened": None,
            "worst_degraded_region": None,
            "best_improved_region": None,
        }
    return table, {
        "fraction_regions_improved": float(np.mean(table["delta_median"] < 0)),
        "fraction_regions_worsened": float(np.mean(table["delta_median"] > 0)),
        "worst_degraded_region": int(table.loc[table["delta_median"].idxmax(), "region_id"]),
        "best_improved_region": int(table.loc[table["delta_median"].idxmin(), "region_id"]),
    }
