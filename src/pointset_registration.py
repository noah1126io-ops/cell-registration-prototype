from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SimilarityTransform:
    scale: float
    rotation_rad: float
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        rotation = np.array(
            [
                [np.cos(self.rotation_rad), -np.sin(self.rotation_rad)],
                [np.sin(self.rotation_rad), np.cos(self.rotation_rad)],
            ],
            dtype=float,
        )
        return self.scale * np.asarray(points, dtype=float) @ rotation.T + self.translation

    def to_affine(self) -> tuple[np.ndarray, np.ndarray]:
        rotation = np.array(
            [
                [np.cos(self.rotation_rad), -np.sin(self.rotation_rad)],
                [np.sin(self.rotation_rad), np.cos(self.rotation_rad)],
            ],
            dtype=float,
        )
        return self.scale * rotation, self.translation.copy()


@dataclass(frozen=True)
class AffineICPResult:
    affine_matrix: np.ndarray
    translation: np.ndarray
    transformed_points: np.ndarray
    flip_y: bool
    image_height: float
    mean_residual: float
    median_residual: float
    n_pairs: int
    success: bool
    message: str


@dataclass(frozen=True)
class FineWarpResult:
    transformed_points: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    displacement_x: np.ndarray
    displacement_y: np.ndarray
    bounds: tuple[float, float, float, float]
    grid_spacing: float
    jacobian_min: float
    jacobian_max: float
    max_displacement: float
    n_pairs: int
    median_pair_distance_before: float
    median_pair_distance_after: float
    success: bool
    message: str


def _as_points(points: np.ndarray, *, name: str) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n_points, 2).")
    if points.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one point.")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must not contain NaN or infinite coordinates.")
    return points


def apply_y_flip(points: np.ndarray, image_height: float) -> np.ndarray:
    """Flip image-space y coordinates around the HE image height."""
    if not np.isfinite(image_height) or image_height <= 0:
        raise ValueError("image_height must be a positive finite value.")
    flipped = _as_points(points, name="points").copy()
    flipped[:, 1] = float(image_height) - flipped[:, 1]
    return flipped


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> SimilarityTransform:
    """Estimate a 2D similarity transform from paired points."""
    src = _as_points(src, name="src")
    dst = _as_points(dst, name="dst")
    if len(src) != len(dst):
        raise ValueError("src and dst must contain the same number of paired points.")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = dst_centered.T @ src_centered / len(src)

    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.eye(2)
    if np.linalg.det(u @ vt) < 0:
        sign[-1, -1] = -1
    rotation = u @ sign @ vt

    variance = np.mean(np.sum(src_centered**2, axis=1))
    scale = 1.0 if variance <= 0 else float(np.trace(np.diag(singular_values) @ sign) / variance)
    translation = dst_mean - scale * (rotation @ src_mean)
    angle = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return SimilarityTransform(scale=scale, rotation_rad=angle, translation=translation)


def initial_similarity(src: np.ndarray, dst: np.ndarray, *, rotation_rad: float = 0.0) -> SimilarityTransform:
    """Create a centroid/scale based similarity initialization with a chosen rotation."""
    src = _as_points(src, name="src")
    dst = _as_points(dst, name="dst")
    src_std = float(np.sqrt(np.mean(np.sum((src - src.mean(axis=0)) ** 2, axis=1))))
    dst_std = float(np.sqrt(np.mean(np.sum((dst - dst.mean(axis=0)) ** 2, axis=1))))
    scale = 1.0 if src_std <= 0 else dst_std / src_std
    rotation = np.array(
        [
            [np.cos(rotation_rad), -np.sin(rotation_rad)],
            [np.sin(rotation_rad), np.cos(rotation_rad)],
        ],
        dtype=float,
    )
    translation = dst.mean(axis=0) - scale * (rotation @ src.mean(axis=0))
    return SimilarityTransform(scale=scale, rotation_rad=float(rotation_rad), translation=translation)


def _nearest_pairs(src_transformed: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(dst)
    distances, indices = tree.query(src_transformed, k=1)
    return np.asarray(distances, dtype=float), np.asarray(indices, dtype=int)


def icp_similarity(
    src: np.ndarray,
    dst: np.ndarray,
    initial: SimilarityTransform | None = None,
    *,
    max_iter: int = 100,
    tol: float = 1e-4,
    trim_quantile: float = 0.8,
) -> tuple[SimilarityTransform, float, int]:
    """Run trimmed nearest-neighbor ICP constrained to similarity transforms."""
    src = _as_points(src, name="src")
    dst = _as_points(dst, name="dst")
    transform = initial or initial_similarity(src, dst)
    previous_error = np.inf
    n_used = 0

    for _ in range(max_iter):
        moved = transform.apply(src)
        distances, indices = _nearest_pairs(moved, dst)
        cutoff = np.quantile(distances, trim_quantile)
        keep = distances <= cutoff
        if keep.sum() < 2:
            break

        delta = umeyama_similarity(src[keep], dst[indices[keep]])
        transform = delta
        error = float(np.mean(distances[keep]))
        n_used = int(keep.sum())
        if abs(previous_error - error) < tol:
            previous_error = error
            break
        previous_error = error

    return transform, float(previous_error), n_used


def best_of_initial_rotations(
    src: np.ndarray,
    dst: np.ndarray,
    rotations_deg: Iterable[float] = range(0, 360, 30),
    **icp_kwargs,
) -> tuple[SimilarityTransform, float, int]:
    """Try multiple similarity ICP starts and keep the lowest residual fit."""
    best_transform = None
    best_error = np.inf
    best_n_used = 0

    for rotation_deg in rotations_deg:
        initial = initial_similarity(src, dst, rotation_rad=np.deg2rad(rotation_deg))
        transform, error, n_used = icp_similarity(src, dst, initial=initial, **icp_kwargs)
        if error < best_error:
            best_transform = transform
            best_error = error
            best_n_used = n_used

    if best_transform is None:
        best_transform = initial_similarity(src, dst)
    return best_transform, float(best_error), best_n_used


def affine_lstsq(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a full 2D affine transform from paired points."""
    src = _as_points(src, name="src")
    dst = _as_points(dst, name="dst")
    if len(src) != len(dst):
        raise ValueError("src and dst must contain the same number of paired points.")
    if len(src) < 3:
        raise ValueError("At least three point pairs are required for affine estimation.")

    design = np.column_stack([src[:, 0], src[:, 1], np.ones(len(src))])
    coeff_x, *_ = np.linalg.lstsq(design, dst[:, 0], rcond=None)
    coeff_y, *_ = np.linalg.lstsq(design, dst[:, 1], rcond=None)
    affine_matrix = np.array([[coeff_x[0], coeff_x[1]], [coeff_y[0], coeff_y[1]]], dtype=float)
    translation = np.array([coeff_x[2], coeff_y[2]], dtype=float)
    return affine_matrix, translation


def apply_affine(points: np.ndarray, affine_matrix: np.ndarray, translation: np.ndarray) -> np.ndarray:
    points = _as_points(points, name="points")
    affine_matrix = np.asarray(affine_matrix, dtype=float)
    translation = np.asarray(translation, dtype=float)
    if affine_matrix.shape != (2, 2) or translation.shape != (2,):
        raise ValueError("affine_matrix must be (2, 2) and translation must be (2,).")
    return points @ affine_matrix.T + translation


def affine_icp(
    src: np.ndarray,
    dst: np.ndarray,
    affine_matrix: np.ndarray,
    translation: np.ndarray,
    *,
    max_iter: int = 80,
    trim_quantile: float = 0.7,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Run trimmed nearest-neighbor ICP with a full affine update."""
    src = _as_points(src, name="src")
    dst = _as_points(dst, name="dst")
    affine_matrix = np.asarray(affine_matrix, dtype=float)
    translation = np.asarray(translation, dtype=float)
    previous_error = np.inf
    n_used = 0

    for _ in range(max_iter):
        moved = apply_affine(src, affine_matrix, translation)
        distances, indices = _nearest_pairs(moved, dst)
        cutoff = np.quantile(distances, trim_quantile)
        keep = distances <= cutoff
        if keep.sum() < 3:
            break

        affine_matrix, translation = affine_lstsq(src[keep], dst[indices[keep]])
        error = float(np.mean(distances[keep]))
        n_used = int(keep.sum())
        if abs(previous_error - error) < tol:
            previous_error = error
            break
        previous_error = error

    return affine_matrix, translation, float(previous_error), n_used


def mutual_nearest_pairs(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mutual nearest-neighbor point pairs within max_distance."""
    source_points = _as_points(source_points, name="source_points")
    target_points = _as_points(target_points, name="target_points")

    target_tree = cKDTree(target_points)
    source_to_target_distance, source_to_target_index = target_tree.query(source_points, k=1)

    source_tree = cKDTree(source_points)
    target_to_source_distance, target_to_source_index = source_tree.query(target_points, k=1)

    source_indices = []
    target_indices = []
    distances = []
    for source_index, target_index in enumerate(source_to_target_index):
        if target_to_source_index[target_index] != source_index:
            continue
        distance = float(source_to_target_distance[source_index])
        if distance <= max_distance:
            source_indices.append(source_index)
            target_indices.append(int(target_index))
            distances.append(distance)

    return np.asarray(source_indices, dtype=int), np.asarray(target_indices, dtype=int), np.asarray(distances, dtype=float)


def estimate_affine_with_y_flip(
    he_points_px: np.ndarray,
    geojson_points_um: np.ndarray,
    *,
    image_height_px: float,
    rotations_deg: Iterable[float] = range(0, 360, 30),
    similarity_trim_quantile: float = 0.8,
    affine_trim_quantile: float = 0.7,
    max_iter: int = 80,
) -> AffineICPResult:
    """Estimate HE-pixel to GeoJSON-world affine alignment, trying both y orientations."""
    he_points_px = _as_points(he_points_px, name="he_points_px")
    geojson_points_um = _as_points(geojson_points_um, name="geojson_points_um")

    best_result = None
    best_orientation_error = np.inf
    for flip_y in (False, True):
        source = apply_y_flip(he_points_px, image_height_px) if flip_y else he_points_px
        similarity, orientation_error, _ = best_of_initial_rotations(
            source,
            geojson_points_um,
            rotations_deg=rotations_deg,
            max_iter=max_iter,
            trim_quantile=similarity_trim_quantile,
        )
        affine_matrix, translation = similarity.to_affine()
        affine_matrix, translation, mean_residual, n_pairs = affine_icp(
            source,
            geojson_points_um,
            affine_matrix,
            translation,
            max_iter=max_iter,
            trim_quantile=affine_trim_quantile,
        )
        transformed = apply_affine(source, affine_matrix, translation)
        distances, _ = _nearest_pairs(transformed, geojson_points_um)
        median_residual = float(np.median(distances))
        result = AffineICPResult(
            affine_matrix=affine_matrix,
            translation=translation,
            transformed_points=transformed,
            flip_y=flip_y,
            image_height=float(image_height_px),
            mean_residual=float(mean_residual if np.isfinite(mean_residual) else np.mean(distances)),
            median_residual=median_residual,
            n_pairs=int(n_pairs),
            success=True,
            message="Affine ICP completed.",
        )
        orientation_error = float(orientation_error)
        if not np.isfinite(orientation_error):
            orientation_error = result.median_residual
        if (
            best_result is None
            or orientation_error < best_orientation_error
            or (
                np.isclose(orientation_error, best_orientation_error)
                and result.median_residual < best_result.median_residual
            )
        ):
            best_result = result
            best_orientation_error = orientation_error

    if best_result is None:
        identity = np.eye(2, dtype=float)
        transformed = apply_affine(he_points_px, identity, np.zeros(2, dtype=float))
        return AffineICPResult(
            affine_matrix=identity,
            translation=np.zeros(2, dtype=float),
            transformed_points=transformed,
            flip_y=False,
            image_height=float(image_height_px),
            mean_residual=float("inf"),
            median_residual=float("inf"),
            n_pairs=0,
            success=False,
            message="Affine ICP failed; identity transform was used.",
        )
    return best_result


def _build_grid(points_a: np.ndarray, points_b: np.ndarray, *, padding: float, grid_spacing: float):
    all_points = np.vstack([points_a, points_b])
    min_x, min_y = all_points.min(axis=0) - padding
    max_x, max_y = all_points.max(axis=0) + padding
    xs = np.arange(min_x, max_x + grid_spacing, grid_spacing, dtype=float)
    ys = np.arange(min_y, max_y + grid_spacing, grid_spacing, dtype=float)
    return np.meshgrid(xs, ys), (float(min_x), float(min_y), float(max_x), float(max_y))


def _sample_field(points: np.ndarray, field_x: np.ndarray, field_y: np.ndarray, bounds, grid_spacing: float) -> np.ndarray:
    min_x, min_y, _, _ = bounds
    cols = (points[:, 0] - min_x) / grid_spacing
    rows = (points[:, 1] - min_y) / grid_spacing
    coords = np.vstack([rows, cols])
    dx = map_coordinates(field_x, coords, order=1, mode="nearest")
    dy = map_coordinates(field_y, coords, order=1, mode="nearest")
    return np.column_stack([dx, dy])


def fine_center_snap_warp(
    source_world_points: np.ndarray,
    target_world_points: np.ndarray,
    *,
    match_radius: float = 10.0,
    grid_spacing: float = 6.0,
    bandwidth: float = 12.0,
    ridge: float = 0.3,
    padding: float = 30.0,
    closeness_sigma: float = 8.0,
) -> FineWarpResult:
    """Create a confidence-weighted smooth center-snap displacement field."""
    source_world_points = _as_points(source_world_points, name="source_world_points")
    target_world_points = _as_points(target_world_points, name="target_world_points")

    (grid_x, grid_y), bounds = _build_grid(
        source_world_points,
        target_world_points,
        padding=padding,
        grid_spacing=grid_spacing,
    )
    field_shape = grid_x.shape
    source_indices, target_indices, distances = mutual_nearest_pairs(
        source_world_points,
        target_world_points,
        max_distance=match_radius,
    )

    if len(source_indices) == 0:
        zeros = np.zeros(field_shape, dtype=float)
        return FineWarpResult(
            transformed_points=source_world_points.copy(),
            grid_x=grid_x,
            grid_y=grid_y,
            displacement_x=zeros,
            displacement_y=zeros,
            bounds=bounds,
            grid_spacing=float(grid_spacing),
            jacobian_min=1.0,
            jacobian_max=1.0,
            max_displacement=0.0,
            n_pairs=0,
            median_pair_distance_before=float("inf"),
            median_pair_distance_after=float("inf"),
            success=False,
            message="No mutual nearest-neighbor pairs within match_radius; fine warp is identity.",
        )

    source_pair_points = source_world_points[source_indices]
    target_pair_points = target_world_points[target_indices]
    displacement = target_pair_points - source_pair_points
    confidence = np.exp(-(distances**2) / (2.0 * closeness_sigma**2))

    numerator_x = np.zeros(field_shape, dtype=float)
    numerator_y = np.zeros(field_shape, dtype=float)
    denominator = np.zeros(field_shape, dtype=float)

    min_x, min_y, _, _ = bounds
    cols = np.rint((source_pair_points[:, 0] - min_x) / grid_spacing).astype(int)
    rows = np.rint((source_pair_points[:, 1] - min_y) / grid_spacing).astype(int)
    rows = np.clip(rows, 0, field_shape[0] - 1)
    cols = np.clip(cols, 0, field_shape[1] - 1)

    for row, col, dx, dy, weight in zip(rows, cols, displacement[:, 0], displacement[:, 1], confidence):
        numerator_x[row, col] += weight * dx
        numerator_y[row, col] += weight * dy
        denominator[row, col] += weight

    sigma_pixels = max(float(bandwidth) / float(grid_spacing), 0.01)
    numerator_x = gaussian_filter(numerator_x, sigma=sigma_pixels)
    numerator_y = gaussian_filter(numerator_y, sigma=sigma_pixels)
    denominator = gaussian_filter(denominator, sigma=sigma_pixels)
    support = denominator[denominator > 0]
    ridge_value = float(ridge) * (float(np.median(support)) if support.size else 1.0)

    displacement_x = numerator_x / (denominator + ridge_value)
    displacement_y = numerator_y / (denominator + ridge_value)
    sampled_displacement = _sample_field(source_world_points, displacement_x, displacement_y, bounds, grid_spacing)
    transformed_points = source_world_points + sampled_displacement

    dfx_dy, dfx_dx = np.gradient(displacement_x, grid_spacing, grid_spacing)
    dfy_dy, dfy_dx = np.gradient(displacement_y, grid_spacing, grid_spacing)
    jacobian = (1.0 + dfx_dx) * (1.0 + dfy_dy) - dfx_dy * dfy_dx

    _, _, after_distances = mutual_nearest_pairs(transformed_points, target_world_points, max_distance=match_radius)
    median_after = float(np.median(after_distances)) if len(after_distances) else float("inf")
    max_displacement = float(np.max(np.sqrt(displacement_x**2 + displacement_y**2)))

    return FineWarpResult(
        transformed_points=transformed_points,
        grid_x=grid_x,
        grid_y=grid_y,
        displacement_x=displacement_x,
        displacement_y=displacement_y,
        bounds=bounds,
        grid_spacing=float(grid_spacing),
        jacobian_min=float(np.min(jacobian)),
        jacobian_max=float(np.max(jacobian)),
        max_displacement=max_displacement,
        n_pairs=int(len(source_indices)),
        median_pair_distance_before=float(np.median(distances)),
        median_pair_distance_after=median_after,
        success=True,
        message="Fine center-snap warp completed.",
    )
