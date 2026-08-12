from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, sobel

from src.registration_evaluation import displacement_endpoint_error, synthetic_raster_metrics


@dataclass(frozen=True)
class SyntheticRegistrationSample:
    fixed_points: np.ndarray
    moving_points: np.ndarray
    fixed_landmarks: np.ndarray
    moving_landmarks: np.ndarray
    fixed_image: np.ndarray
    moving_image: np.ndarray
    moving_tissue_mask: np.ndarray
    ground_truth_warped_image: np.ndarray
    ground_truth_warped_mask: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    ground_truth_displacement_x: np.ndarray
    ground_truth_displacement_y: np.ndarray
    bounds: tuple[float, float, float, float]
    spacing: float
    metadata: dict
    deformation_type: str
    amplitude_um: float
    dropout_fraction: float
    seed: int


def ground_truth_displacement(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    deformation_type: str,
    amplitude_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(grid_x, dtype=float)
    y = np.asarray(grid_y, dtype=float)
    center_x = float((x.min() + x.max()) / 2)
    center_y = float((y.min() + y.max()) / 2)
    width = max(float(x.max() - x.min()), 1.0)
    height = max(float(y.max() - y.min()), 1.0)
    radius2 = ((x - center_x) / (0.23 * width)) ** 2 + ((y - center_y) / (0.23 * height)) ** 2
    gaussian = np.exp(-0.5 * radius2)
    amplitude = float(amplitude_um)
    if deformation_type == "identity":
        return np.zeros_like(x), np.zeros_like(y)
    if deformation_type == "smooth local translation":
        return amplitude * gaussian, -0.35 * amplitude * gaussian
    if deformation_type == "Gaussian bulge":
        radial_x = (x - center_x) / max(0.23 * width, 1e-6)
        radial_y = (y - center_y) / max(0.23 * height, 1e-6)
        norm = np.maximum(np.hypot(radial_x, radial_y), 1e-6)
        return amplitude * gaussian * radial_x / norm, amplitude * gaussian * radial_y / norm
    if deformation_type == "Gaussian compression":
        bulge_x, bulge_y = ground_truth_displacement(x, y, "Gaussian bulge", amplitude)
        return -bulge_x, -bulge_y
    if deformation_type == "shear-like smooth deformation":
        envelope = np.exp(-0.5 * ((y - center_y) / (0.32 * height)) ** 2)
        return amplitude * ((y - center_y) / max(height / 2, 1e-6)) * envelope, np.zeros_like(y)
    if deformation_type == "sinusoidal deformation":
        return (
            amplitude * np.sin(2 * np.pi * (y - y.min()) / height),
            0.6 * amplitude * np.sin(2 * np.pi * (x - x.min()) / width + np.pi / 4),
        )
    raise ValueError(f"Unsupported deformation_type: {deformation_type}")


def _sample_field(
    points: np.ndarray,
    field_x: np.ndarray,
    field_y: np.ndarray,
    spacing: float,
) -> np.ndarray:
    cols = points[:, 0] / spacing
    rows = points[:, 1] / spacing
    dx = map_coordinates(field_x, [rows, cols], order=1, mode="nearest")
    dy = map_coordinates(field_y, [rows, cols], order=1, mode="nearest")
    return np.column_stack([dx, dy])


def inverse_transform_points(
    fixed_points: np.ndarray,
    field_x: np.ndarray,
    field_y: np.ndarray,
    spacing: float,
    *,
    iterations: int = 20,
) -> np.ndarray:
    fixed = np.asarray(fixed_points, dtype=float)
    moving = fixed.copy()
    for _ in range(iterations):
        moving = fixed - _sample_field(moving, field_x, field_y, spacing)
    return moving


def _render_markers(points: np.ndarray, shape: tuple[int, int], *, seed: int) -> np.ndarray:
    image = np.zeros(shape, dtype=float)
    for index, (x, y) in enumerate(points):
        row, col = int(round(y)), int(round(x))
        if 1 <= row < shape[0] - 1 and 1 <= col < shape[1] - 1:
            image[row, col] = 120.0 + ((index * 37 + seed) % 120)
    image = gaussian_filter(image, sigma=1.0)
    texture = 25.0 + 15.0 * np.sin(np.indices(shape)[1] / 11.0) + 8.0 * np.cos(np.indices(shape)[0] / 17.0)
    return np.clip(texture + image, 0, 255).astype(np.uint8)


def warp_raster_with_forward_field(
    moving_image: np.ndarray,
    field_x: np.ndarray,
    field_y: np.ndarray,
    *,
    spacing: float,
    order: int = 1,
    inverse_iterations: int = 20,
) -> np.ndarray:
    rows, cols = np.indices(moving_image.shape[:2], dtype=float)
    source_rows = rows.copy()
    source_cols = cols.copy()
    for _ in range(inverse_iterations):
        sampled_x = map_coordinates(field_x, [source_rows, source_cols], order=1, mode="nearest")
        sampled_y = map_coordinates(field_y, [source_rows, source_cols], order=1, mode="nearest")
        source_cols = cols - sampled_x / spacing
        source_rows = rows - sampled_y / spacing
    if moving_image.ndim == 2:
        return map_coordinates(moving_image, [source_rows, source_cols], order=order, mode="constant", cval=0)
    channels = [
        map_coordinates(moving_image[..., channel], [source_rows, source_cols], order=order, mode="constant", cval=0)
        for channel in range(moving_image.shape[2])
    ]
    return np.stack(channels, axis=2)


def generate_synthetic_registration_sample(
    *,
    deformation_type: str = "Gaussian bulge",
    amplitude_um: float = 5.0,
    dropout_fraction: float = 0.0,
    seed: int = 42,
    size: int = 128,
    spacing: float = 1.0,
    n_points: int = 240,
    added_unmatched_fraction: float = 0.0,
    coordinate_noise_um: float = 0.0,
) -> SyntheticRegistrationSample:
    if dropout_fraction < 0 or dropout_fraction >= 1:
        raise ValueError("dropout_fraction must be in [0, 1).")
    rng = np.random.default_rng(seed)
    axes = np.arange(size, dtype=float) * spacing
    grid_x, grid_y = np.meshgrid(axes, axes)
    field_x, field_y = ground_truth_displacement(grid_x, grid_y, deformation_type, amplitude_um)
    center = (size - 1) * spacing / 2
    radii = np.array([0.39 * size * spacing, 0.34 * size * spacing])
    fixed_candidates = []
    while len(fixed_candidates) < n_points:
        candidate = rng.uniform(8 * spacing, (size - 8) * spacing, size=2)
        if np.sum(((candidate - center) / radii) ** 2) <= 1:
            fixed_candidates.append(candidate)
    fixed_points = np.asarray(fixed_candidates)
    moving_all = inverse_transform_points(fixed_points, field_x, field_y, spacing)
    landmark_indices = np.linspace(0, len(fixed_points) - 1, 16, dtype=int)
    fixed_landmarks = fixed_points[landmark_indices].copy()
    moving_landmarks = moving_all[landmark_indices].copy()
    keep = rng.random(len(fixed_points)) >= dropout_fraction
    fixed_kept = fixed_points[keep]
    moving_points = moving_all[keep]
    if coordinate_noise_um > 0:
        moving_points = moving_points + rng.normal(0.0, coordinate_noise_um, moving_points.shape)
    n_added = int(round(len(moving_points) * added_unmatched_fraction))
    if n_added:
        moving_points = np.vstack([
            moving_points,
            rng.uniform(8 * spacing, (size - 8) * spacing, size=(n_added, 2)),
        ])
    fixed_image = _render_markers(fixed_kept, (size, size), seed=seed)
    moving_image = _render_markers(moving_points, (size, size), seed=seed)
    yy, xx = np.indices((size, size), dtype=float)
    fixed_mask = ((xx - (size - 1) / 2) / (0.39 * size)) ** 2 + ((yy - (size - 1) / 2) / (0.34 * size)) ** 2 <= 1
    # Generate moving support by inverse-transforming fixed-mask coordinates.
    moving_mask = warp_raster_with_forward_field(
        fixed_mask.astype(float), -field_x, -field_y, spacing=spacing, order=1
    ) >= 0.5
    ground_truth_image = warp_raster_with_forward_field(
        moving_image, field_x, field_y, spacing=spacing, order=1
    )
    ground_truth_mask = warp_raster_with_forward_field(
        moving_mask.astype(float), field_x, field_y, spacing=spacing, order=0
    ) >= 0.5
    metadata = {
        "bounds_um": [0.0, 0.0, float((size - 1) * spacing), float((size - 1) * spacing)],
        "output_pixel_size_um": float(spacing),
        "output_origin": "upper-left",
        "width": size, "height": size,
        "row0_world_y": 0.0, "col0_world_x": 0.0,
    }
    return SyntheticRegistrationSample(
        fixed_points=fixed_kept,
        moving_points=moving_points,
        fixed_landmarks=fixed_landmarks,
        moving_landmarks=moving_landmarks,
        fixed_image=fixed_image,
        moving_image=moving_image,
        moving_tissue_mask=moving_mask,
        ground_truth_warped_image=ground_truth_image,
        ground_truth_warped_mask=ground_truth_mask,
        grid_x=grid_x, grid_y=grid_y,
        ground_truth_displacement_x=field_x,
        ground_truth_displacement_y=field_y,
        bounds=(0.0, 0.0, float((size - 1) * spacing), float((size - 1) * spacing)),
        spacing=float(spacing), metadata=metadata,
        deformation_type=deformation_type, amplitude_um=float(amplitude_um),
        dropout_fraction=float(dropout_fraction), seed=int(seed),
    )


def evaluate_synthetic_displacement(
    sample: SyntheticRegistrationSample,
    estimated_x: np.ndarray,
    estimated_y: np.ndarray,
) -> tuple[np.ndarray, dict, np.ndarray]:
    epe_table, summary = displacement_endpoint_error(
        estimated_x, estimated_y,
        sample.ground_truth_displacement_x, sample.ground_truth_displacement_y,
    )
    epe_map = epe_table["epe_um"].to_numpy().reshape(sample.grid_x.shape)
    return epe_table, summary, epe_map


def marker_point_raster_consistency(
    warped_image: np.ndarray,
    transformed_points: np.ndarray,
    *,
    radius: int = 3,
) -> tuple[np.ndarray, dict]:
    image = np.asarray(warped_image, dtype=float)
    if image.ndim == 3:
        image = np.mean(image, axis=2)
    errors = []
    for x, y in np.asarray(transformed_points, dtype=float):
        row, col = int(round(y)), int(round(x))
        r0, r1 = max(0, row - radius), min(image.shape[0], row + radius + 1)
        c0, c1 = max(0, col - radius), min(image.shape[1], col + radius + 1)
        patch = image[r0:r1, c0:c1] - np.min(image[r0:r1, c0:c1])
        if patch.size == 0 or float(np.sum(patch)) <= 0:
            continue
        rr, cc = np.indices(patch.shape, dtype=float)
        detected_row = r0 + float(np.sum(rr * patch) / np.sum(patch))
        detected_col = c0 + float(np.sum(cc * patch) / np.sum(patch))
        errors.append(np.hypot(detected_col - x, detected_row - y))
    values = np.asarray(errors, dtype=float)
    return values, {
        "point_raster_consistency_median_pixels": float(np.median(values)) if values.size else None,
        "point_raster_consistency_p95_pixels": float(np.percentile(values, 95)) if values.size else None,
    }


def synthetic_same_modality_raster_metrics(reference: np.ndarray, recovered: np.ndarray) -> dict:
    return synthetic_raster_metrics(reference, recovered)
