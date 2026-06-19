from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegistrationResult:
    affine_matrix: np.ndarray
    success: bool
    message: str
    ecc_score: float | None = None


def identity_transform() -> np.ndarray:
    """Return a 2D identity affine transform placeholder."""
    return np.eye(3, dtype=np.float32)


def _normalize_density_map(density_map: np.ndarray) -> np.ndarray:
    image = np.asarray(density_map, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("Density maps must be 2D arrays.")

    min_value = float(np.min(image))
    max_value = float(np.max(image))
    if max_value == min_value:
        return np.zeros_like(image, dtype=np.float32)

    return (image - min_value) / (max_value - min_value)


def estimate_affine_transform(
    fixed_density_map: np.ndarray,
    moving_density_map: np.ndarray,
    *,
    max_iterations: int = 500,
    epsilon: float = 1.0e-6,
) -> RegistrationResult:
    """Estimate an affine transform that warps moving density onto fixed density."""
    try:
        import cv2

        fixed = _normalize_density_map(fixed_density_map)
        moving = _normalize_density_map(moving_density_map)

        if fixed.shape != moving.shape:
            raise ValueError("Fixed and moving density maps must have the same shape.")
        if not np.any(fixed) or not np.any(moving):
            raise ValueError("Density maps must contain non-zero signal.")

        initial_warp = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            int(max_iterations),
            float(epsilon),
        )
        ecc_score, warp_matrix = cv2.findTransformECC(
            fixed,
            moving,
            initial_warp,
            cv2.MOTION_AFFINE,
            criteria,
        )

        fixed_to_moving = identity_transform()
        fixed_to_moving[:2, :] = warp_matrix
        affine = np.linalg.inv(fixed_to_moving).astype(np.float32)
        return RegistrationResult(
            affine_matrix=affine,
            success=True,
            message="Affine registration succeeded.",
            ecc_score=float(ecc_score),
        )
    except Exception as exc:
        return RegistrationResult(
            affine_matrix=identity_transform(),
            success=False,
            message=f"Affine registration failed; using identity transform. Reason: {exc}",
            ecc_score=None,
        )


def _copy_to_output_shape(image: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    height, width = output_shape[:2]
    image = np.asarray(image)
    output = np.zeros((height, width, *image.shape[2:]), dtype=image.dtype)
    copy_height = min(height, image.shape[0])
    copy_width = min(width, image.shape[1])
    output[:copy_height, :copy_width] = image[:copy_height, :copy_width]
    return output


def warp_image(image: np.ndarray, affine_matrix: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Warp an image with linear interpolation."""
    try:
        import cv2
    except ImportError:
        return _copy_to_output_shape(image, output_shape)

    height, width = output_shape[:2]
    return cv2.warpAffine(
        np.asarray(image),
        affine_matrix[:2, :].astype(np.float32),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def warp_mask(mask: np.ndarray, affine_matrix: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Warp an integer label mask with nearest-neighbor interpolation."""
    try:
        import cv2
    except ImportError:
        return _copy_to_output_shape(mask, output_shape)

    height, width = output_shape[:2]
    warped = cv2.warpAffine(
        np.asarray(mask),
        affine_matrix[:2, :].astype(np.float32),
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped.astype(np.asarray(mask).dtype, copy=False)


def transform_cell_centroids(cell_features: pd.DataFrame, affine_matrix: np.ndarray) -> pd.DataFrame:
    """Apply an affine transform to cell centroid columns."""
    transformed = cell_features.copy()
    if transformed.empty:
        return transformed

    points = transformed[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float32)
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float32)])
    transformed_points = homogeneous @ affine_matrix[:2, :].T
    transformed["centroid_x"] = transformed_points[:, 0]
    transformed["centroid_y"] = transformed_points[:, 1]
    return transformed


# TODO: Add non-rigid registration after affine alignment is validated.
