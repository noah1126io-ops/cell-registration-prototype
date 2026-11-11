"""Interpretable, rotation-invariant multi-scale local geometric fingerprints."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

from config import FINGERPRINT


@dataclass(frozen=True)
class FingerprintPair:
    fixed: np.ndarray
    moving: np.ndarray
    feature_names: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray
    global_neighbor_scale_um: float


def _global_neighbor_scale(*point_sets: np.ndarray) -> float:
    values = []
    for points in point_sets:
        distances = cKDTree(points).query(points, k=2)[0][:, 1]
        values.append(distances[np.isfinite(distances)])
    pooled = np.concatenate(values)
    return max(float(np.median(pooled)), 1e-6)


def _raw_fingerprints(points: np.ndarray, global_scale: float) -> tuple[np.ndarray, tuple[str, ...]]:
    points = np.asarray(points, dtype=float)
    k = min(int(FINGERPRINT["knn_k"]), len(points) - 1)
    if k < 3:
        raise ValueError("At least four points are required for fingerprints.")
    distances, indices = cKDTree(points).query(points, k=k + 1)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    local_scale = np.maximum(np.median(distances[:, : min(10, k)], axis=1), 1e-6)

    columns: list[np.ndarray] = []
    names: list[str] = []
    for rank in FINGERPRINT["distance_ranks"]:
        effective = min(int(rank), k) - 1
        columns.append(np.log1p(distances[:, effective] / global_scale))
        names.append(f"log_knn_abs_r{rank}")
        columns.append(distances[:, effective] / local_scale)
        names.append(f"knn_local_norm_r{rank}")

    for radius in FINGERPRINT["radial_radii_um"]:
        density = np.sum(np.exp(-0.5 * (distances / float(radius)) ** 2), axis=1)
        count = np.sum(distances <= float(radius), axis=1)
        columns.extend([np.log1p(density), np.log1p(count)])
        names.extend([f"gaussian_density_{radius:g}", f"radial_count_{radius:g}"])

    angular_k = min(16, k)
    vectors = points[indices[:, :angular_k]] - points[:, None, :]
    angles = np.arctan2(vectors[..., 1], vectors[..., 0])
    for harmonic in FINGERPRINT["angular_harmonics"]:
        # Fourier magnitudes are invariant to a shared local rotation.
        moment = np.mean(np.exp(1j * int(harmonic) * angles), axis=1)
        columns.append(np.abs(moment))
        names.append(f"angular_fourier_abs_{harmonic}")

    covariance_k = min(int(FINGERPRINT["covariance_neighbors"]), k)
    covariance_vectors = vectors[:, :covariance_k]
    covariance = np.einsum("nki,nkj->nij", covariance_vectors, covariance_vectors) / covariance_k
    eigenvalues = np.linalg.eigvalsh(covariance)
    columns.append(eigenvalues[:, 1] / np.maximum(eigenvalues[:, 0], 1e-6))
    columns.append(np.sqrt(np.sum(eigenvalues, axis=1)) / global_scale)
    names.extend(["covariance_eigen_ratio", "covariance_trace_scale"])

    pair_k = min(int(FINGERPRINT["pairwise_neighbors"]), k)
    pairwise_quantiles = np.empty((len(points), 3), dtype=float)
    angular_gap_quantiles = np.empty((len(points), 3), dtype=float)
    for row in range(len(points)):
        local = points[indices[row, :pair_k]]
        pair_values = pdist(local) / local_scale[row]
        pairwise_quantiles[row] = np.quantile(pair_values, (0.25, 0.50, 0.75))
        local_angles = np.sort(angles[row, :pair_k])
        gaps = np.diff(np.r_[local_angles, local_angles[0] + 2 * np.pi]) / (2 * np.pi)
        angular_gap_quantiles[row] = np.quantile(gaps, (0.25, 0.50, 0.75))
    for index, quantile in enumerate((25, 50, 75)):
        columns.append(pairwise_quantiles[:, index])
        names.append(f"neighbor_pair_distance_q{quantile}")
        columns.append(angular_gap_quantiles[:, index])
        names.append(f"angular_gap_q{quantile}")

    raw = np.column_stack(columns)
    return raw, tuple(names)


def compute_fingerprint_pair(fixed: np.ndarray, moving: np.ndarray) -> FingerprintPair:
    """Fit one robust normalization jointly, without using correspondence IDs."""
    global_scale = _global_neighbor_scale(fixed, moving)
    fixed_raw, names = _raw_fingerprints(fixed, global_scale)
    moving_raw, moving_names = _raw_fingerprints(moving, global_scale)
    if names != moving_names:
        raise AssertionError("Fingerprint schemas differ.")
    pooled = np.vstack([fixed_raw, moving_raw])
    center = np.median(pooled, axis=0)
    q25, q75 = np.percentile(pooled, (25, 75), axis=0)
    scale = np.maximum(q75 - q25, 0.25)
    fixed_scaled = np.clip((fixed_raw - center) / scale, -6.0, 6.0)
    moving_scaled = np.clip((moving_raw - center) / scale, -6.0, 6.0)
    if not np.isfinite(fixed_scaled).all() or not np.isfinite(moving_scaled).all():
        raise RuntimeError("Fingerprint contains NaN or infinite values.")
    return FingerprintPair(
        fixed=fixed_scaled,
        moving=moving_scaled,
        feature_names=names,
        center=center,
        scale=scale,
        global_neighbor_scale_um=global_scale,
    )


def pairwise_fingerprint_cost(pair: FingerprintPair) -> np.ndarray:
    dimensions = float(pair.fixed.shape[1])
    moving_norm = np.sum(pair.moving * pair.moving, axis=1)[:, None]
    fixed_norm = np.sum(pair.fixed * pair.fixed, axis=1)[None, :]
    squared = moving_norm + fixed_norm - 2.0 * pair.moving @ pair.fixed.T
    return np.maximum(squared, 0.0) / dimensions
