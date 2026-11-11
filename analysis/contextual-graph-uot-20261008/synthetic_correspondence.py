"""Serial-section-like synthetic point clouds with persistent cell identities."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import PRIOR_VALIDATION

# Reuse the validated spatial sampler and smooth-field implementation rather
# than duplicating the previous benchmark's geometry.
sys.path.insert(0, str(PRIOR_VALIDATION))
from validation_lib import _inverse_points, _sample_ellipse_points, _true_field  # noqa: E402


@dataclass(frozen=True)
class CorrespondenceSample:
    fixed_points: np.ndarray
    moving_points: np.ndarray
    fixed_ids: np.ndarray
    moving_ids: np.ndarray
    true_fixed_index: np.ndarray
    common_moving_mask: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    true_field_x: np.ndarray
    true_field_y: np.ndarray
    spacing: float
    bounds: tuple[float, float, float, float]
    seed: int
    difficulty: str
    amplitude_um: float
    missing_fraction: float
    unmatched_fraction: float
    overlap_fraction: float
    noise_um: float
    local_missing_strength: float


def _spatial_missing_probability(
    points: np.ndarray,
    rng: np.random.Generator,
    target_fraction: float,
    local_strength: float,
    extent: float,
) -> np.ndarray:
    """Mix random deletion with two smooth local missing-cell clusters."""
    centers = points[rng.choice(len(points), size=2, replace=False)]
    scale = 0.11 * extent
    cluster = np.zeros(len(points), dtype=float)
    for center in centers:
        cluster = np.maximum(
            cluster,
            np.exp(-0.5 * np.sum(((points - center) / scale) ** 2, axis=1)),
        )
    if float(np.mean(cluster)) > 0:
        cluster = cluster / float(np.mean(cluster))
    probability = target_fraction * ((1.0 - local_strength) + local_strength * cluster)
    return np.clip(probability, 0.0, 0.92)


def generate_correspondence_sample(
    *,
    seed: int,
    difficulty: str,
    amplitude_um: float,
    missing_fraction: float,
    unmatched_fraction: float,
    overlap_fraction: float,
    noise_um: float,
    local_missing_strength: float,
    size: int = 128,
    spacing: float = 2.0,
    n_latent: int = 260,
) -> CorrespondenceSample:
    """Generate two affine-aligned slices while retaining persistent IDs.

    Fixed-specific and moving-specific nuclei receive non-overlapping IDs and
    therefore have no correspondence. Common nuclei share their latent ID.
    """
    if not 0 <= missing_fraction < 1:
        raise ValueError("missing_fraction must be in [0, 1).")
    if not 0 < overlap_fraction <= 1:
        raise ValueError("overlap_fraction must be in (0, 1].")
    rng = np.random.default_rng(seed)
    axes = np.arange(size, dtype=float) * spacing
    grid_x, grid_y = np.meshgrid(axes, axes)
    extent = float(axes[-1])
    field_x, field_y = _true_field(grid_x, grid_y, float(amplitude_um))

    latent_fixed = _sample_ellipse_points(rng, n_latent, extent)
    latent_ids = np.asarray([f"cell_{index:04d}" for index in range(n_latent)])
    moving_clean = _inverse_points(latent_fixed, field_x, field_y, spacing)

    if overlap_fraction < 1.0:
        overlap_cutoff = float(np.quantile(latent_fixed[:, 0], overlap_fraction))
        in_overlap = latent_fixed[:, 0] <= overlap_cutoff
    else:
        in_overlap = np.ones(n_latent, dtype=bool)
    deletion_probability = _spatial_missing_probability(
        latent_fixed, rng, missing_fraction, local_missing_strength, extent
    )
    keep_moving = in_overlap & (rng.random(n_latent) >= deletion_probability)

    common_points = moving_clean[keep_moving]
    common_points = common_points + rng.normal(0.0, noise_um, common_points.shape)
    common_ids = latent_ids[keep_moving]

    n_moving_added = int(round(max(len(common_points), 1) * unmatched_fraction))
    moving_added = (
        _sample_ellipse_points(rng, n_moving_added, extent)
        if n_moving_added
        else np.empty((0, 2), dtype=float)
    )
    moving_added_ids = np.asarray(
        [f"moving_new_{seed}_{index:04d}" for index in range(n_moving_added)]
    )
    moving_points = np.vstack([common_points, moving_added])
    moving_ids = np.concatenate([common_ids, moving_added_ids])
    moving_order = rng.permutation(len(moving_points))
    moving_points = moving_points[moving_order]
    moving_ids = moving_ids[moving_order]

    # Preserve unequal cardinality on both sides without manufacturing matches.
    n_fixed_added = max(8, int(round(0.06 * n_latent)))
    fixed_added = _sample_ellipse_points(rng, n_fixed_added, extent)
    fixed_added_ids = np.asarray(
        [f"fixed_new_{seed}_{index:04d}" for index in range(n_fixed_added)]
    )
    fixed_points = np.vstack([latent_fixed, fixed_added])
    fixed_ids = np.concatenate([latent_ids, fixed_added_ids])
    fixed_order = rng.permutation(len(fixed_points))
    fixed_points = fixed_points[fixed_order]
    fixed_ids = fixed_ids[fixed_order]

    fixed_lookup = {cell_id: index for index, cell_id in enumerate(fixed_ids)}
    true_fixed_index = np.asarray(
        [fixed_lookup.get(cell_id, -1) for cell_id in moving_ids], dtype=int
    )
    common_mask = true_fixed_index >= 0

    # Construct and exactly recover a nontrivial affine transform. Returned
    # coordinates are the recovered affine-aligned coordinates used by matching.
    theta = np.deg2rad(3.0)
    affine_matrix = np.array(
        [
            [0.985 * np.cos(theta), -1.015 * np.sin(theta) + 0.012],
            [0.985 * np.sin(theta), 1.015 * np.cos(theta)],
        ]
    )
    affine_translation = np.array([11.0, -7.0])
    raw_moving = (moving_points - affine_translation) @ np.linalg.inv(affine_matrix).T
    affine_aligned = raw_moving @ affine_matrix.T + affine_translation
    np.testing.assert_allclose(affine_aligned, moving_points, atol=1e-10, rtol=0)

    return CorrespondenceSample(
        fixed_points=fixed_points,
        moving_points=affine_aligned,
        fixed_ids=fixed_ids,
        moving_ids=moving_ids,
        true_fixed_index=true_fixed_index,
        common_moving_mask=common_mask,
        grid_x=grid_x,
        grid_y=grid_y,
        true_field_x=field_x,
        true_field_y=field_y,
        spacing=float(spacing),
        bounds=(0.0, 0.0, extent, extent),
        seed=int(seed),
        difficulty=str(difficulty),
        amplitude_um=float(amplitude_um),
        missing_fraction=float(missing_fraction),
        unmatched_fraction=float(unmatched_fraction),
        overlap_fraction=float(overlap_fraction),
        noise_um=float(noise_um),
        local_missing_strength=float(local_missing_strength),
    )


def assert_identity_alignment(sample: CorrespondenceSample) -> None:
    """Validate that every stored correspondence points to the same cell ID."""
    common = np.flatnonzero(sample.true_fixed_index >= 0)
    if not np.array_equal(
        sample.moving_ids[common], sample.fixed_ids[sample.true_fixed_index[common]]
    ):
        raise AssertionError("Persistent cell IDs and correspondence indices diverged.")
    unmatched = np.flatnonzero(sample.true_fixed_index < 0)
    if any(cell_id in set(sample.fixed_ids.tolist()) for cell_id in sample.moving_ids[unmatched]):
        raise AssertionError("An unmatched moving ID unexpectedly exists in the fixed slice.")

