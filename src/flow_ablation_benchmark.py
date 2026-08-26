from __future__ import annotations

import json
import time
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from src.density_flow import _jacobian, _sample_field, tissue_aware_density_flow_registration
from src.flow_ablation import FlowAblationConfig, ablation_preset
from src.pointset_registration import point_bidirectional_distance_metrics
from src.synthetic_registration_benchmark import (
    generate_synthetic_registration_sample,
    ground_truth_displacement,
)


STANDARD_ABLATIONS = (
    "Full", "-Density", "-Support", "-Structure", "-Smoothness",
    "-Magnitude", "-Boundary", "-SoftJacobian",
)


def moving_dropout_indices(
    points: np.ndarray,
    fraction: float,
    *,
    seed: int,
    spatially_biased: bool = False,
) -> np.ndarray:
    if not 0.0 <= fraction < 1.0:
        raise ValueError("dropout fraction must be in [0, 1).")
    count = len(points)
    keep_count = max(1, int(round(count * (1.0 - fraction))))
    if spatially_biased:
        order = np.argsort(np.asarray(points)[:, 0])
        return np.sort(order[:keep_count])
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=keep_count, replace=False))


def run_ablation_benchmark(
    *,
    ablations: Iterable[str] = STANDARD_ABLATIONS,
    dropout_fractions: Iterable[float] = (0.0, 0.10, 0.25, 0.50, 0.75),
    include_spatially_biased_dropout: bool = True,
    seed: int = 42,
    size: int = 96,
    iterations_per_level: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run reproducible synthetic ablations; results are methodological diagnostics only."""
    sample = generate_synthetic_registration_sample(
        deformation_type="smooth local translation",
        amplitude_um=4.0,
        dropout_fraction=0.0,
        seed=seed,
        size=size,
        n_points=140,
    )
    cases = [(float(fraction), False) for fraction in dropout_fractions]
    if include_spatially_biased_dropout:
        cases.extend((float(fraction), True) for fraction in dropout_fractions if fraction > 0)
    rows = []
    grid_side = int(np.ceil((sample.bounds[2] - sample.bounds[0]) / 2.0)) + 1
    sample_rows = np.linspace(0.0, sample.fixed_image.shape[0] - 1, grid_side)
    sample_cols = np.linspace(0.0, sample.fixed_image.shape[1] - 1, grid_side)
    sample_grid_rows, sample_grid_cols = np.meshgrid(sample_rows, sample_cols, indexing="ij")
    fixed_support_grid = map_coordinates(
        sample.ground_truth_warped_mask.astype(float), [sample_grid_rows, sample_grid_cols], order=1, mode="nearest"
    )
    fixed_gray = np.asarray(sample.fixed_image, dtype=float)
    if fixed_gray.ndim == 3:
        fixed_gray = np.mean(fixed_gray, axis=2)
    fixed_structure_grid = map_coordinates(
        fixed_gray, [sample_grid_rows, sample_grid_cols], order=1, mode="nearest"
    )
    fixed_structure_grid = (fixed_structure_grid - fixed_structure_grid.min()) / max(
        float(np.ptp(fixed_structure_grid)), np.finfo(float).eps
    )
    for dropout_fraction, biased in cases:
        keep = moving_dropout_indices(
            sample.moving_points, dropout_fraction, seed=seed + 17, spatially_biased=biased
        )
        moving = sample.moving_points[keep]
        for label in ablations:
            config = ablation_preset(label, base=FlowAblationConfig())
            started = time.perf_counter()
            result = tissue_aware_density_flow_registration(
                sample.fixed_points,
                moving,
                bounds=sample.bounds,
                density_pixel_size=2.0,
                density_blur_scales=(4.0, 2.0),
                optimization_levels=2,
                iterations_per_level=iterations_per_level,
                learning_rate=0.08,
                update_smoothing_sigma=3.0,
                max_displacement=15.0,
                displacement_p95_limit=12.0,
                detect_axis_reversal=False,
                minimum_absolute_median_improvement=0.01,
                minimum_relative_median_improvement=0.001,
                moving_tissue_mask=sample.moving_tissue_mask,
                moving_tissue_metadata=sample.metadata,
                moving_structure_image=sample.moving_image,
                fixed_support_feature_grid=fixed_support_grid,
                fixed_structure_feature_grid=fixed_structure_grid,
                tissue_support_channel_weight=0.25,
                structure_channel_weight=0.15,
                ablation_config=config,
            )
            runtime = time.perf_counter() - started
            attempted_x = np.asarray(result.attempted_displacement_x)
            attempted_y = np.asarray(result.attempted_displacement_y)
            transformed = np.asarray(result.attempted_transformed_points)
            point_metrics = point_bidirectional_distance_metrics(sample.fixed_points, transformed)
            bidirectional_distances = np.concatenate([
                cKDTree(sample.fixed_points).query(transformed, k=1)[0],
                cKDTree(transformed).query(sample.fixed_points, k=1)[0],
            ])
            transformed_landmarks = sample.moving_landmarks + _sample_field(
                sample.moving_landmarks,
                attempted_x,
                attempted_y,
                result.bounds,
                result.grid_spacing,
            )
            tre = np.linalg.norm(transformed_landmarks - sample.fixed_landmarks, axis=1)
            true_x, true_y = ground_truth_displacement(
                result.grid_x,
                result.grid_y,
                sample.deformation_type,
                sample.amplitude_um,
            )
            epe = np.hypot(attempted_x - true_x, attempted_y - true_y)
            jacobian = _jacobian(attempted_x, attempted_y, result.grid_spacing)
            local_summary = (result.metrics or {}).get("local_region_summary", {})
            rows.append({
                "ablation": label,
                "dropout_fraction": dropout_fraction,
                "dropout_mode": "spatially_biased" if biased else "random",
                "n_fixed": len(sample.fixed_points),
                "n_moving": len(moving),
                "tre_median_um": float(np.median(tre)),
                "tre_p95_um": float(np.percentile(tre, 95)),
                "symmetric_median_um": point_metrics["symmetric_median_distance"],
                "p90_um": float(np.percentile(bidirectional_distances, 90)),
                "p95_um": float(np.percentile(bidirectional_distances, 95)),
                "within_3": point_metrics.get("symmetric_within_3"),
                "within_5": point_metrics.get("symmetric_within_5"),
                "within_10": point_metrics.get("symmetric_within_10"),
                "mutual_nearest": (result.metrics or {}).get("attempted", {}).get("mutual_nearest_fraction"),
                "local_improved_fraction": local_summary.get("fraction_blocks_improved"),
                "local_worsened_fraction": local_summary.get("fraction_blocks_worsened"),
                "displacement_epe_median_um": float(np.median(epe)),
                "displacement_epe_p95_um": float(np.percentile(epe, 95)),
                "displacement_p95_um": float(np.percentile(np.hypot(attempted_x, attempted_y), 95)),
                "displacement_max_um": float(np.max(np.hypot(attempted_x, attempted_y))),
                "jacobian_p05": float(np.percentile(jacobian, 5)),
                "jacobian_median": float(np.median(jacobian)),
                "jacobian_p95": float(np.percentile(jacobian, 95)),
                "fold_over_fraction": float(np.mean(jacobian <= 0)),
                "runtime_seconds": runtime,
                "status": "applied" if result.applied else "rejected",
                "rejection_reason": result.rejection_reason,
            })
    results = pd.DataFrame(rows)
    numeric = [
        column for column in results.select_dtypes(include=[np.number]).columns
        if column != "dropout_fraction"
    ]
    summary = (
        results.groupby(["ablation", "dropout_mode"], dropna=False)[numeric]
        .median(numeric_only=True)
        .reset_index()
    )
    metadata = {
        "seed": seed,
        "synthetic_case": sample.deformation_type,
        "amplitude_um": sample.amplitude_um,
        "ablations": list(ablations),
        "dropout_fractions": list(map(float, dropout_fractions)),
        "spatially_biased_dropout_included": include_spatially_biased_dropout,
        "interpretation": "Synthetic methodological benchmark; not evidence of biological necessity or accuracy.",
    }
    return results, summary, metadata


def ablation_benchmark_artifacts(results: pd.DataFrame, summary: pd.DataFrame, metadata: dict) -> dict[str, bytes]:
    return {
        "ablation_results.csv": results.to_csv(index=False).encode("utf-8"),
        "ablation_summary.csv": summary.to_csv(index=False).encode("utf-8"),
        "ablation_metadata.json": json.dumps(metadata, indent=2, allow_nan=False).encode("utf-8"),
    }
