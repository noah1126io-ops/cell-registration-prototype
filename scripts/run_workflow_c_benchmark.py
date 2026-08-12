from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.density_flow import joint_density_tissue_structure_registration, tissue_aware_density_flow_registration
from src.registration_evaluation import (
    EVALUATION_VERSION,
    METRIC_DEFINITIONS_VERSION,
    deformation_validity_metrics,
    displacement_endpoint_error,
    landmark_tre_metrics,
    pointset_stage_metrics,
    sample_displacement_field,
)
from src.synthetic_registration_benchmark import generate_synthetic_registration_sample


METHODS = ("affine-only", "tissue-aware density flow", "Joint Flow")
DEFORMATIONS = (
    "identity", "smooth local translation", "Gaussian bulge", "Gaussian compression",
    "shear-like smooth deformation", "sinusoidal deformation",
)


def _run_method(sample, method: str):
    zeros = np.zeros_like(sample.grid_x)
    if method == "affine-only":
        return zeros, zeros, True, None
    common = dict(
        bounds=sample.bounds,
        density_pixel_size=sample.spacing,
        density_blur_scales=(8.0, 4.0, 2.0),
        optimization_levels=3,
        iterations_per_level=5,
        learning_rate=0.08,
        update_smoothing_sigma=4.0,
        detect_axis_reversal=False,
        global_translation_initialization="off",
        max_displacement=20.0,
        displacement_p95_limit=15.0,
        minimum_jacobian_p05=0.5,
        maximum_jacobian_p95=1.8,
        minimum_absolute_median_improvement=0.01,
        minimum_relative_median_improvement=0.001,
    )
    if method == "tissue-aware density flow":
        result = tissue_aware_density_flow_registration(sample.fixed_points, sample.moving_points, **common)
    elif method == "Joint Flow":
        result = joint_density_tissue_structure_registration(
            sample.fixed_points,
            sample.moving_points,
            affine_he_image=sample.moving_image,
            affine_he_tissue_mask=sample.moving_tissue_mask,
            affine_he_metadata=sample.metadata,
            stage_a_scales_um=(16.0, 8.0), stage_b_scales_um=(8.0, 4.0),
            stage_a_iterations=5, stage_b_iterations=5,
            **common,
        )
    else:
        raise ValueError(method)
    return (
        np.asarray(result.attempted_displacement_x),
        np.asarray(result.attempted_displacement_y),
        bool(result.applied),
        result.rejection_reason,
    )


def run_benchmark(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for deformation in args.deformations:
        for amplitude in args.amplitudes:
            for dropout in args.dropouts:
                sample = generate_synthetic_registration_sample(
                    deformation_type=deformation,
                    amplitude_um=amplitude,
                    dropout_fraction=dropout,
                    seed=args.seed,
                    size=args.size,
                    n_points=args.n_points,
                )
                for method in args.methods:
                    started = time.perf_counter()
                    field_x, field_y, applied, rejection = _run_method(sample, method)
                    runtime = time.perf_counter() - started
                    estimated_landmarks = sample.moving_landmarks + sample_displacement_field(
                        sample.moving_landmarks, field_x, field_y,
                        bounds=sample.bounds, spacing=sample.spacing,
                    )
                    _, tre = landmark_tre_metrics(
                        sample.fixed_landmarks, sample.moving_landmarks, estimated_landmarks
                    )
                    _, epe = displacement_endpoint_error(
                        field_x, field_y,
                        sample.ground_truth_displacement_x, sample.ground_truth_displacement_y,
                    )
                    transformed_points = sample.moving_points + sample_displacement_field(
                        sample.moving_points, field_x, field_y,
                        bounds=sample.bounds, spacing=sample.spacing,
                    )
                    internal = pointset_stage_metrics(sample.fixed_points, transformed_points)
                    deformation_metrics = deformation_validity_metrics(field_x, field_y, spacing=sample.spacing)
                    rows.append({
                        "deformation_type": deformation, "amplitude_um": amplitude,
                        "dropout_fraction": dropout, "method": method, "seed": args.seed,
                        "tre_median_um": tre["tre_median_after_um"],
                        "tre_p90_um": tre["tre_p90_after_um"],
                        "tre_p95_um": tre["tre_p95_after_um"],
                        "tre_max_um": tre["tre_max_after_um"],
                        "epe_median_um": epe["epe_median_um"],
                        "epe_p95_um": epe["epe_p95_um"],
                        "directional_error_median_degrees": epe["directional_error_median_degrees"],
                        "deformation_recovery_ratio": epe["deformation_recovery_ratio"],
                        "internal_symmetric_nn_median_um": internal["symmetric_nn_median_um"],
                        "jacobian_min": deformation_metrics["jacobian_min"],
                        "jacobian_max": deformation_metrics["jacobian_max"],
                        "fold_over_fraction": deformation_metrics["fold_over_fraction"],
                        "runtime_seconds": runtime,
                        "applied": applied, "rejected": method != "affine-only" and not applied,
                        "rejection_reason": rejection,
                    })
    results = pd.DataFrame(rows)
    summaries = []
    grouping_sets = [
        ("method",), ("method", "deformation_type"),
        ("method", "amplitude_um"), ("method", "dropout_fraction"),
    ]
    for group_columns in grouping_sets:
        for keys, group in results.groupby(list(group_columns), dropna=False):
            key_values = keys if isinstance(keys, tuple) else (keys,)
            row = {"summary_by": "+".join(group_columns)}
            row.update(dict(zip(group_columns, key_values)))
            row.update({
                "median_tre_um": float(group["tre_median_um"].median()),
                "median_p95_tre_um": float(group["tre_p95_um"].median()),
                "median_epe_um": float(group["epe_median_um"].median()),
                "median_p95_epe_um": float(group["epe_p95_um"].median()),
                "success_rate": float(group["applied"].mean()),
                "rejection_rate": float(group["rejected"].mean()),
                "fold_over_rate": float((group["fold_over_fraction"] > 0).mean()),
                "median_runtime_seconds": float(group["runtime_seconds"].median()),
            })
            summaries.append(row)
    return results, pd.DataFrame(summaries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Workflow C deterministic synthetic benchmark")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--n-points", type=int, default=240)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--deformations", nargs="+", choices=DEFORMATIONS, default=list(DEFORMATIONS))
    parser.add_argument("--amplitudes", nargs="+", type=float, default=[1.0, 2.0, 5.0, 10.0])
    parser.add_argument("--dropouts", nargs="+", type=float, default=[0.0, 0.05, 0.10, 0.20])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results, summary = run_benchmark(args)
    results.to_csv(args.output / "benchmark_results.csv", index=False)
    summary.to_csv(args.output / "benchmark_summary.csv", index=False)
    metadata = {
        "evaluation_version": EVALUATION_VERSION,
        "metric_definitions_version": METRIC_DEFINITIONS_VERSION,
        "seed": args.seed, "methods": args.methods, "deformations": args.deformations,
        "amplitudes_um": args.amplitudes, "dropout_fractions": args.dropouts,
        "note": "Synthetic ground-truth benchmark; no biological accuracy claim.",
    }
    (args.output / "benchmark_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
