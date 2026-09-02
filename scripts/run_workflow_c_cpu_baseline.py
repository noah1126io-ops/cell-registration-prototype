from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.density_flow import joint_density_tissue_structure_registration
from src.pointset_registration import estimate_affine_with_y_flip, point_bidirectional_distance_metrics
from src.registration_evaluation import deformation_validity_metrics
from src.runtime_instrumentation import RuntimeRecorder
from src.synthetic_registration_benchmark import generate_synthetic_registration_sample
from src.workflow_c_result import build_workflow_c_result_artifact


def run_representative_cpu_baseline() -> tuple[dict, bytes]:
    """Run one deterministic CPU Joint Flow case and return timings plus a valid artifact."""
    runtime = RuntimeRecorder()

    started = runtime.start()
    sample = generate_synthetic_registration_sample(
        deformation_type="Gaussian bulge",
        amplitude_um=5.0,
        dropout_fraction=0.10,
        seed=42,
        size=128,
        n_points=240,
    )
    runtime.stop("input_preprocessing", started)

    started = runtime.start()
    affine = estimate_affine_with_y_flip(
        sample.moving_points,
        sample.fixed_points,
        image_height_px=128.0,
        image_width_px=128.0,
        flip_candidates=((False, False),),
    )
    runtime.stop("affine_icp", started)

    result = joint_density_tissue_structure_registration(
        sample.fixed_points,
        affine.transformed_points,
        affine_he_image=sample.moving_image,
        affine_he_tissue_mask=sample.moving_tissue_mask,
        affine_he_metadata=sample.metadata,
        bounds=sample.bounds,
        density_pixel_size=sample.spacing,
        stage_a_scales_um=(16.0, 8.0),
        stage_b_scales_um=(8.0, 4.0),
        stage_a_iterations=8,
        stage_b_iterations=10,
        detect_axis_reversal=False,
        minimum_absolute_median_improvement=0.01,
        minimum_relative_median_improvement=0.001,
    )
    runtime.stages.update(result.metrics.get("runtime_breakdown", {}))

    attempted_points = np.asarray(result.attempted_transformed_points, dtype=float)
    applied_points = np.asarray(result.transformed_points, dtype=float)
    started = runtime.start()
    point_metrics = {
        "affine": point_bidirectional_distance_metrics(sample.fixed_points, affine.transformed_points),
        "attempted": point_bidirectional_distance_metrics(sample.fixed_points, attempted_points),
        "applied": point_bidirectional_distance_metrics(sample.fixed_points, applied_points),
    }
    runtime.stop("point_metrics_export", started)

    started = runtime.start()
    safety = deformation_validity_metrics(
        np.asarray(result.attempted_displacement_x),
        np.asarray(result.attempted_displacement_y),
        spacing=result.grid_spacing,
    )
    runtime.stop("jacobian_safety_qc_export", started)

    metrics = {
        "compute_backend": "cpu",
        "dtype": "float64",
        "point_metrics": point_metrics,
        "safety": safety,
        "runtime_breakdown": runtime.snapshot(),
    }
    started = runtime.start()
    artifact = build_workflow_c_result_artifact(
        fixed_geojson_points=sample.fixed_points,
        original_moving_he_points=sample.moving_points,
        affine_he_points=affine.transformed_points,
        attempted_registered_he_points=attempted_points,
        applied_registered_he_points=applied_points,
        affine_matrix=affine.affine_matrix,
        affine_translation=affine.translation,
        affine_flip_x=affine.flip_x,
        affine_flip_y=affine.flip_y,
        affine_image_width=affine.image_width,
        affine_image_height=affine.image_height,
        attempted_displacement_x=np.asarray(result.attempted_displacement_x),
        attempted_displacement_y=np.asarray(result.attempted_displacement_y),
        applied_displacement_x=np.asarray(result.displacement_x),
        applied_displacement_y=np.asarray(result.displacement_y),
        grid_x=result.grid_x,
        grid_y=result.grid_y,
        field_bounds=result.bounds,
        field_spacing=result.grid_spacing,
        fine_method="joint density + tissue-structure flow",
        fine_applied=bool(result.applied),
        output_pixel_size_um=1.0,
        output_origin="upper-left",
        inverse_iterations=12,
        inverse_tolerance_pixels=0.05,
        metrics=metrics,
        parameters={"baseline_case": "synthetic_gaussian_bulge", "device": "cpu"},
        provenance={"hostname": platform.node(), "python": platform.python_version()},
    )
    metrics["runtime_breakdown"]["artifact_creation_export"] = runtime.stop(
        "artifact_creation_export", started
    )
    metrics["runtime_breakdown"]["total_runtime_seconds"] = runtime.snapshot()["total_runtime_seconds"]
    metrics["artifact_size_bytes"] = len(artifact)
    return metrics, artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one deterministic Workflow C CPU baseline.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-output", type=Path)
    args = parser.parse_args()

    metrics, artifact = run_representative_cpu_baseline()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    if args.artifact_output is not None:
        args.artifact_output.parent.mkdir(parents=True, exist_ok=True)
        args.artifact_output.write_bytes(artifact)
    print(json.dumps(metrics["runtime_breakdown"], indent=2))


if __name__ == "__main__":
    main()
