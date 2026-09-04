from __future__ import annotations

import json
import os
import platform
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from src.array_backend import get_array_backend
from src.density_flow import (
    joint_density_tissue_structure_registration,
    tissue_aware_density_flow_registration,
)
from src.synthetic_registration_benchmark import generate_synthetic_registration_sample
from src.workflow_c_result import build_workflow_c_result_artifact, load_workflow_c_result_artifact
from src.workflow_c_slurm import update_status


def _require_slurm_cuda() -> tuple[str, dict[str, Any]]:
    job_id = os.environ.get("SLURM_JOB_ID")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not job_id:
        raise RuntimeError("Workflow C CUDA worker must run inside a Slurm allocation (SLURM_JOB_ID is missing).")
    if visible is None or not visible.strip() or visible.strip() in {"-1", "NoDevFiles"}:
        raise RuntimeError("Workflow C CUDA worker has no allocated GPU (CUDA_VISIBLE_DEVICES is empty).")
    backend = get_array_backend("cuda", dtype="float64")
    provenance = backend.provenance()
    backend.synchronize()
    return job_id, provenance


def _load_case(config: dict[str, Any], scratch: Path):
    synthetic = config.get("synthetic")
    if synthetic is not None:
        sample = generate_synthetic_registration_sample(**synthetic)
        return {
            "fixed": sample.fixed_points, "moving": sample.moving_points,
            "original_moving": sample.moving_points, "image": sample.moving_image,
            "mask": sample.moving_tissue_mask, "metadata": sample.metadata,
            "bounds": sample.bounds,
        }
    inputs = config.get("inputs", {})
    required = {"fixed_points", "moving_points"}
    missing = sorted(required - inputs.keys())
    if missing:
        raise ValueError(f"Missing Workflow C inputs: {missing}")
    fixed = np.load(scratch / inputs["fixed_points"], allow_pickle=False)
    moving = np.load(scratch / inputs["moving_points"], allow_pickle=False)
    image = np.load(scratch / inputs["he_image"], allow_pickle=False) if "he_image" in inputs else None
    mask = np.load(scratch / inputs["tissue_mask"], allow_pickle=False) if "tissue_mask" in inputs else None
    metadata = json.loads((scratch / inputs["metadata"]).read_text(encoding="utf-8")) if "metadata" in inputs else None
    bounds = tuple(config["bounds"]) if "bounds" in config else None
    optional_arrays = {}
    for key in ("original_fixed_points", "original_moving_points", "success_metric_fixed_points", "success_metric_moving_points"):
        if key in inputs:
            optional_arrays[key] = np.load(scratch / inputs[key], allow_pickle=False)
    return {
        "fixed": fixed, "moving": moving,
        "original_fixed": optional_arrays.get("original_fixed_points", fixed),
        "original_moving": optional_arrays.get("original_moving_points", moving),
        "success_metric_fixed": optional_arrays.get("success_metric_fixed_points"),
        "success_metric_moving": optional_arrays.get("success_metric_moving_points"),
        "image": image, "mask": mask, "metadata": metadata, "bounds": bounds,
    }


def execute_config(config: dict[str, Any], scratch: Path, slurm_provenance: dict[str, Any]) -> bytes:
    case = _load_case(config, scratch)
    fixed, moving = case["fixed"], case["moving"]
    image, mask, metadata, bounds = case["image"], case["mask"], case["metadata"], case["bounds"]
    method = config.get("fine_method", "joint density + tissue-structure flow")
    parameters = dict(config.get("parameters", {}))
    parameters.update(device="cuda", dtype="float64")
    if bounds is not None:
        parameters.setdefault("bounds", bounds)
    if case.get("success_metric_fixed") is not None:
        parameters["success_metric_fixed_points"] = case["success_metric_fixed"]
    if case.get("success_metric_moving") is not None:
        parameters["success_metric_moving_points"] = case["success_metric_moving"]
    if method == "joint density + tissue-structure flow":
        if image is None or mask is None or metadata is None:
            raise ValueError("Joint Flow requires he_image, tissue_mask, and metadata inputs.")
        result = joint_density_tissue_structure_registration(
            fixed, moving, affine_he_image=image, affine_he_tissue_mask=mask,
            affine_he_metadata=metadata, **parameters,
        )
    elif method == "tissue-aware density flow":
        result = tissue_aware_density_flow_registration(fixed, moving, **parameters)
    else:
        raise ValueError(f"CUDA worker does not support fine_method={method!r}")

    height = float(metadata.get("height", image.shape[0]) if metadata else np.ptp(moving[:, 1]) + 1.0)
    width = float(metadata.get("width", image.shape[1]) if metadata else np.ptp(moving[:, 0]) + 1.0)
    provenance = {
        **slurm_provenance,
        "SLURM_JOB_ID": os.environ["SLURM_JOB_ID"],
        "hostname": platform.node(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "backend": result.metrics.get("compute_backend"),
        "dtype": "float64",
    }
    metrics = dict(result.metrics)
    affine = config.get("affine", {})
    return build_workflow_c_result_artifact(
        fixed_geojson_points=case.get("original_fixed", fixed), original_moving_he_points=case["original_moving"],
        affine_he_points=moving, attempted_registered_he_points=result.attempted_transformed_points,
        applied_registered_he_points=result.transformed_points,
        affine_matrix=np.asarray(affine.get("matrix", np.eye(2)), dtype=float),
        affine_translation=np.asarray(affine.get("translation", np.zeros(2)), dtype=float),
        affine_flip_x=bool(affine.get("flip_x", False)), affine_flip_y=bool(affine.get("flip_y", False)),
        affine_image_width=float(affine.get("image_width", width)),
        affine_image_height=float(affine.get("image_height", height)),
        attempted_displacement_x=result.attempted_displacement_x,
        attempted_displacement_y=result.attempted_displacement_y,
        applied_displacement_x=result.displacement_x, applied_displacement_y=result.displacement_y,
        grid_x=result.grid_x, grid_y=result.grid_y, field_bounds=result.bounds,
        field_spacing=result.grid_spacing, fine_method=method, fine_applied=result.applied,
        output_pixel_size_um=float(metadata.get("output_pixel_size_um", 1.0) if metadata else 1.0),
        output_origin=str(metadata.get("output_origin", "upper-left") if metadata else "upper-left"),
        inverse_iterations=int(config.get("inverse_iterations", 12)),
        inverse_tolerance_pixels=float(config.get("inverse_tolerance_pixels", 0.05)),
        metrics=metrics, parameters={**config.get("parameters", {}), "device": "cuda", "dtype": "float64"},
        provenance=provenance,
    )


def run_worker(config_path: Path) -> Path:
    config_path = Path(config_path).resolve()
    run_dir = config_path.parent
    job_id = os.environ.get("SLURM_JOB_ID", "unknown")
    scratch_root = Path(os.environ.get("SLURM_TMPDIR") or f"/tmp/cellreg_{job_id}")
    scratch = scratch_root / run_dir.name
    try:
        job_id, provenance = _require_slurm_cuda()
        update_status(run_dir, "running", slurm_job_id=job_id, hostname=platform.node())
        if scratch.exists():
            raise FileExistsError(f"Scratch directory already exists: {scratch}")
        scratch.mkdir(parents=True)
        shutil.copy2(config_path, scratch / "config.json")
        if (run_dir / "input").exists():
            shutil.copytree(run_dir / "input", scratch / "input")
            for source in (run_dir / "input").iterdir():
                copied = scratch / "input" / source.name
                if not copied.is_file() or copied.stat().st_size != source.stat().st_size:
                    raise IOError(f"Scratch input copy verification failed: {source.name}")
        config = json.loads((scratch / "config.json").read_text(encoding="utf-8"))
        artifact = execute_config(config, scratch, provenance)
        load_workflow_c_result_artifact(artifact)
        result_dir = run_dir / "result"
        result_dir.mkdir(exist_ok=True)
        temporary = result_dir / ".workflow_c_registration_result.zip.tmp"
        destination = result_dir / "workflow_c_registration_result.zip"
        temporary.write_bytes(artifact)
        if temporary.stat().st_size != len(artifact):
            raise IOError("Final artifact copy size verification failed.")
        os.replace(temporary, destination)
        load_workflow_c_result_artifact(destination.read_bytes())
        update_status(run_dir, "completed", slurm_job_id=job_id, hostname=platform.node(),
                      artifact=str(destination), artifact_size_bytes=destination.stat().st_size,
                      gpu_model=provenance.get("gpu_model"), backend="cuda", dtype="float64")
        shutil.rmtree(scratch)
        return destination
    except BaseException as exc:
        update_status(run_dir, "failed", slurm_job_id=job_id, hostname=platform.node(), error=f"{type(exc).__name__}: {exc}")
        raise
