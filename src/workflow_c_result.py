from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from src.workflow_c_run_export import json_safe


ARTIFACT_VERSION = "workflow-c-registration-result-v1"
REQUIRED_ARRAYS = (
    "fixed_geojson_points",
    "original_moving_he_points",
    "affine_he_points",
    "attempted_registered_he_points",
    "applied_registered_he_points",
    "affine_matrix",
    "affine_translation",
    "attempted_displacement_x",
    "attempted_displacement_y",
    "applied_displacement_x",
    "applied_displacement_y",
    "grid_x",
    "grid_y",
    "field_bounds",
    "field_spacing_um",
    "output_pixel_size_um",
    "coordinate_convention",
    "output_origin",
)


@dataclass(frozen=True)
class WorkflowCResultArtifact:
    arrays: dict[str, np.ndarray]
    metrics: dict[str, Any]
    parameters: dict[str, Any]
    provenance: dict[str, Any]
    manifest: dict[str, Any]

    @property
    def fine_method(self) -> str:
        return str(self.manifest.get("fine_method", "off"))

    @property
    def applied(self) -> bool:
        return bool(self.manifest.get("fine_applied", False))

    @property
    def field_spacing(self) -> float:
        return float(self.manifest["field_spacing_um"])

    @property
    def field_bounds(self) -> tuple[float, float, float, float]:
        return tuple(map(float, self.arrays["field_bounds"]))


def _validated_array(name: str, value: np.ndarray, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D.")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numeric values.")
    return array


def validate_workflow_c_result(artifact: WorkflowCResultArtifact) -> None:
    missing = sorted(set(REQUIRED_ARRAYS) - set(artifact.arrays))
    if missing:
        raise ValueError(f"Workflow C result is missing arrays: {missing}")
    for name in (
        "fixed_geojson_points", "original_moving_he_points", "affine_he_points",
        "attempted_registered_he_points", "applied_registered_he_points",
    ):
        points = _validated_array(name, artifact.arrays[name], ndim=2)
        if points.shape[1] != 2:
            raise ValueError(f"{name} must have shape (n, 2).")
    matrix = _validated_array("affine_matrix", artifact.arrays["affine_matrix"], ndim=2)
    translation = _validated_array("affine_translation", artifact.arrays["affine_translation"], ndim=1)
    if matrix.shape != (2, 2) or translation.shape != (2,):
        raise ValueError("Affine transform must contain a 2x2 matrix and length-2 translation.")
    field_shape = _validated_array("grid_x", artifact.arrays["grid_x"], ndim=2).shape
    for name in (
        "grid_y", "attempted_displacement_x", "attempted_displacement_y",
        "applied_displacement_x", "applied_displacement_y",
    ):
        if _validated_array(name, artifact.arrays[name], ndim=2).shape != field_shape:
            raise ValueError(f"{name} must match the displacement grid shape.")
    if _validated_array("field_bounds", artifact.arrays["field_bounds"], ndim=1).shape != (4,):
        raise ValueError("field_bounds must contain four values.")
    required_manifest = {
        "artifact_version", "fine_method", "fine_applied", "field_spacing_um",
        "output_pixel_size_um", "output_origin", "inverse_iterations",
        "inverse_tolerance_pixels", "affine_flip_x", "affine_flip_y",
        "affine_image_width", "affine_image_height",
    }
    missing_manifest = sorted(required_manifest - set(artifact.manifest))
    if missing_manifest:
        raise ValueError(f"Workflow C result is missing manifest values: {missing_manifest}")
    if artifact.manifest.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("Unsupported Workflow C result artifact version.")
    if artifact.field_spacing <= 0 or not np.isfinite(artifact.field_spacing):
        raise ValueError("field_spacing_um must be positive and finite.")
    min_x, min_y, max_x, max_y = artifact.field_bounds
    if max_x <= min_x or max_y <= min_y:
        raise ValueError("field_bounds must have positive width and height.")
    expected_shape = (
        int(np.ceil((max_y - min_y) / artifact.field_spacing)) + 1,
        int(np.ceil((max_x - min_x) / artifact.field_spacing)) + 1,
    )
    if field_shape != expected_shape:
        raise ValueError(
            "Displacement grid shape is inconsistent with field_bounds and field_spacing_um."
        )
    output_pixel_size = float(artifact.manifest["output_pixel_size_um"])
    if output_pixel_size <= 0 or not np.isfinite(output_pixel_size):
        raise ValueError("output_pixel_size_um must be positive and finite.")
    if artifact.manifest["output_origin"] not in {"upper-left", "upper-right", "lower-left"}:
        raise ValueError("Unsupported output_origin in Workflow C result.")
    if int(artifact.manifest["inverse_iterations"]) < 1:
        raise ValueError("inverse_iterations must be at least 1.")
    inverse_tolerance = float(artifact.manifest["inverse_tolerance_pixels"])
    if inverse_tolerance <= 0 or not np.isfinite(inverse_tolerance):
        raise ValueError("inverse_tolerance_pixels must be positive and finite.")
    if not np.isclose(float(artifact.arrays["field_spacing_um"]), artifact.field_spacing):
        raise ValueError("NPZ and manifest field spacing values do not match.")
    if not np.isclose(float(artifact.arrays["output_pixel_size_um"]), output_pixel_size):
        raise ValueError("NPZ and manifest output pixel size values do not match.")
    if str(artifact.arrays["output_origin"]) != str(artifact.manifest["output_origin"]):
        raise ValueError("NPZ and manifest output origin values do not match.")


def build_workflow_c_result_artifact(
    *,
    fixed_geojson_points: np.ndarray,
    original_moving_he_points: np.ndarray,
    affine_he_points: np.ndarray,
    attempted_registered_he_points: np.ndarray,
    applied_registered_he_points: np.ndarray,
    affine_matrix: np.ndarray,
    affine_translation: np.ndarray,
    affine_flip_x: bool,
    affine_flip_y: bool,
    affine_image_width: float,
    affine_image_height: float,
    attempted_displacement_x: np.ndarray,
    attempted_displacement_y: np.ndarray,
    applied_displacement_x: np.ndarray,
    applied_displacement_y: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    field_bounds: tuple[float, float, float, float],
    field_spacing: float,
    fine_method: str,
    fine_applied: bool,
    output_pixel_size_um: float,
    output_origin: str,
    inverse_iterations: int,
    inverse_tolerance_pixels: float,
    metrics: Mapping[str, Any],
    parameters: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> bytes:
    arrays = {
        "fixed_geojson_points": np.asarray(fixed_geojson_points),
        "original_moving_he_points": np.asarray(original_moving_he_points),
        "affine_he_points": np.asarray(affine_he_points),
        "attempted_registered_he_points": np.asarray(attempted_registered_he_points),
        "applied_registered_he_points": np.asarray(applied_registered_he_points),
        "affine_matrix": np.asarray(affine_matrix),
        "affine_translation": np.asarray(affine_translation),
        "attempted_displacement_x": np.asarray(attempted_displacement_x),
        "attempted_displacement_y": np.asarray(attempted_displacement_y),
        "applied_displacement_x": np.asarray(applied_displacement_x),
        "applied_displacement_y": np.asarray(applied_displacement_y),
        "grid_x": np.asarray(grid_x),
        "grid_y": np.asarray(grid_y),
        "field_bounds": np.asarray(field_bounds, dtype=float),
        "field_spacing_um": np.asarray(float(field_spacing)),
        "output_pixel_size_um": np.asarray(float(output_pixel_size_um)),
        "coordinate_convention": np.asarray(
            "world_xy_um; image pixels use col_row with row 0 defined by output_origin"
        ),
        "output_origin": np.asarray(str(output_origin)),
    }
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "fine_method": str(fine_method),
        "fine_applied": bool(fine_applied),
        "field_spacing_um": float(field_spacing),
        "coordinate_convention": "world_xy_um; image pixels use col_row with row 0 defined by output_origin",
        "output_pixel_size_um": float(output_pixel_size_um),
        "output_origin": str(output_origin),
        "inverse_iterations": int(inverse_iterations),
        "inverse_tolerance_pixels": float(inverse_tolerance_pixels),
        "affine_flip_x": bool(affine_flip_x),
        "affine_flip_y": bool(affine_flip_y),
        "affine_image_width": float(affine_image_width),
        "affine_image_height": float(affine_image_height),
        "registration_recalculation_required_for_raster_warp": False,
        "array_names": sorted(arrays),
    }
    artifact = WorkflowCResultArtifact(
        arrays=arrays,
        metrics=dict(metrics),
        parameters=dict(parameters),
        provenance=dict(provenance),
        manifest=manifest,
    )
    validate_workflow_c_result(artifact)
    array_buffer = io.BytesIO()
    np.savez_compressed(array_buffer, **arrays)
    files = {
        "registration_result.npz": array_buffer.getvalue(),
        "metrics.json": json.dumps(json_safe(metrics), indent=2, allow_nan=False).encode("utf-8"),
        "parameters.json": json.dumps(json_safe(parameters), indent=2, allow_nan=False).encode("utf-8"),
        "provenance.json": json.dumps(json_safe(provenance), indent=2, allow_nan=False).encode("utf-8"),
        "manifest.json": json.dumps(json_safe(manifest), indent=2, allow_nan=False).encode("utf-8"),
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return output.getvalue()


def load_workflow_c_result_artifact(source: bytes | bytearray | io.BytesIO) -> WorkflowCResultArtifact:
    payload = source.getvalue() if hasattr(source, "getvalue") else bytes(source)
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            required_files = {
                "registration_result.npz", "metrics.json", "parameters.json",
                "provenance.json", "manifest.json",
            }
            missing = sorted(required_files - set(archive.namelist()))
            if missing:
                raise ValueError(f"Workflow C result archive is missing files: {missing}")
            with np.load(io.BytesIO(archive.read("registration_result.npz")), allow_pickle=False) as values:
                arrays = {name: values[name].copy() for name in values.files}
            artifact = WorkflowCResultArtifact(
                arrays=arrays,
                metrics=json.loads(archive.read("metrics.json")),
                parameters=json.loads(archive.read("parameters.json")),
                provenance=json.loads(archive.read("provenance.json")),
                manifest=json.loads(archive.read("manifest.json")),
            )
    except (zipfile.BadZipFile, json.JSONDecodeError, OSError) as exc:
        raise ValueError("Invalid Workflow C result artifact.") from exc
    try:
        validate_workflow_c_result(artifact)
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Invalid Workflow C result artifact metadata.") from exc
    return artifact
