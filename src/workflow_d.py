from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass

import numpy as np

from src.density_flow import density_flow_deformation_diagnostics, density_flow_image_outputs
from src.export import array_to_png_bytes
from src.pointset_registration import AffineICPResult, FineWarpResult, warp_he_image_to_world
from src.raster_deformation_qc import checkerboard_comparison, edge_overlay, raster_difference_metrics
from src.workflow_c_result import WorkflowCResultArtifact
from src.workflow_c_run_export import json_safe


DENSITY_FLOW_METHODS = {
    "tissue-aware density flow",
    "joint density + tissue-structure flow",
}


@dataclass(frozen=True)
class WorkflowDRasterResult:
    affine_image: np.ndarray
    attempted_image: np.ndarray
    final_image: np.ndarray
    warp_metadata: dict
    attempted_inverse_diagnostics: dict | None
    raster_applied: bool
    raster_rejection_reason: str | None
    difference_metrics: dict
    checkerboard_image: np.ndarray
    edge_overlay_image: np.ndarray
    attempted_jacobian: np.ndarray
    applied_jacobian: np.ndarray


def affine_result_from_artifact(artifact: WorkflowCResultArtifact) -> AffineICPResult:
    arrays = artifact.arrays
    manifest = artifact.manifest
    return AffineICPResult(
        affine_matrix=np.asarray(arrays["affine_matrix"], dtype=float),
        translation=np.asarray(arrays["affine_translation"], dtype=float),
        transformed_points=np.asarray(arrays["affine_he_points"], dtype=float),
        flip_x=bool(manifest["affine_flip_x"]),
        flip_y=bool(manifest["affine_flip_y"]),
        image_width=float(manifest["affine_image_width"]),
        image_height=float(manifest["affine_image_height"]),
        mean_residual=float(artifact.metrics.get("affine_mean_residual", np.nan)),
        median_residual=float(artifact.metrics.get("affine_median_residual", np.nan)),
        n_pairs=int(artifact.metrics.get("affine_n_pairs", 0)),
        success=True,
        message="Reconstructed from Workflow C result artifact.",
    )


def fine_result_from_artifact(artifact: WorkflowCResultArtifact) -> FineWarpResult:
    arrays = artifact.arrays
    attempted_diagnostics = density_flow_deformation_diagnostics(
        arrays["attempted_displacement_x"],
        arrays["attempted_displacement_y"],
        pixel_size=artifact.field_spacing,
    )
    return FineWarpResult(
        transformed_points=np.asarray(arrays["applied_registered_he_points"], dtype=float),
        grid_x=np.asarray(arrays["grid_x"], dtype=float),
        grid_y=np.asarray(arrays["grid_y"], dtype=float),
        displacement_x=np.asarray(arrays["applied_displacement_x"], dtype=float),
        displacement_y=np.asarray(arrays["applied_displacement_y"], dtype=float),
        bounds=artifact.field_bounds,
        grid_spacing=artifact.field_spacing,
        jacobian_min=float(np.min(attempted_diagnostics["jacobian"])),
        jacobian_max=float(np.max(attempted_diagnostics["jacobian"])),
        max_displacement=float(np.max(attempted_diagnostics["raw_magnitude"])),
        n_candidate_pairs=0,
        n_pairs=len(arrays["applied_registered_he_points"]) if artifact.applied else 0,
        n_filtered_pairs=0,
        median_pair_distance_before=float(artifact.metrics.get("affine_symmetric_median", np.nan)),
        median_pair_distance_after=float(artifact.metrics.get("final_applied_symmetric_median", np.nan)),
        success=artifact.applied,
        message="Reconstructed from Workflow C result artifact.",
        attempted_transformed_points=np.asarray(arrays["attempted_registered_he_points"], dtype=float),
        attempted_displacement_x=np.asarray(arrays["attempted_displacement_x"], dtype=float),
        attempted_displacement_y=np.asarray(arrays["attempted_displacement_y"], dtype=float),
        attempted_metrics=artifact.metrics.get("attempted_metrics"),
        applied_metrics=artifact.metrics.get("applied_metrics"),
        rejection_reason=artifact.metrics.get("rejection_reason"),
        applied=artifact.applied,
        metrics=None,
    )


def run_workflow_d_raster_deformation(
    he_image: np.ndarray,
    artifact: WorkflowCResultArtifact,
) -> WorkflowDRasterResult:
    """Apply an existing Workflow C transform without recomputing registration."""
    image = np.asarray(he_image)
    if image.ndim not in {2, 3}:
        raise ValueError("HE image must be a grayscale or RGB/RGBA array.")
    affine_result = affine_result_from_artifact(artifact)
    fine_result = fine_result_from_artifact(artifact)
    manifest = artifact.manifest
    affine_image, warp_metadata = warp_he_image_to_world(
        image,
        affine_result,
        None,
        output_pixel_size_um=float(manifest["output_pixel_size_um"]),
        bounds=artifact.field_bounds,
        output_origin=str(manifest["output_origin"]),
    )
    inverse_diagnostics = None
    raster_rejection_reason = None
    if artifact.fine_method in DENSITY_FLOW_METHODS:
        outputs = density_flow_image_outputs(
            affine_image,
            warp_metadata,
            fine_result,
            inverse_iterations=int(manifest["inverse_iterations"]),
            inverse_convergence_tolerance_pixels=float(manifest["inverse_tolerance_pixels"]),
        )
        attempted_image = outputs["attempted"]
        final_image = outputs["final"]
        inverse_diagnostics = outputs["attempted_inverse_diagnostics"]
        raster_applied = bool(outputs["raster_applied"])
        raster_rejection_reason = outputs["raster_rejection_reason"]
    else:
        attempted_preview = FineWarpResult(
            **{
                **fine_result.__dict__,
                "transformed_points": fine_result.attempted_transformed_points,
                "displacement_x": fine_result.attempted_displacement_x,
                "displacement_y": fine_result.attempted_displacement_y,
                "success": True,
                "applied": True,
            }
        )
        attempted_image, _ = warp_he_image_to_world(
            image,
            affine_result,
            attempted_preview,
            output_pixel_size_um=float(manifest["output_pixel_size_um"]),
            bounds=artifact.field_bounds,
            output_origin=str(manifest["output_origin"]),
        )
        if artifact.applied:
            final_image, _ = warp_he_image_to_world(
                image,
                affine_result,
                fine_result,
                output_pixel_size_um=float(manifest["output_pixel_size_um"]),
                bounds=artifact.field_bounds,
                output_origin=str(manifest["output_origin"]),
            )
            raster_applied = True
        else:
            final_image = affine_image.copy()
            raster_applied = False
            raster_rejection_reason = "point_field_not_applied"

    attempted_field = density_flow_deformation_diagnostics(
        artifact.arrays["attempted_displacement_x"],
        artifact.arrays["attempted_displacement_y"],
        pixel_size=artifact.field_spacing,
    )
    applied_field = density_flow_deformation_diagnostics(
        artifact.arrays["applied_displacement_x"],
        artifact.arrays["applied_displacement_y"],
        pixel_size=artifact.field_spacing,
    )
    return WorkflowDRasterResult(
        affine_image=affine_image,
        attempted_image=attempted_image,
        final_image=final_image,
        warp_metadata=warp_metadata,
        attempted_inverse_diagnostics=inverse_diagnostics,
        raster_applied=raster_applied,
        raster_rejection_reason=raster_rejection_reason,
        difference_metrics=raster_difference_metrics(affine_image, attempted_image),
        checkerboard_image=checkerboard_comparison(affine_image, attempted_image),
        edge_overlay_image=edge_overlay(affine_image, attempted_image),
        attempted_jacobian=np.asarray(attempted_field["jacobian"]),
        applied_jacobian=np.asarray(applied_field["jacobian"]),
    )


def build_workflow_d_export(result: WorkflowDRasterResult, artifact: WorkflowCResultArtifact) -> bytes:
    files = {
        "images/affine_he.png": array_to_png_bytes(result.affine_image),
        "images/attempted_warped_he.png": array_to_png_bytes(result.attempted_image),
        "images/final_applied_he.png": array_to_png_bytes(result.final_image),
        "images/checkerboard.png": array_to_png_bytes(result.checkerboard_image),
        "images/edge_overlay.png": array_to_png_bytes(result.edge_overlay_image),
        "raster_qc.json": json.dumps(json_safe(result.difference_metrics), indent=2, allow_nan=False).encode("utf-8"),
        "warp_metadata.json": json.dumps(json_safe(result.warp_metadata), indent=2, allow_nan=False).encode("utf-8"),
        "workflow_c_manifest.json": json.dumps(json_safe(artifact.manifest), indent=2, allow_nan=False).encode("utf-8"),
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return output.getvalue()
