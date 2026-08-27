import io
import zipfile

import numpy as np

from src.density_flow import density_flow_image_outputs
from src.pointset_registration import warp_he_image_to_world
from src.workflow_c_result import build_workflow_c_result_artifact, load_workflow_c_result_artifact
from src.workflow_d import (
    affine_result_from_artifact,
    build_workflow_d_export,
    fine_result_from_artifact,
    run_workflow_d_raster_deformation,
)


def _artifact(*, applied: bool, displacement: float):
    rows, columns = np.indices((8, 9), dtype=float)
    affine_points = np.array([[1.5, 1.5], [4.5, 2.5], [6.5, 5.5]])
    attempted_x = np.full(rows.shape, displacement, dtype=float)
    zeros = np.zeros(rows.shape, dtype=float)
    attempted_points = affine_points + np.array([displacement, 0.0])
    payload = build_workflow_c_result_artifact(
        fixed_geojson_points=attempted_points,
        original_moving_he_points=affine_points,
        affine_he_points=affine_points,
        attempted_registered_he_points=attempted_points,
        applied_registered_he_points=attempted_points if applied else affine_points,
        affine_matrix=np.eye(2),
        affine_translation=np.zeros(2),
        affine_flip_x=False,
        affine_flip_y=False,
        affine_image_width=9.0,
        affine_image_height=8.0,
        attempted_displacement_x=attempted_x,
        attempted_displacement_y=zeros,
        applied_displacement_x=attempted_x if applied else zeros,
        applied_displacement_y=zeros,
        grid_x=columns,
        grid_y=rows,
        field_bounds=(0.0, 0.0, 8.0, 7.0),
        field_spacing=1.0,
        fine_method="tissue-aware density flow",
        fine_applied=applied,
        output_pixel_size_um=1.0,
        output_origin="upper-left",
        inverse_iterations=12,
        inverse_tolerance_pixels=0.05,
        metrics={
            "affine_mean_residual": 1.0,
            "affine_median_residual": 1.0,
            "affine_n_pairs": 3,
            "affine_symmetric_median": 1.0,
            "final_applied_symmetric_median": 0.5,
            "rejection_reason": None if applied else "synthetic_rejection",
        },
        parameters={},
        provenance={"test": True},
    )
    return load_workflow_c_result_artifact(payload)


def _asymmetric_image():
    rows, columns = np.indices((8, 9), dtype=np.uint8)
    return np.stack([columns * 20, rows * 25, columns * 7 + rows * 3], axis=2)


def test_workflow_d_matches_established_density_flow_raster_path():
    artifact = _artifact(applied=True, displacement=0.5)
    image = _asymmetric_image()

    result = run_workflow_d_raster_deformation(image, artifact)
    affine, metadata = warp_he_image_to_world(
        image,
        affine_result_from_artifact(artifact),
        None,
        output_pixel_size_um=1.0,
        bounds=artifact.field_bounds,
        output_origin="upper-left",
    )
    legacy = density_flow_image_outputs(
        affine,
        metadata,
        fine_result_from_artifact(artifact),
        inverse_iterations=12,
        inverse_convergence_tolerance_pixels=0.05,
    )

    np.testing.assert_array_equal(result.affine_image, affine)
    np.testing.assert_array_equal(result.attempted_image, legacy["attempted"])
    np.testing.assert_array_equal(result.final_image, legacy["final"])
    assert result.raster_applied == legacy["raster_applied"]


def test_rejected_workflow_c_field_keeps_final_raster_affine_only():
    artifact = _artifact(applied=False, displacement=0.75)
    result = run_workflow_d_raster_deformation(_asymmetric_image(), artifact)

    assert not np.array_equal(result.attempted_image, result.affine_image)
    np.testing.assert_array_equal(result.final_image, result.affine_image)
    assert result.raster_applied is False
    assert result.raster_rejection_reason == "point_field_not_applied"


def test_zero_field_returns_identical_affine_attempted_and_final_images():
    artifact = _artifact(applied=True, displacement=0.0)
    result = run_workflow_d_raster_deformation(_asymmetric_image(), artifact)

    np.testing.assert_array_equal(result.attempted_image, result.affine_image)
    np.testing.assert_array_equal(result.final_image, result.affine_image)


def test_workflow_d_export_contains_images_and_qc_without_registration_recompute():
    artifact = _artifact(applied=True, displacement=0.5)
    result = run_workflow_d_raster_deformation(_asymmetric_image(), artifact)
    payload = build_workflow_d_export(result, artifact)

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = set(archive.namelist())

    assert {
        "images/affine_he.png",
        "images/attempted_warped_he.png",
        "images/final_applied_he.png",
        "images/checkerboard.png",
        "images/edge_overlay.png",
        "raster_qc.json",
        "warp_metadata.json",
        "workflow_c_manifest.json",
    } <= names
