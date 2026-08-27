import io
import json
import zipfile

import numpy as np
import pytest

from src.workflow_c_result import (
    ARTIFACT_VERSION,
    build_workflow_c_result_artifact,
    load_workflow_c_result_artifact,
)


def _artifact_bytes(*, applied: bool = True) -> bytes:
    rows, columns = np.indices((6, 7), dtype=float)
    fixed = np.array([[1.0, 1.0], [4.0, 2.0], [5.0, 4.0]])
    affine = fixed - np.array([0.5, 0.0])
    attempted = affine + np.array([0.5, 0.0])
    applied_points = attempted if applied else affine
    attempted_x = np.full(rows.shape, 0.5)
    zeros = np.zeros(rows.shape)
    return build_workflow_c_result_artifact(
        fixed_geojson_points=fixed,
        original_moving_he_points=affine,
        affine_he_points=affine,
        attempted_registered_he_points=attempted,
        applied_registered_he_points=applied_points,
        affine_matrix=np.eye(2),
        affine_translation=np.zeros(2),
        affine_flip_x=False,
        affine_flip_y=False,
        affine_image_width=7.0,
        affine_image_height=6.0,
        attempted_displacement_x=attempted_x,
        attempted_displacement_y=zeros,
        applied_displacement_x=attempted_x if applied else zeros,
        applied_displacement_y=zeros,
        grid_x=columns,
        grid_y=rows,
        field_bounds=(0.0, 0.0, 6.0, 5.0),
        field_spacing=1.0,
        fine_method="tissue-aware density flow",
        fine_applied=applied,
        output_pixel_size_um=1.0,
        output_origin="upper-left",
        inverse_iterations=12,
        inverse_tolerance_pixels=0.05,
        metrics={"affine_symmetric_median": np.float32(1.5)},
        parameters={"maximum_final_displacement": np.int64(35)},
        provenance={"source": "synthetic"},
    )


def test_workflow_c_result_round_trip_preserves_registration_state():
    artifact = load_workflow_c_result_artifact(_artifact_bytes())

    assert artifact.manifest["artifact_version"] == ARTIFACT_VERSION
    assert artifact.fine_method == "tissue-aware density flow"
    assert artifact.applied is True
    assert artifact.field_bounds == (0.0, 0.0, 6.0, 5.0)
    np.testing.assert_allclose(
        artifact.arrays["attempted_registered_he_points"],
        artifact.arrays["fixed_geojson_points"],
    )
    assert artifact.metrics["affine_symmetric_median"] == pytest.approx(1.5)
    assert artifact.parameters["maximum_final_displacement"] == 35


def test_workflow_c_result_archive_keeps_fields_in_npz_not_json():
    payload = _artifact_bytes()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert "registration_result.npz" in archive.namelist()
        manifest = json.loads(archive.read("manifest.json"))
        metrics_text = archive.read("metrics.json").decode("utf-8")

    assert "attempted_displacement_x" in manifest["array_names"]
    assert "coordinate_convention" in manifest["array_names"]
    assert "output_pixel_size_um" in manifest["array_names"]
    assert "[[0.5" not in metrics_text


def test_workflow_c_result_rejects_incomplete_archive():
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("manifest.json", "{}")

    with pytest.raises(ValueError, match="missing files"):
        load_workflow_c_result_artifact(output.getvalue())
