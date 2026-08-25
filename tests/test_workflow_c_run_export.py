import inspect
import io
import json
import zipfile
from datetime import datetime, timezone

import numpy as np

from app import show_mask_to_mask_workflow, show_point_registration_workflow
from src.workflow_c_run_export import (
    EXPECTED_ARTIFACTS,
    build_workflow_c_run_bundle,
    experiment_history_csv,
    generate_run_id,
    git_provenance,
    input_file_record,
    json_safe,
    parameter_changes,
    sanitize_filename_component,
)


def _bundle(**overrides):
    arguments = {
        "run_id": "workflow_c_20260722_120000_test",
        "label": "test",
        "notes": "QC run",
        "parameters": {"fine_method": "tissue-aware density flow", "sigma": np.float64(2.0)},
        "metrics": {
            "fine_status": "rejected",
            "affine_symmetric_median": np.float64(10.0),
            "attempted_symmetric_median": np.float64(10.2),
            "jacobian_min": np.float64(0.2),
            "unavailable_numeric": np.nan,
        },
        "artifacts": {
            "points/attempted_he_nuclei.csv": b"x,y\n1,2\n",
            "images/attempted_he.png": b"attempted-png",
            "images/affine_he.png": None,
        },
        "optimization_history": [{"iteration": 1, "accepted": False}],
        "provenance": {
            "git_commit_sha": "abc123",
            "git_dirty": True,
            "python_version": "3.test",
            "platform": "test-platform",
            "os": "test-os",
            "package_versions": {"numpy": "test"},
            "array_shapes": {"grid": [4, 5]},
            "point_counts": {"fixed": 3, "moving": 2},
            "coordinate_conventions": {"world": "xy um"},
        },
        "input_files": [input_file_record("HE nuclei.npy", b"npy-data")],
        "now_utc": datetime(2026, 7, 22, 3, 0, tzinfo=timezone.utc),
        "now_local": datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc),
    }
    arguments.update(overrides)
    return build_workflow_c_run_bundle(**arguments)


def test_valid_zip_generation_and_manifest_fields():
    bundle, manifest = _bundle()

    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        names = set(archive.namelist())
        archived_manifest = json.loads(archive.read("manifest.json"))

    assert zipfile.is_zipfile(io.BytesIO(bundle))
    assert {"run_summary.md", "manifest.json", "parameters.json", "metrics_summary.json"} <= names
    for field in (
        "run_id", "utc_timestamp", "local_timestamp", "git_commit_sha", "git_dirty",
        "python_version", "platform", "os", "package_versions", "input_files",
        "included_files", "skipped_unavailable_files", "array_shapes", "point_counts",
        "coordinate_conventions",
    ):
        assert field in archived_manifest
    assert archived_manifest == manifest


def test_unavailable_images_are_skipped_and_recorded():
    bundle, manifest = _bundle()

    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        assert "images/affine_he.png" not in archive.namelist()
    assert "images/affine_he.png" in manifest["skipped_unavailable_files"]


def test_parameter_diff_detects_numeric_delta_and_ratio():
    rows = parameter_changes(
        {"sigma": 4.0, "mode": "auto", "same": 1},
        {"sigma": 2.0, "mode": "off", "same": 1},
    )
    by_name = {row["parameter"]: row for row in rows}

    assert set(by_name) == {"sigma", "mode"}
    assert by_name["sigma"]["numeric_delta"] == 2.0
    assert by_name["sigma"]["numeric_ratio"] == 2.0
    assert by_name["mode"]["change_type"] == "changed"


def test_json_serialization_converts_numpy_scalars_and_nonfinite_to_null():
    converted = json_safe({"integer": np.int64(3), "value": np.float32(1.5), "missing": np.nan})
    encoded = json.dumps(converted, allow_nan=False)

    assert json.loads(encoded) == {"integer": 3, "value": 1.5, "missing": None}


def test_sha256_input_hashing():
    record = input_file_record("points.npy", b"abc")

    assert record["size_bytes"] == 3
    assert record["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_rejected_run_retains_attempted_outputs_and_numeric_nulls():
    bundle, _ = _bundle()

    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        assert archive.read("images/attempted_he.png") == b"attempted-png"
        metrics = json.loads(archive.read("metrics_summary.json"))

    assert metrics["fine_status"] == "rejected"
    assert isinstance(metrics["attempted_symmetric_median"], float)
    assert metrics["unavailable_numeric"] is None


def test_original_inputs_are_excluded_by_default():
    bundle, manifest = _bundle()

    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        assert not any(name.startswith("inputs/") for name in archive.namelist())
    assert manifest["include_original_inputs"] is False


def test_original_inputs_can_be_included_explicitly():
    bundle, manifest = _bundle(include_original_inputs=True)

    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        input_names = [name for name in archive.namelist() if name.startswith("inputs/")]
        assert len(input_names) == 1
        assert archive.read(input_names[0]) == b"npy-data"
    assert manifest["include_original_inputs"] is True


def test_run_and_input_filenames_are_sanitized_and_unique():
    now = datetime(2026, 7, 22, 12, 0)
    first = generate_run_id("patient / sample: 01", now=now)
    second = generate_run_id("patient / sample: 01", now=now, existing_run_ids=[first])

    assert first == "workflow_c_20260722_120000_patient-sample-01"
    assert second == f"{first}_2"
    assert sanitize_filename_component("../sensitive HE?.tif") == "sensitive-HE-.tif"
    assert "/" not in first and "\\" not in first


def test_workflow_a_and_b_do_not_reference_run_export():
    assert "workflow_c_run_export" not in inspect.getsource(show_point_registration_workflow)
    assert "workflow_c_run_export" not in inspect.getsource(show_mask_to_mask_workflow)


def test_git_provenance_fails_gracefully_outside_repository(tmp_path):
    provenance = git_provenance(tmp_path)

    assert provenance == {"git_commit_sha": None, "git_dirty": None}


def test_raster_qc_artifacts_are_exported_without_amplified_preview():
    artifacts = {
        "raster_warp_metrics.json": b"{}",
        "raster_warp_metrics.csv": b"region,value\nfull,1\n",
        "inverse_solver_history.csv": b"iteration,max_residual\n1,0.1\n",
        "local_region_metrics.csv": b"region_id,delta_median\n0,-0.2\n",
        "true_displacement_pixel_summary.json": b"{}",
        "images/affine_vs_warped_difference.png": b"difference",
        "images/checkerboard_comparison.png": b"checkerboard",
        "images/edge_overlay.png": b"edges",
        "images/roi_comparisons.png": b"rois",
    }
    bundle, _ = _bundle(artifacts=artifacts)

    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        names = set(archive.namelist())

    assert set(artifacts) <= names
    assert not any("amplified" in name or "exaggerated" in name for name in names)


def test_evaluation_artifacts_and_manifest_metadata_are_exported():
    artifacts = {
        "evaluation/evaluation_summary.json": b'{"tre":null,"count":3}',
        "evaluation/evaluation_summary.csv": b"domain,status\nIndependent,NOT AVAILABLE\n",
        "evaluation/pointset_metrics.csv": b"stage,value\naffine,1.2\n",
        "evaluation/deformation_metrics.csv": b"field,jacobian_min\napplied,1.0\n",
    }
    metadata = {
        "landmark_file_name": None,
        "landmark_file_sha256": None,
        "landmark_count": 0,
        "independent_validation_available": False,
        "evaluation_version": "test-v1",
        "local_block_size_um": 100.0,
        "metric_definitions_version": "test-definitions",
    }
    bundle, manifest = _bundle(artifacts=artifacts, evaluation_metadata=metadata)

    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        names = set(archive.namelist())
        evaluation_json = json.loads(archive.read("evaluation/evaluation_summary.json"))

    assert set(artifacts) <= names
    assert manifest["evaluation"] == metadata
    assert evaluation_json["tre"] is None
    assert isinstance(evaluation_json["count"], int)


def test_joint_stage_a_checkpoint_artifacts_are_declared_as_npz_and_qc_images():
    expected = set(EXPECTED_ARTIFACTS)
    assert {
        "fields/stage_a_point_metric_best.npz",
        "fields/stage_a_canonical_objective_best.npz",
        "fields/stage_a_strongest_exploratory_safe.npz",
        "fields/stage_a_final_accepted.npz",
        "fields/stage_a_selected_for_stage_b.npz",
        "images/stage_a_objective_best_warp_grid.png",
        "images/stage_a_strongest_safe_warp_grid.png",
        "images/stage_a_selected_displacement.png",
    } <= expected
    assert not any(path.endswith(".json") and "displacement_field" in path for path in expected)


def test_rejected_candidate_analysis_and_human_review_are_exported_without_arrays():
    rejection = {
        "method": "joint density + tissue-structure flow",
        "candidate_type": "rejected_nonlinear_qc_candidate",
        "checks": [{"check": "Jacobian p05", "observed": 0.772, "required_min": 0.8}],
    }
    review = {
        "review_status": "not_reviewed",
        "annotation_only": True,
        "changes_registration_result": False,
    }
    artifacts = {
        "evaluation/rejection_analysis.json": json.dumps(rejection).encode("utf-8"),
        "evaluation/rejection_analysis.csv": b"check,observed,required_min\nJacobian p05,0.772,0.8\n",
        "evaluation/attempted_candidate_metrics.csv": b"Metric,Affine,Attempted\nmedian,10,9.5\n",
        "review/human_qc_review.json": json.dumps(review).encode("utf-8"),
        "fields/attempted_displacement_field.npz": b"npz-field-payload",
    }
    bundle, _ = _bundle(artifacts=artifacts)

    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        names = set(archive.namelist())
        exported_rejection = json.loads(archive.read("evaluation/rejection_analysis.json"))
        exported_review = json.loads(archive.read("review/human_qc_review.json"))

    assert set(artifacts) <= names
    assert exported_rejection["checks"][0]["observed"] == 0.772
    assert exported_review["review_status"] == "not_reviewed"
    assert "displacement_x" not in json.dumps(exported_rejection)


def test_rejected_candidate_artifacts_and_history_fields_are_declared():
    expected = set(EXPECTED_ARTIFACTS)
    assert {
        "evaluation/rejection_analysis.json",
        "evaluation/rejection_analysis.csv",
        "evaluation/attempted_candidate_metrics.csv",
        "review/human_qc_review.json",
    } <= expected

    csv_text = experiment_history_csv(
        [
            {
                "run_id": "run-1",
                "final_status": "REJECTED",
                "attempted_alignment_status": "IMPROVED",
                "safety_status": "FAILED",
                "primary_rejection_reason": "jacobian_p05",
                "failed_gate_count": 1,
                "minimum_safety_margin": -0.028,
                "human_review": "Probably better",
                "human_review_note_present": True,
            }
        ]
    ).decode("utf-8")

    for field in (
        "final_status",
        "attempted_alignment_status",
        "safety_status",
        "primary_rejection_reason",
        "failed_gate_count",
        "minimum_safety_margin",
        "human_review",
        "human_review_note_present",
    ):
        assert field in csv_text.splitlines()[0]
