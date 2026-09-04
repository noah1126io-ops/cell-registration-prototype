import inspect
from pathlib import Path

import numpy as np
import pytest

from app import show_he_geojson_preparation, show_mask_to_mask_workflow, show_point_registration_workflow
from src.workflow_c_result import build_workflow_c_result_artifact
from src.workflow_c_slurm import update_status
from src.workflow_c_slurm_ui import activate_run, load_run_view, safe_run_dir


def _artifact() -> bytes:
    points = np.array([[1.0, 1.0], [3.0, 2.0], [5.0, 4.0]])
    grid_x, grid_y = np.meshgrid(np.arange(7.0), np.arange(6.0))
    zeros = np.zeros_like(grid_x)
    return build_workflow_c_result_artifact(
        fixed_geojson_points=points, original_moving_he_points=points,
        affine_he_points=points, attempted_registered_he_points=points,
        applied_registered_he_points=points, affine_matrix=np.eye(2),
        affine_translation=np.zeros(2), affine_flip_x=False, affine_flip_y=False,
        affine_image_width=7, affine_image_height=6,
        attempted_displacement_x=zeros, attempted_displacement_y=zeros,
        applied_displacement_x=zeros, applied_displacement_y=zeros,
        grid_x=grid_x, grid_y=grid_y, field_bounds=(0.0, 0.0, 6.0, 5.0),
        field_spacing=1.0, fine_method="joint density + tissue-structure flow",
        fine_applied=True, output_pixel_size_um=1.0, output_origin="upper-left",
        inverse_iterations=12, inverse_tolerance_pixels=0.05,
        metrics={"safety": {}}, parameters={"device": "cuda"},
        provenance={"backend": "cuda", "hostname": "gpu-node"},
    )


@pytest.mark.parametrize("state", ["queued", "running", "failed"])
def test_run_view_renders_application_states_from_status(tmp_path, state):
    run = tmp_path / f"run-{state}"
    run.mkdir()
    update_status(run, state, slurm_job_id="123", error="failure" if state == "failed" else None)
    if state == "failed":
        (run / "stdout.log").write_text("short stdout")
        (run / "stderr.log").write_text("short stderr")

    view = load_run_view(tmp_path, run.name)

    assert view["state"] == state
    assert view["status"]["slurm_job_id"] == "123"
    if state == "failed":
        assert view["stderr.log"] == "short stderr"


def test_completed_run_exposes_valid_artifact(tmp_path):
    run = tmp_path / "complete"
    (run / "result").mkdir(parents=True)
    (run / "result" / "workflow_c_registration_result.zip").write_bytes(_artifact())
    update_status(run, "completed", slurm_job_id="456")

    view = load_run_view(tmp_path, run.name)

    assert view["artifact"].fine_method == "joint density + tissue-structure flow"
    assert view["artifact"].provenance["backend"] == "cuda"
    assert view["artifact_bytes"]


@pytest.mark.parametrize("payload", [None, b"not a zip"])
def test_completed_run_handles_missing_or_invalid_artifact(tmp_path, payload):
    run = tmp_path / "broken"
    (run / "result").mkdir(parents=True)
    if payload is not None:
        (run / "result" / "workflow_c_registration_result.zip").write_bytes(payload)
    update_status(run, "completed")

    assert "artifact_error" in load_run_view(tmp_path, run.name)


def test_run_recovery_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError, match="direct child"):
        safe_run_dir(tmp_path, "../outside")
    with pytest.raises(ValueError, match="direct child"):
        safe_run_dir(tmp_path, "/etc")


def test_activate_run_stores_session_scoped_identifiers(tmp_path):
    run = tmp_path / "run-1"
    run.mkdir()
    state = {}
    activate_run(state, run, "789")
    assert state["workflow-c-slurm-active-run"] == {
        "run_id": "run-1", "run_dir": str(run.resolve()), "job_id": "789",
    }


def test_gpu_submission_is_explicit_and_reuses_existing_submit_layer():
    source = inspect.getsource(show_he_geojson_preparation)
    assert 'key="workflow-c-run-on-gpu"' in source
    assert "if run_gpu:" in source
    assert "submit_prepared_workflow_c(" in source
    assert 'key="workflow-c-run-registration"' in source
    assert "Run Workflow C on GPU" in source


def test_workflow_a_b_do_not_gain_slurm_submission_behavior():
    for workflow in (show_point_registration_workflow, show_mask_to_mask_workflow):
        source = inspect.getsource(workflow)
        assert "submit_prepared_workflow_c" not in source
        assert "workflow-c-run-on-gpu" not in source
