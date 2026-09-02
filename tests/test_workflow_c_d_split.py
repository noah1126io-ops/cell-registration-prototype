import inspect

import numpy as np

from app import (
    WORKFLOW_C_CURRENT_RESULT_KEY,
    WORKFLOW_D_ARTIFACT_SOURCE_KEY,
    WORKFLOW_D_LABEL,
    WORKFLOW_SELECTOR_KEY,
    _continue_to_workflow_d,
    _store_current_workflow_c_result,
    _workflow_d_artifact_payload,
    main,
    show_he_geojson_preparation,
    show_mask_to_mask_workflow,
    show_point_registration_workflow,
    show_raster_deformation_workflow,
)


def test_main_exposes_separate_workflow_c_and_d_entries():
    source = inspect.getsource(main)

    assert '"Workflow C: Point Registration"' in source
    assert "WORKFLOW_D_LABEL" in source
    assert WORKFLOW_D_LABEL == "Workflow D: Raster Deformation"
    assert "show_he_geojson_preparation()" in source
    assert "show_raster_deformation_workflow()" in source


def test_workflow_c_exports_registration_artifact_without_running_final_raster_block():
    source = inspect.getsource(show_he_geojson_preparation)

    assert "workflow_c_raster_outputs_enabled = False" in source
    assert "if he_image is not None and workflow_c_raster_outputs_enabled" in source
    assert "build_workflow_c_result_artifact" in source
    assert 'artifacts["workflow_c_result.zip"]' in source
    assert "_store_current_workflow_c_result" in source
    assert "Continue to Workflow D" in source


def test_workflow_d_consumes_artifact_and_does_not_run_registration():
    source = inspect.getsource(show_raster_deformation_workflow)

    assert "load_workflow_c_result_artifact" in source
    assert "run_workflow_d_raster_deformation" in source
    assert "estimate_affine_with_y_flip" not in source
    assert "tissue_aware_density_flow_registration" not in source
    assert "joint_density_tissue_structure_registration" not in source
    assert "Loaded from current Workflow C result" in source
    assert "workflow-d-result-artifact" in source


def test_workflows_a_and_b_do_not_reference_workflow_d():
    for workflow in (show_point_registration_workflow, show_mask_to_mask_workflow):
        source = inspect.getsource(workflow)
        assert "Workflow D" not in source
        assert "run_workflow_d_raster_deformation" not in source


def test_current_workflow_c_result_is_passed_to_workflow_d_through_session_state():
    state = {}
    image = np.arange(18, dtype=np.uint8).reshape(3, 3, 2)

    _store_current_workflow_c_result(
        state,
        artifact_bytes=b"workflow-c-artifact",
        run_id="run-123",
        registration_status="applied",
        method="tissue-aware density flow",
        preset="balanced",
        he_image=image,
        registration_metadata={"output_origin": "upper-left"},
    )
    payload, context = _workflow_d_artifact_payload(state, "Current Workflow C result", None)

    assert payload == b"workflow-c-artifact"
    assert context["run_id"] == "run-123"
    assert context["registration_status"] == "applied"
    assert context["method"] == "tissue-aware density flow"
    assert context["preset"] == "balanced"
    assert context["registration_metadata"] == {"output_origin": "upper-left"}
    np.testing.assert_array_equal(context["he_image"], image)
    assert context["he_image"] is not image


def test_uploaded_workflow_c_artifact_remains_available():
    class UploadedArtifact:
        def getvalue(self):
            return b"uploaded-workflow-c-artifact"

    payload, context = _workflow_d_artifact_payload(
        {WORKFLOW_C_CURRENT_RESULT_KEY: {"artifact_bytes": b"current"}},
        "Uploaded Workflow C artifact",
        UploadedArtifact(),
    )

    assert payload == b"uploaded-workflow-c-artifact"
    assert context is None


def test_continue_navigation_only_selects_workflow_d_without_registration_recalculation():
    state = {
        WORKFLOW_C_CURRENT_RESULT_KEY: {
            "artifact_bytes": b"existing-result",
            "run_id": "run-123",
        }
    }

    _continue_to_workflow_d(state)

    assert state[WORKFLOW_SELECTOR_KEY] == WORKFLOW_D_LABEL
    assert state[WORKFLOW_D_ARTIFACT_SOURCE_KEY] == "Current Workflow C result"
    assert state[WORKFLOW_C_CURRENT_RESULT_KEY]["artifact_bytes"] == b"existing-result"
    source = inspect.getsource(_continue_to_workflow_d)
    assert "estimate_affine_with_y_flip" not in source
    assert "tissue_aware_density_flow_registration" not in source
    assert "joint_density_tissue_structure_registration" not in source
