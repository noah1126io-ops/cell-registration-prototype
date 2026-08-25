import inspect

from app import (
    WORKFLOW_C_FINE_METHODS,
    WORKFLOW_C_METHOD_GROUPS,
    _workflow_c_joint_presets,
    show_he_geojson_preparation,
    show_mask_to_mask_workflow,
    show_point_registration_workflow,
)


def test_workflow_c_keeps_every_fine_alignment_method_available():
    assert set(WORKFLOW_C_FINE_METHODS) == {
        "joint density + tissue-structure flow",
        "tissue-aware density flow",
        "off",
        "cluster-anchor",
        "matched nuclei RBF",
        "local translation field",
        "center-snap",
    }
    assert WORKFLOW_C_METHOD_GROUPS["joint density + tissue-structure flow"] == (
        "Recommended / current research"
    )
    assert WORKFLOW_C_METHOD_GROUPS["tissue-aware density flow"] == "Baseline"


def test_joint_flow_preset_values_are_unchanged_and_custom_is_available():
    presets = _workflow_c_joint_presets()

    assert list(presets) == [
        "Joint Safe",
        "Joint Tissue-shape",
        "Joint Strong exploratory",
        "Custom",
    ]
    assert presets["Joint Safe"] == {
        "a_scales": "32, 16, 8",
        "b_scales": "8, 4",
        "a_iter": 8,
        "b_iter": 10,
        "a_lr": 0.10,
        "b_lr": 0.05,
        "a_smooth": 6.0,
        "b_smooth": 3.0,
        "density": 1.0,
        "support": 0.70,
        "structure": 0.35,
        "explore_max": 35.0,
        "explore_p95": 25.0,
    }
    assert presets["Joint Tissue-shape"]["a_iter"] == 12
    assert presets["Joint Tissue-shape"]["a_lr"] == 0.14
    assert presets["Joint Strong exploratory"]["a_iter"] == 16
    assert presets["Custom"]["a_lr"] == 0.10


def test_workflow_c_ui_has_five_steps_run_action_and_six_result_tabs():
    source = inspect.getsource(show_he_geojson_preparation)

    for step in range(1, 6):
        assert f"STEP {step}" in source
    assert 'key="workflow-c-run-registration"' in source
    assert "overview_tab, alignment_tab, deformation_tab" in source
    assert "diagnostics_tab, export_tab = st.tabs" in source
    assert 'with st.expander("Advanced parameters - Joint two-stage optimizer"' in source


def test_workflow_c_ui_controls_do_not_leak_into_workflows_a_or_b():
    for workflow in (show_point_registration_workflow, show_mask_to_mask_workflow):
        source = inspect.getsource(workflow)
        assert "WORKFLOW_C_FINE_METHODS" not in source
        assert "workflow-c-run-registration" not in source
        assert "Joint Flow preset" not in source


def test_rejected_candidate_ui_is_qc_only_and_has_no_safety_override():
    source = inspect.getsource(show_he_geojson_preparation)

    assert 'key="workflow-c-show-rejected-candidate"' in source
    assert '["Summary", "Compare", "Full diagnostics"]' in source
    assert "NOT APPLIED - REJECTED RESEARCH QC CANDIDATE" in source
    assert "human_qc_review_record" in source
    assert "Apply anyway" not in source
    assert "Force apply" not in source
    assert "Ignore safety" not in source
