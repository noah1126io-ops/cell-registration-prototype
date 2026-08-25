import inspect

import numpy as np

from app import show_mask_to_mask_workflow, show_point_registration_workflow
from src.density_flow import tissue_aware_density_flow_registration
from src.registration_evaluation import (
    deformation_validity_metrics,
    displacement_endpoint_error,
    generate_method_scorecard,
    human_qc_review_record,
    landmark_tre_metrics,
    local_region_evaluation,
    raster_fidelity_summary,
    rejected_candidate_statuses,
    rejection_gate_analysis,
    transform_validation_landmarks,
)


def test_landmark_perfect_transform_and_known_offset():
    fixed = np.array([[0.0, 0.0], [10.0, 10.0]])
    before = fixed + np.array([3.0, 4.0])
    table, summary = landmark_tre_metrics(fixed, before, fixed)

    assert np.allclose(table["tre_before_um"], 5.0)
    assert np.allclose(table["tre_after_um"], 0.0)
    assert summary["tre_median_before_um"] == 5.0
    assert summary["tre_median_after_um"] == 0.0


def test_rejected_fine_landmark_result_equals_affine():
    moving = np.array([[2.0, 4.0], [8.0, 6.0]])
    zero = np.zeros((8, 8), dtype=float)
    transformed = transform_validation_landmarks(
        moving,
        affine_matrix=np.eye(2),
        translation=np.array([1.0, -2.0]),
        attempted_displacement_x=np.ones((8, 8)),
        attempted_displacement_y=np.zeros((8, 8)),
        applied_displacement_x=zero,
        applied_displacement_y=zero,
        field_bounds=(0.0, 0.0, 7.0, 7.0),
        field_spacing=1.0,
    )

    assert not np.allclose(transformed["attempted"], transformed["affine"])
    assert np.allclose(transformed["applied"], transformed["affine"])


def test_landmarks_are_not_optimizer_inputs_and_evaluation_is_workflow_c_only():
    signature = inspect.signature(tissue_aware_density_flow_registration)
    assert not any("landmark" in name for name in signature.parameters)
    assert "registration_evaluation" not in inspect.getsource(show_point_registration_workflow)
    assert "registration_evaluation" not in inspect.getsource(show_mask_to_mask_workflow)


def test_displacement_endpoint_error_and_xy_reversal():
    true_x = np.full((3, 4), 3.0)
    true_y = np.full((3, 4), 4.0)
    _, exact = displacement_endpoint_error(true_x, true_y, true_x, true_y)
    _, offset = displacement_endpoint_error(true_x + 3.0, true_y + 4.0, true_x, true_y)
    _, reversed_xy = displacement_endpoint_error(true_y, true_x, true_x, true_y)

    assert exact["epe_max_um"] == 0.0
    assert offset["epe_median_um"] == 5.0
    assert reversed_xy["epe_median_um"] > 0.0


def test_local_evaluation_detects_worsened_region_and_skips_sparse_region():
    first = np.array([[10 + x, 10 + y] for x in range(3) for y in range(2)], dtype=float)
    second = np.array([[70 + x, 70 + y] for x in range(3) for y in range(2)], dtype=float)
    sparse = np.array([[115.0, 115.0]])
    fixed = np.vstack([first, second, sparse])
    affine = fixed.copy()
    applied = fixed.copy()
    applied[len(first):len(first) + len(second)] += np.array([3.0, 0.0])
    zero = np.zeros((13, 13), dtype=float)

    table, summary = local_region_evaluation(
        fixed,
        affine,
        applied,
        applied,
        zero,
        zero,
        zero,
        zero,
        bounds=(0.0, 0.0, 120.0, 120.0),
        spacing=10.0,
        block_size_um=50.0,
        min_points=5,
    )

    assert len(table) == 2
    assert summary["fraction_worsened"] > 0
    assert table["fixed_point_count"].min() >= 5


def test_scorecard_never_calls_internal_only_improvement_validated():
    affine = {"symmetric_nn_median_um": 2.0}
    applied = {"symmetric_nn_median_um": 1.0}
    deformation = deformation_validity_metrics(np.zeros((4, 4)), np.zeros((4, 4)), spacing=1.0)
    raster = raster_fidelity_summary({"converged": True})
    _, summary = generate_method_scorecard(
        landmark_summary=None,
        affine_internal_metrics=affine,
        applied_internal_metrics=applied,
        local_summary={"fraction_worsened": 0.0},
        applied_deformation_metrics=deformation,
        raster_fidelity=raster,
        fine_applied=True,
    )

    assert summary["overall_status"] == "INTERNAL IMPROVEMENT ONLY"


def test_rejected_unsafe_deformation_cannot_receive_pass():
    safe = deformation_validity_metrics(np.zeros((4, 4)), np.zeros((4, 4)), spacing=1.0)
    scorecard, summary = generate_method_scorecard(
        landmark_summary=None,
        affine_internal_metrics={"symmetric_nn_median_um": 2.0},
        applied_internal_metrics={"symmetric_nn_median_um": 2.0},
        local_summary={"fraction_worsened": 0.0},
        applied_deformation_metrics=safe,
        raster_fidelity={"inverse_solver_converged": True},
        fine_applied=False,
        fine_rejected=True,
    )

    safety_status = scorecard.loc[scorecard["domain"] == "Deformation safety", "status"].iloc[0]
    assert safety_status == "FAIL"
    assert summary["overall_status"] == "UNSAFE / REJECTED"


def test_rejection_gate_analysis_reports_all_gates_and_signed_margins():
    table = rejection_gate_analysis(
        {
            "Jacobian p05": {"observed": 0.772, "required_min": 0.8},
            "Jacobian max": {"observed": 1.2, "required_max": 3.0},
            "inverse raster convergence": {"observed": None, "required_min": 1.0},
        }
    ).set_index("check")

    assert table.loc["Jacobian p05", "status"] == "FAIL"
    assert np.isclose(table.loc["Jacobian p05", "margin_to_threshold"], -0.028)
    assert table.loc["Jacobian max", "status"] == "PASS"
    assert np.isclose(table.loc["Jacobian max", "margin_to_threshold"], 1.8)
    assert table.loc["inverse raster convergence", "status"] == "NOT AVAILABLE"


def test_rejected_candidate_separates_alignment_improvement_from_failed_safety():
    analysis = rejection_gate_analysis(
        {"Jacobian p05": {"observed": 0.772, "required_min": 0.8}}
    )
    statuses = rejected_candidate_statuses(10.0, 9.5, analysis)

    assert statuses["attempted_alignment_status"] == "IMPROVED"
    assert statuses["safety_status"] == "FAILED"
    assert statuses["final_status"] == "REJECTED"
    assert statuses["failed_gate_count"] == 1


def test_human_qc_review_is_annotation_only_and_defaults_to_not_reviewed():
    default_review = human_qc_review_record()
    reviewed = human_qc_review_record(
        "Probably better",
        note="Boundary alignment appears improved.",
        flags=["tissue boundaries look better aligned"],
    )

    assert default_review == {
        "review_status": "not_reviewed",
        "assessment": "Not reviewed",
        "note": "",
        "note_present": False,
        "flags": [],
        "annotation_only": True,
        "changes_registration_result": False,
    }
    assert reviewed["review_status"] == "reviewed"
    assert reviewed["note_present"] is True
    assert reviewed["changes_registration_result"] is False
