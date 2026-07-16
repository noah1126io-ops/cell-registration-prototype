from dataclasses import replace
import inspect

import numpy as np
import pytest

from app import _workflow_c_result_variants, show_he_geojson_preparation
from src.pointset_registration import FineWarpResult


def _fine_result(*, applied: bool) -> FineWarpResult:
    grid_x, grid_y = np.meshgrid(np.arange(2, dtype=float), np.arange(2, dtype=float))
    attempted_dx = np.full((2, 2), 2.0)
    attempted_dy = np.full((2, 2), -1.0)
    affine_points = np.array([[1.0, 2.0], [3.0, 4.0]])
    attempted_points = affine_points + np.array([2.0, -1.0])
    zeros = np.zeros((2, 2), dtype=float)
    return FineWarpResult(
        transformed_points=attempted_points if applied else affine_points,
        grid_x=grid_x,
        grid_y=grid_y,
        displacement_x=attempted_dx if applied else zeros,
        displacement_y=attempted_dy if applied else zeros,
        bounds=(0.0, 0.0, 4.0, 4.0),
        grid_spacing=1.0,
        jacobian_min=0.8,
        jacobian_max=1.2,
        max_displacement=3.0,
        n_candidate_pairs=4,
        n_pairs=3,
        n_filtered_pairs=1,
        median_pair_distance_before=2.0,
        median_pair_distance_after=1.0 if applied else 2.0,
        success=applied,
        message="applied" if applied else "rejected",
        attempted_transformed_points=attempted_points,
        attempted_displacement_x=attempted_dx,
        attempted_displacement_y=attempted_dy,
        rejection_reason=None if applied else "jacobian_min_below_limit",
        applied=applied,
    )


def test_workflow_c_applied_result_uses_fine_candidate():
    affine_points = np.array([[1.0, 2.0], [3.0, 4.0]])
    variants = _workflow_c_result_variants(affine_points, _fine_result(applied=True))

    assert variants["fine_applied"] is True
    assert variants["applied_result_label"] == "Affine + fine warp"
    np.testing.assert_allclose(variants["final_points"], affine_points + np.array([2.0, -1.0]))
    np.testing.assert_allclose(variants["final_displacement_x"], 2.0)


@pytest.mark.parametrize(
    "rejection_reason",
    ["jacobian_min_below_limit", "valid_region_median_distance_worsened"],
)
def test_workflow_c_rejected_result_preserves_attempted_but_applies_affine_only(rejection_reason):
    affine_points = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = replace(_fine_result(applied=False), rejection_reason=rejection_reason)
    variants = _workflow_c_result_variants(affine_points, result)

    assert variants["fine_applied"] is False
    assert variants["applied_result_label"] == "Affine only"
    np.testing.assert_allclose(variants["attempted_points"], affine_points + np.array([2.0, -1.0]))
    np.testing.assert_allclose(variants["attempted_displacement_x"], 2.0)
    np.testing.assert_allclose(variants["final_points"], affine_points)
    np.testing.assert_allclose(variants["final_displacement_x"], 0.0)
    np.testing.assert_allclose(variants["final_displacement_y"], 0.0)


def test_workflow_c_disabled_or_legacy_result_has_safe_attempted_fallback():
    affine_points = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = _fine_result(applied=False)
    result = FineWarpResult(
        **{
            **result.__dict__,
            "attempted_transformed_points": None,
            "attempted_displacement_x": None,
            "attempted_displacement_y": None,
        }
    )
    variants = _workflow_c_result_variants(affine_points, result)

    np.testing.assert_allclose(variants["attempted_points"], affine_points)
    np.testing.assert_allclose(variants["attempted_displacement_x"], 0.0)
    np.testing.assert_allclose(variants["final_points"], affine_points)


def test_workflow_c_status_panel_precedes_unconditional_result_tabs():
    source = inspect.getsource(show_he_geojson_preparation)

    status_position = source.index('st.subheader(tr("実行状態", "Run status"))')
    tabs_position = source.index("= st.tabs(")
    assert status_position < tabs_position
    assert 'if fine_applied:\n        overview_tab' not in source
    assert 'if not fine_applied:\n        return' not in source
    assert 'if not fine_result.success:\n        return' not in source
    assert "st.stop()" not in source


def test_workflow_c_cluster_anchor_status_uses_separate_counts():
    source = inspect.getsource(show_he_geojson_preparation)

    assert "Accepted cluster anchors" in source
    assert "Rejected cluster anchors" in source
    assert "Boundary pin anchors" in source
    assert "Total candidate cluster anchors" in source
    assert "Fine snap pairs" not in source
