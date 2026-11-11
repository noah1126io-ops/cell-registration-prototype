from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import diagnose


def test_all_moving_ids_are_preserved_across_stage_outputs(frozen_real):
    _, moving, state, trace = frozen_real
    assert len(moving) == 1754
    assert len(trace) == 1754
    assert trace["moving_nucleus_id"].is_unique
    assert np.array_equal(trace["moving_index"], np.arange(1754))
    stages = diagnose.stage_diagnostics(state, trace)
    assert (stages["input_moving_nuclei"] == 1754).all()
    assert (stages["valid_moving_nuclei"] == 1754).all()


def test_final_classification_is_exclusive_and_matches_source_counts(frozen_real):
    _, _, _, trace = frozen_real
    allowed = {"matched", "ambiguous", "unmatched"}
    assert set(trace["status"]) <= allowed
    assert trace["status"].value_counts().to_dict() == {
        "unmatched": 1295,
        "ambiguous": 434,
        "matched": 25,
    }


def test_confidence_recomputation_and_saved_result_match(frozen_real):
    _, _, _, trace = frozen_real
    assert trace["confidence_recompute_absolute_error"].max() < 1e-12
    verification = diagnose.verify_against_saved(trace)
    assert all(verification["exact_columns"].values())
    assert max(verification["numeric_max_absolute_errors"].values()) < 1e-12


def test_synthetic_ground_truth_ids_remain_aligned():
    table = pd.read_csv(diagnose.SOURCE / "correspondence_results.csv")
    table = table.loc[
        (table["method"] == "M5 Full contextual graph-UOT")
        & table["scenario_id"].str.startswith("primary_")
    ]
    common = table["is_common_cell"].astype(bool)
    assert (table.loc[common, "moving_cell_id"] == table.loc[common, "true_fixed_cell_id"]).all()
    assert (table.loc[common, "true_fixed_index"] >= 0).all()
    assert (table.loc[~common, "true_fixed_index"] < 0).all()


def test_candidate_indices_and_fixed_coordinates_are_consistent(frozen_real):
    fixed, _, _, trace = frozen_real
    for rank in (1, 2, 3):
        index = trace[f"rank{rank}_fixed_index"].to_numpy(dtype=int)
        assert np.all((index >= 0) & (index < len(fixed)))
        assert np.allclose(trace[f"rank{rank}_fixed_x_um"], fixed[index, 0])
        assert np.allclose(trace[f"rank{rank}_fixed_y_um"], fixed[index, 1])


def test_diagnostic_run_did_not_modify_source_artifacts():
    summary = json.loads((diagnose.OUT / "diagnostic_summary.json").read_text())
    for filename, expected in summary["source_sha256_after_diagnostics"].items():
        actual = hashlib.sha256((diagnose.SOURCE / filename).read_bytes()).hexdigest()
        assert actual == expected


def test_nan_inf_detection_is_explicit():
    diagnose.validate_finite("finite", np.asarray([0.0, 1.0]))
    with pytest.raises(ValueError, match="NaN/Inf"):
        diagnose.validate_finite("bad", np.asarray([0.0, np.nan, np.inf]))


def test_frozen_reexecution_is_deterministic(frozen_real):
    _, _, _, trace = frozen_real
    saved = pd.read_csv(diagnose.OUT / "per_nucleus_trace.csv")
    columns = [
        "moving_nucleus_id",
        "top1_fixed_index",
        "top2_fixed_index",
        "row_mass_ratio",
        "normalized_top1_mass",
        "top1_top2_mass_margin",
        "confidence_saved_algorithm",
        "status",
    ]
    pd.testing.assert_frame_equal(
        trace[columns].reset_index(drop=True),
        saved[columns].reset_index(drop=True),
        check_exact=False,
        atol=1e-12,
        rtol=0,
    )
