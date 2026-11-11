from __future__ import annotations

import json

import numpy as np
import pandas as pd

import experiment_lib as lib


def test_required_outputs_exist():
    required = {
        "REPORT.md",
        "method_comparison.csv",
        "per_case_metrics.csv",
        "precision_coverage.csv",
        "calibration_metrics.csv",
        "real_like_synthetic_summary.csv",
        "negative_control_metrics.csv",
        "real_exploratory_metrics.csv",
        "diagnostic_summary.json",
        "precision_coverage.png",
        "matched_f1_comparison.png",
        "graph_discrimination.png",
        "unmatched_detection.png",
        "mass_vs_accuracy.png",
        "real_like_domain_gap.png",
        "real_candidate_examples.png",
    }
    assert not [name for name in required if not (lib.OUT / name).is_file()]


def test_real_like_generator_is_deterministic_and_preserves_ids():
    first = lib.generate_real_like_sample(seed=4101, strength="Weak")
    second = lib.generate_real_like_sample(seed=4101, strength="Weak")
    np.testing.assert_array_equal(first.fixed_points, second.fixed_points)
    np.testing.assert_array_equal(first.moving_points, second.moving_points)
    np.testing.assert_array_equal(first.true_fixed_index, second.true_fixed_index)
    common = first.true_fixed_index >= 0
    assert np.array_equal(
        first.moving_ids[common], first.fixed_ids[first.true_fixed_index[common]]
    )


def test_split_groups_are_disjoint_and_balanced():
    manifest = pd.read_csv(lib.OUT / "split_manifest.csv")
    assert manifest["template_group"].is_unique
    assert manifest.groupby("split")["seed"].nunique().to_dict() == {
        "Test": 5,
        "Train": 5,
        "Validation": 5,
    }
    assert manifest.groupby("split")["strength"].nunique().min() == 3


def test_candidate_indices_are_fixed_and_valid_across_methods():
    table = pd.read_csv(lib.OUT / "per_nucleus_results.csv")
    assert (table["candidate_count"] == 30).all()
    assert ((table["top1_fixed_index"] >= 0) & (table["top1_fixed_index"] < table["fixed_n"])).all()
    hashes = table.groupby("scenario_id")["candidate_hash"].nunique()
    assert (hashes == 1).all()
    comparison = pd.read_csv(lib.OUT / "method_comparison.csv")
    assert (comparison["true_candidate_recall"] == 1.0).all()


def test_calibration_has_no_test_or_real_leakage():
    summary = json.loads((lib.OUT / "diagnostic_summary.json").read_text())
    policy = summary["split_policy"]
    assert not policy["test_used_for_training_or_threshold_selection"]
    assert not policy["real_used_for_training_or_threshold_selection"]
    assert set(policy["train_seeds"]).isdisjoint(policy["validation_seeds"])
    assert set(policy["train_seeds"]).isdisjoint(policy["test_seeds"])
    assert set(policy["validation_seeds"]).isdisjoint(policy["test_seeds"])
    assert summary["b2_match_model"]["training_rows"] > 0
    assert summary["b2_match_model"]["converged"]


def test_negative_controls_are_ground_truth_unmatched_and_evaluated():
    table = pd.read_csv(lib.OUT / "negative_control_per_nucleus.csv")
    assert set(table["strength"].unique()) == {
        "identity_permutation",
        "unrelated_pattern",
        "translated_disjoint",
    }
    assert not table["is_common_cell"].astype(bool).any()
    assert (table["true_fixed_index"] == -1).all()
    metrics = pd.read_csv(lib.OUT / "negative_control_metrics.csv")
    assert set(lib.MAIN_METHODS) <= set(metrics["method"])


def test_statuses_are_exclusive_and_unmatched_handling_is_finite():
    table = pd.read_csv(lib.OUT / "per_nucleus_results.csv")
    assert set(table["status"].unique()) <= {"matched", "ambiguous", "unmatched"}
    core = [
        "absolute_min_cost",
        "absolute_cost_margin",
        "top1_graph_support",
        "normalized_top1_mass",
        "top1_top2_mass_margin",
        "normalized_row_entropy",
        "row_mass_ratio",
        "calibrated_confidence",
    ]
    assert np.isfinite(table[core].to_numpy(float)).all()
    comparison = pd.read_csv(lib.OUT / "method_comparison.csv")
    assert (
        comparison["matched_n"]
        + comparison["ambiguous_n"]
        + comparison["predicted_unmatched_n"]
        == comparison["n_rows"]
    ).all()


def test_existing_candidate_fixed_baselines_are_reproduced():
    table = pd.read_csv(lib.OUT / "baseline_reproduction.csv").set_index("method")
    assert abs(table.loc[lib.METHOD_B0, "uot_top1_accuracy"] - 0.8523434259643301) < 1e-12
    assert abs(table.loc[lib.METHOD_B0, "matched_f1"] - 0.8071157305616630) < 1e-12
    assert abs(table.loc[lib.METHOD_B3, "uot_top1_accuracy"] - 0.8361675653255910) < 1e-12
    assert abs(table.loc[lib.METHOD_B3, "matched_f1"] - 0.7990053874844593) < 1e-12


def test_source_files_remain_unchanged():
    summary = json.loads((lib.OUT / "diagnostic_summary.json").read_text())
    for relative, expected in summary["source_sha256"].items():
        assert lib.sha256_file(lib.ROOT / relative) == expected


def test_real_confidence_is_marked_unlabeled_and_counts_sum():
    summary = json.loads((lib.OUT / "diagnostic_summary.json").read_text())
    assert not summary["real_has_ground_truth"]
    table = pd.read_csv(lib.OUT / "real_exploratory_metrics.csv")
    assert not table["accuracy_available"].astype(bool).any()
    assert ((table["matched_n"] + table["ambiguous_n"] + table["unmatched_n"]) == 1754).all()

