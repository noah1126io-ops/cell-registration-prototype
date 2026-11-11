"""Analysis-only validation tests for the completed audit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import audit_lib as lib


def summary() -> dict:
    return json.loads((lib.OUT / "diagnostic_summary.json").read_text(encoding="utf-8"))


def test_original_subset_counts_and_stable_ids():
    inputs = lib.load_inputs()
    assert inputs.original_fixed.shape == (1979, 2)
    assert inputs.original_moving.shape == (1842, 2)
    assert inputs.fixed.shape == (1830, 2)
    assert inputs.moving.shape == (1754, 2)
    assert len(np.unique(inputs.fixed_original_index)) == 1830
    assert len(np.unique(inputs.moving_original_index)) == 1754


def test_affine_direction_reconstruction_and_inverse():
    inputs = lib.load_inputs()
    audit = lib.affine_audit(inputs)
    assert audit["transform_direction"].startswith("normalized HE pixel xy -> fluorescence")
    assert audit["saved_affine_reconstruction_max_abs_error_um"] <= 1e-12
    assert audit["inverse_roundtrip_max_abs_error_px"] <= 1e-10
    assert audit["determinant_after_explicit_flip"] > 0
    assert audit["effective_determinant_including_flip"] < 0
    assert audit["reestimation_exactly_matches_saved_matrix"]
    assert audit["reestimation_exactly_matches_saved_translation"]
    assert audit["reestimated_flip_matches_saved"]


def test_coordinate_units_and_unknown_acquisition_metadata_are_explicit():
    inputs = lib.load_inputs()
    assert inputs.metadata["output_pixel_size_um"] == 1.0
    assert inputs.metadata["output_origin"] == "lower-left"
    audit = json.loads((lib.OUT / "affine_transform_audit.json").read_text(encoding="utf-8"))
    assert audit["raw_he_pixel_spacing_metadata"] == "UNKNOWN"
    assert audit["raw_he_crop_offset_metadata"] == "UNKNOWN"
    assert audit["raw_he_downsampling_metadata"] == "UNKNOWN"


def test_roi_filter_reconstruction_is_exact():
    inputs = lib.load_inputs()
    fixed, moving, expected_fixed, expected_moving, _, _ = lib.reconstructed_subset_masks(inputs)
    np.testing.assert_array_equal(fixed, expected_fixed)
    np.testing.assert_array_equal(moving, expected_moving)
    roi = json.loads((lib.OUT / "roi_overlap_metrics.json").read_text(encoding="utf-8"))
    assert roi["saved_fixed_subset_exactly_reproduced"]
    assert roi["saved_moving_subset_exactly_reproduced"]


def test_candidate_index_integrity_and_frozen_hash():
    with np.load(lib.OUT / "audit_cache.npz", allow_pickle=False) as cache:
        indices = cache["baseline_candidate_indices"]
    assert indices.shape == (1754, 30)
    assert indices.min() >= 0 and indices.max() < 1830
    assert all(len(np.unique(row)) == 30 for row in indices)
    digest = __import__("hashlib").sha256(indices.astype(np.int64).tobytes()).hexdigest()
    assert digest == "778f89d8bd48f8573ab8c2aecf565e7f7761d2c830bc19f156e98bdea1f8f8df"


def test_ncc_boundary_conditions_and_known_baseline():
    cells = pd.read_csv(lib.OUT / "regional_ncc_per_nucleus.csv")
    baseline = cells[cells["search_radius_um"] == 18.0]
    assert len(baseline) == 1754
    assert int(baseline["hits_boundary"].sum()) == 1025
    assert np.all(np.abs(baseline[["offset_x_um", "offset_y_um"]].to_numpy()) <= 18.0 + 1e-12)
    np.testing.assert_allclose(
        np.mod(baseline[["offset_x_um", "offset_y_um"]].to_numpy(), 3.0), 0.0, atol=1e-12
    )
    anchors = pd.read_csv(lib.OUT / "regional_ncc_anchor_diagnostics.csv")
    anchor18 = anchors[anchors["search_radius_um"] == 18.0]
    assert len(anchor18) == 331
    assert int(anchor18["hits_boundary"].sum()) == 193


def test_baseline_reproduction_and_graph_equivalence():
    checks = summary()["baseline_reproduction"]
    assert checks["regional_offset_x_max_abs_error_um"] <= 1e-12
    assert checks["regional_offset_y_max_abs_error_um"] <= 1e-12
    assert checks["regional_ncc_max_abs_error"] <= 1e-12
    assert checks["regional_margin_max_abs_error"] <= 1e-12
    assert checks["fast_graph_support_max_abs_error"] <= 1e-12
    assert checks["graph_cost_top1_exact"]


def test_deterministic_regional_matching_on_fixed_synthetic_case():
    parameters = dict(lib.DIFFICULTIES["Easy"])
    sample = lib.generate_correspondence_sample(
        seed=9917, difficulty="Easy", n_latent=80, size=72, spacing=2.0, **parameters
    )
    first = lib.regional_match_vectorized(sample.fixed_points, sample.moving_points, 18.0)
    second = lib.regional_match_vectorized(sample.fixed_points, sample.moving_points, 18.0)
    np.testing.assert_array_equal(first.best_offsets, second.best_offsets)
    np.testing.assert_array_equal(first.best_ncc, second.best_ncc)
    np.testing.assert_array_equal(first.cell_anchor_index, second.cell_anchor_index)


def test_synthetic_affine_positive_controls_and_persistent_ids():
    table = pd.read_csv(lib.OUT / "synthetic_affine_sensitivity.csv")
    assert table["persistent_ids_validated"].all()
    means = table.groupby("condition").mean(numeric_only=True)
    correct = means.loc["correct_affine"]
    large = means.loc["large_translation_error"]
    assert large["true_match_spatial_rank_median"] > correct["true_match_spatial_rank_median"]
    assert large["ncc_boundary_hit_fraction"] > correct["ncc_boundary_hit_fraction"]
    assert large["graph_top1_accuracy_common"] < correct["graph_top1_accuracy_common"]
    assert large["true_match_candidate_recall_at_30"] < correct["true_match_candidate_recall_at_30"]


def test_required_outputs_are_finite_and_present():
    csv_names = [
        "regional_ncc_window_metrics.csv", "candidate_source_audit.csv",
        "candidate_k_sensitivity.csv", "spatial_region_metrics.csv",
        "synthetic_affine_sensitivity.csv",
    ]
    for name in csv_names:
        frame = pd.read_csv(lib.OUT / name)
        numeric = frame.select_dtypes(include=[np.number])
        assert len(frame) > 0 and np.isfinite(numeric.to_numpy()).all(), name
    required = [
        "REPORT.md", "coordinate_system_audit.md", "affine_transform_audit.json",
        "roi_overlap_metrics.json", "diagnostic_summary.json",
        "raw_vs_affine_overlay.png", "coordinate_transform_diagram.png", "roi_overlap.png",
        "affine_residual_vectors.png", "regional_ncc_boundary_map.png",
        "regional_ncc_window_comparison.png", "candidate_distance_distributions.png",
        "candidate_source_comparison.png", "graph_support_spatial_map.png",
    ]
    for name in required:
        assert (lib.OUT / name).stat().st_size > 100, name


def test_production_sources_unchanged_since_audit():
    record = summary()
    assert record["production_changes"] is False
    assert record["workflow_integration"] is False
    for relative, digest in record["source_hashes_after"].items():
        assert lib.file_sha256(lib.ROOT / relative) == digest

