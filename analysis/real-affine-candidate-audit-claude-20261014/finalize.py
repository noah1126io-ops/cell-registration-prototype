"""Assemble required deliverables (JSON/MD) from cache/. Reads only; writes only in this directory."""
from __future__ import annotations

import json
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

import audit_core as ac

C = ac.OUT / "cache"
L = lambda n: json.loads((C / n).read_text())  # noqa: E731
coord, corr, aff, he, cand, nccv, dom, spat = (L(n) for n in (
    "coordinate.json", "correlogram.json", "affine.json", "he_roi.json", "candidates.json",
    "ncc_validation.json", "domain.json", "spatial.json"))
holdout = pd.read_csv(C / "affine_holdout.csv")
pose = pd.read_csv(C / "pose_landscape.csv")
cap = pd.read_csv(C / "icp_capture_range.csv")
ncc = pd.read_csv(ac.OUT / "ncc_window_sensitivity.csv")
dm = pd.read_csv(C / "domain_method_metrics.csv")
dcase = pd.read_csv(C / "domain_case_summary.csv")

affine_audit = {
    "transform_direction": "HE raw pixel xy -> (flip_x: 901-x) -> p @ A.T + t -> fluorescence GeoJSON world xy",
    "saved": {**coord["affine_saved"], "post_flip": coord["post_flip_decomposition"],
              "effective_raw_to_world": coord["effective_raw_to_world"]},
    "implementation_checks": {k: coord[k] for k in ("reconstruction_max_abs_error_um",
                                                     "effective_matrix_reconstruction_max_abs_error_um",
                                                     "inverse_roundtrip_max_abs_error_px")},
    "alignment_proxies_saved_vs_corrected_vs_null": {
        "saved": aff["nn_saved_subset"], "diagnostic_refit": aff["nn_corrected_subset"],
        "nulls_saved": aff["nn_null_subset"], "poisson_within_r": aff["poisson_expectation_within_r"],
        "radial_cross_g_saved": corr["radial_g_saved_affine"], "radial_cross_g_refit": aff["radial_g_corrected"]},
    "local_correlogram": {k: corr[k] for k in ("blocks_total", "blocks_peak_ratio_ge_5", "block_peak_ratio_median",
                                               "block_peak_ratio_min", "null_peak_ratio_median", "null_peak_ratio_max",
                                               "block_offset_median_um", "block_offset_norm_range_um",
                                               "block_fraction_with_partner_at_peak", "linear_offset_model")},
    "diagnostic_refit": {k: aff[k] for k in ("method", "seed_pairs", "corrected_matrix_post_flip", "corrected_translation",
                                             "corrected_decomposition", "relative_world_correction",
                                             "correction_displacement_um", "icp_objective_trimmed_q0.7")},
    "heldout_validation": holdout[["split", "train_n", "test_n", "seed_pairs", "heldout_corrected_within_3um",
                                   "heldout_saved_within_3um", "heldout_null_within_3um",
                                   "heldout_corrected_moving_to_fixed_nn_median_um",
                                   "heldout_saved_moving_to_fixed_nn_median_um"]].to_dict("records"),
    "pose_landscape_best_per_flip": pose.sort_values("affine_trimmed_error").groupby(["flip_x", "flip_y"]).head(1)[
        ["flip_x", "flip_y", "init_rotation_deg", "affine_trimmed_error", "effective_det", "within_3um"]].to_dict("records"),
    "icp_capture_range": {"starts": int(len(cap)), "reached_shared_nucleus_basin": int((cap.within_3um > 0.5).sum()),
                          "basin_init_offsets_dx_range": [int(cap[cap.within_3um > 0.5].init_dx.min()), int(cap[cap.within_3um > 0.5].init_dx.max())],
                          "basin_init_offsets_dy_range": [int(cap[cap.within_3um > 0.5].init_dy.min()), int(cap[cap.within_3um > 0.5].init_dy.max())]},
    "he_image_independent_check": {k: he[k] for k in ("strict", "loose", "translation_scan_min")},
    "true_TRE": None,
    "true_TRE_note": "No independent landmarks; TRE not computed. Held-out and image-based checks are proxies.",
}
ac.save_json(ac.OUT / "affine_audit.json", affine_audit)

roi = {
    "subset_filter": coord["subset_filter_reproduced"] | {"edge_margin_px": 10.0, "near_he_um": 43.0},
    "tissue_mask_semantics": coord["tissue_mask_semantics"],
    "saved": he["roi_saved"], "diagnostic_refit": he["roi_corrected"],
    "note": "HE image footprint under the re-fit is displaced ~50 um in -y relative to the fluorescence field; "
            "the fluorescence acquisition polygon is unknown, so true common area is not certified from point hulls.",
}
ac.save_json(ac.OUT / "roi_overlap_metrics.json", roi)

git = lambda *a: subprocess.run(["git", *a], cwd=ac.ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
summary = {
    "purpose": "independent Claude audit of Real affine / coordinate / candidate discovery",
    "dataset": "serial_section_pair_20260904", "date": "2026-10-14",
    "branch": git("branch", "--show-current"), "git_head": git("rev-parse", "HEAD"),
    "python": sys.version, "platform": platform.platform(),
    "audit_status": {
        "coordinate_system": "PASS (internal path) / provenance metadata UNKNOWN",
        "affine_direction": "PASS",
        "affine_alignment": "FAIL",
        "roi_overlap": "FAIL (crop offset; mask is image footprint)",
        "regional_ncc": "FAIL as configured (window < residual); method itself works after re-fit",
        "candidate_discovery": "PASS at K=30 (proxy partner in final set 94.4%); FAIL at K=10",
        "candidate_pruning_ranking": "FAIL (proxy partner top-1 0.08%)",
    },
    "cause_classification": ["B. Affine misalignment (primary)", "C. ROI/overlap (crop offset defeating centroid initialisation)",
                             "D. Regional NCC limitation (secondary consequence of B)",
                             "E. Candidate ranking (consequence of B; discovery itself largely adequate at K=30)"],
    "not_supported": ["A. coordinate convention error", "F. biological/domain difference as main explanation of the Real gap"],
    "key_numbers": {
        "within_3um_saved_refit_null": [aff["nn_saved_subset"]["within_3um"], aff["nn_corrected_subset"]["within_3um"],
                                        aff["nn_null_subset"]["mirror_x"]["within_3um"]],
        "icp_objective_saved_refit": [aff["icp_objective_trimmed_q0.7"]["saved"], aff["icp_objective_trimmed_q0.7"]["corrected"]],
        "correction_median_um": [aff["correction_displacement_um"]["median_dx"], aff["correction_displacement_um"]["median_dy"]],
        "correction_rotation_deg": aff["relative_world_correction"]["polar_rotation_deg"],
        "fixed_in_he_white_saved_refit": [he["strict"]["fixed_saved"], he["strict"]["fixed_corrected"], he["strict"]["white_area_fraction"]],
        "ncc_window": ncc[["condition", "search_radius_um", "nucleus_boundary_hit_rate", "anchor_boundary_hit_rate",
                           "agree_with_correlogram_within_6um_rate"]].to_dict("records"),
        "domain_methods": dm.to_dict("records"),
        "domain_cases": dcase.to_dict("records"),
        "candidates": cand,
    },
    "reproducibility": {"ncc_vectorized_vs_frozen": nccv, "b2_refit_reproduces_stored": dom["b2_status_identical"],
                        "candidate_hash_reproduced": cand["stored_candidate_hash_reproduced"],
                        "graph_fast_vs_frozen_max_abs_diff": cand["graph_fast_vs_frozen_max_abs_diff"],
                        "all_uot_solvers_converged": bool(dcase["solver_all_converged"].dropna().astype(bool).all()),
                        "all_costs_finite": bool(dcase["finite_costs"].dropna().astype(bool).all())},
    "spatial_correlations_naive": spat,
    "codex_audit_compared": str(ac.CODEX_AUDIT),
    "changes_outside_this_directory": "none (sha256 of tracked files and prior analysis files verified before/after)",
}
ac.save_json(ac.OUT / "diagnostic_summary.json", summary)
print("finalized")
