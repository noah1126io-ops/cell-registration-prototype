"""Analysis-only tests for the Claude audit (synthetic checks + saved-artifact integrity)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_core as ac  # noqa: E402

C = ac.OUT / "cache"


def test_displacement_peak_recovers_known_shift():
    rng = np.random.default_rng(0)
    fixed = rng.uniform(0, 500, (500, 2))
    moving = fixed[:350] - np.array([7.0, -41.0]) + rng.normal(0, 0.8, (350, 2))
    r = ac.displacement_peak(fixed, moving, search_um=60, step_um=1.0, density=500 / 500**2)
    assert abs(r["refined_dx"] - 7.0) < 1.0 and abs(r["refined_dy"] + 41.0) < 1.0
    assert r["peak_ratio"] > 10


def test_displacement_peak_null_is_low():
    rng = np.random.default_rng(1)
    fixed = rng.uniform(0, 500, (500, 2)); moving = rng.uniform(0, 500, (350, 2))
    r = ac.displacement_peak(fixed, moving, search_um=60, step_um=2.0, density=500 / 500**2)
    assert r["peak_ratio"] < 5


def test_graph_support_fast_matches_frozen_on_small_case():
    from graph_consistency import graph_support_scores
    from candidate_generation import CandidateSet
    rng = np.random.default_rng(2)
    fixed = rng.uniform(0, 200, (120, 2)); moving = fixed[:100] + rng.normal(0, 1, (100, 2))
    from scipy.spatial.distance import cdist
    idx = np.argsort(cdist(moving, fixed), axis=1)[:, :12]
    base = rng.uniform(0, 3, idx.shape)
    z = np.zeros(idx.shape)
    cs = CandidateSet(idx, z, z, z, base, z, z, z, z)
    assert np.max(np.abs(ac.graph_support_fast(fixed, moving, idx, base) - graph_support_scores(fixed, moving, cs))) < 1e-12


def test_decompose_affine_reflection_and_rotation():
    th = np.deg2rad(1.24)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) * 1.1
    out = ac.decompose_affine(R)
    assert abs(out["polar_rotation_deg"] - 1.24) < 1e-9 and not out["polar_factor_is_reflection"]
    assert ac.decompose_affine(R @ np.diag([1, -1]))["polar_factor_is_reflection"]


@pytest.mark.skipif(not (C / "coordinate.json").exists(), reason="audit not run")
def test_saved_transform_integrity():
    c = json.loads((C / "coordinate.json").read_text())
    assert c["reconstruction_max_abs_error_um"] == 0.0
    assert c["inverse_roundtrip_max_abs_error_px"] < 1e-9
    assert c["subset_filter_reproduced"]["fixed"] and c["subset_filter_reproduced"]["moving"]


@pytest.mark.skipif(not (C / "candidates.json").exists(), reason="audit not run")
def test_baseline_reproduced_and_ids_consistent():
    cand = json.loads((C / "candidates.json").read_text())
    assert cand["stored_candidate_hash_reproduced"]
    assert cand["graph_fast_vs_frozen_max_abs_diff"] < 1e-12
    ncc = json.loads((C / "ncc_validation.json").read_text())
    assert ncc["best_offsets_equal"] and ncc["anchors_equal"] and ncc["best_ncc_max_abs_diff"] < 1e-12
    dom = json.loads((C / "domain.json").read_text())
    assert dom["b2_status_identical"]
    src = pd.read_csv(ac.OUT / "candidate_source_audit.csv")
    assert src["moving_stable_id"].is_unique and len(src) == 1754
    finals = np.array([list(map(int, s.split(";"))) for s in src["final30"]])
    assert finals.min() >= 0 and finals.max() < 1830 and all(len(set(r)) == 30 for r in finals)


@pytest.mark.skipif(not (ac.OUT / "ncc_window_sensitivity.csv").exists(), reason="audit not run")
def test_outputs_finite():
    for name in ("ncc_window_sensitivity.csv", "candidate_k_sensitivity.csv", "spatial_residual_metrics.csv"):
        t = pd.read_csv(ac.OUT / name).select_dtypes("number")
        assert np.isfinite(t.drop(columns=[c for c in t if c.startswith("agree_")], errors="ignore").to_numpy()).all(), name
