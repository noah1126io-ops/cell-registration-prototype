"""Tests for the research-only Region-Cell Matching study (2026-11-13).

Baseline protection: reused files unchanged, frozen B2 / UOT conservative reproduced numerically, engine candidates read
back identically; safety: truth isolation, anchor minimum, decoy, reranking rule; UI default off.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
STUDY = REPO / "analysis/region-cell-matching-20261113"
MANIFEST = json.loads((STUDY / "baseline_manifest.json").read_text())


def test_reused_research_files_unchanged():
    from src.registration_research import legacy
    assert legacy.reused_file_hashes() == MANIFEST["reused_research_sha256"]


# the only existing file this study is allowed to touch: the Explorer page gets a default-off research tab
ALLOWED_TO_CHANGE = {"src/registration_explorer/page.py"}


def test_production_and_engine_files_unchanged():
    from src.registration_research import legacy
    changed = [rel for rel, h in MANIFEST["production_and_engine_sha256"].items() if legacy.sha256_file(REPO / rel) != h]
    assert set(changed) <= ALLOWED_TO_CHANGE, changed


def test_frozen_b2_models_are_the_saved_ones():
    from src.registration_research import legacy
    b2 = legacy.frozen_b2()
    d = json.loads(legacy.B2_SUMMARY.read_text())
    assert b2["match_threshold"] == d["b2_thresholds"]["match"]["threshold"]
    assert b2["unmatched_threshold"] == d["b2_thresholds"]["unmatched"]["threshold"]
    np.testing.assert_array_equal(b2["match_model"].coefficients, np.asarray(d["b2_match_model"]["coefficients"]))


@pytest.mark.slow
def test_real_b0_b1_b3_b2_reproduce_2026_10_14():
    """Re-running the frozen pipeline through the adapter reproduces the 10-14 real per-nucleus table exactly."""
    from src.registration_research import cell_correspondence as CC
    from src.registration_research.legacy import GRAPH_STUDY, modules
    E, _, _ = modules()
    fixed = np.load(E.REAL_INPUT / "success_metric_fixed_points.npy")
    moving = np.load(E.REAL_INPUT / "success_metric_moving_points.npy")
    res = CC.run_pipeline(fixed, moving, None, 20260904, "parity")
    ref = pd.read_csv(GRAPH_STUDY / "real_exploratory_per_nucleus.csv")
    for key, name in (("B0", E.METHOD_B0), ("B1", E.METHOD_B1), ("B3", E.METHOD_B3), ("B2", E.METHOD_B2)):
        r = ref[ref["method"] == name].sort_values("moving_index")
        f = res["frames"][key].sort_values("moving_index")
        assert list(f["status"]) == list(r["status"]), key
        np.testing.assert_array_equal(f["top1_fixed_index"].to_numpy(), r["top1_fixed_index"].to_numpy())
        np.testing.assert_allclose(f["calibrated_confidence"].to_numpy(), r["calibrated_confidence"].to_numpy(), rtol=0, atol=1e-9)


def test_uot_conservative_reproduces_2026_09_11():
    """Method D (validation_lib.run_uot, unchanged) on SYN-FIELD seed 1101 scenario_00 = the stored 09-11 value."""
    import sys
    from src.registration_research import deformation_experiment as DX
    from src.registration_research.legacy import UOT_STUDY, modules
    _, _, V = modules()
    sys.path.insert(0, str(UOT_STUDY))
    import validate_synthetic as VS
    s = VS.scenario_table()[0]
    smp = V.generate_synthetic_sample(seed=1101, amplitude_um=s["amplitude_um"], missing_fraction=s["missing_fraction"],
                                      unmatched_fraction=s["unmatched_fraction"], overlap_fraction=s["overlap_fraction"], noise_um=s["noise_um"])
    out, _ = DX.run_all(smp.fixed_points, smp.moving_points, smp.grid_x, smp.grid_y, smp.spacing, methods=("D",))
    ev = V.evaluate_synthetic(smp, {**out["D"], "method": "UOT conservative"})
    ref = pd.read_csv(UOT_STUDY / "synthetic_metrics.csv")
    row = ref[(ref["method"] == "UOT conservative") & (ref["seed"] == 1101) & (ref["scenario_id"] == s["scenario_id"])].iloc[0]
    assert abs(ev["field_error_um_median"] - row["field_error_um_median"]) < 1e-9


def test_candidate_adapter_reads_engine_records_unchanged():
    from src.registration_research import candidate_adapter as CA
    p = REPO / "analysis/adaptive-registration-validation-20261111/runs/P6_2_orig__D/result.json"
    res = json.loads(p.read_text())
    cands = CA.from_benchmark_result(p, "same")
    raw = res["gates"]["same"]["candidates"]
    assert [c.id for c in cands] == [r["id"] for r in raw]
    for c, r in zip(cands, raw):
        np.testing.assert_array_equal(c.M_final, np.asarray(r["T_final"]["M"]))
        np.testing.assert_array_equal(c.t_final, np.asarray(r["T_final"]["t"]))


def _small_case():
    from src.registration_research.legacy import modules
    E, _, _ = modules()
    return E.generate_real_like_sample(seed=4101, strength="Weak")


def test_truth_never_changes_matching_or_adoption():
    from src.registration_research import cell_correspondence as CC
    s = _small_case()
    a = CC.run_pipeline(s.fixed_points, s.moving_points, s.true_fixed_index, 1, "with-truth")
    b = CC.run_pipeline(s.fixed_points, s.moving_points, None, 1, "no-truth")
    for k in CC.METHODS:
        assert list(a["frames"][k]["status"]) == list(b["frames"][k]["status"]), k
        np.testing.assert_array_equal(a["frames"][k]["top1_fixed_index"], b["frames"][k]["top1_fixed_index"])


def test_mirror_decoy_keeps_density_and_destroys_identity():
    from src.registration_research.cell_correspondence import mirror_decoy
    pts = np.random.default_rng(0).uniform(0, 100, (50, 2))
    m = mirror_decoy(pts)
    assert len(m) == len(pts)
    np.testing.assert_allclose(m.mean(0), pts.mean(0))
    assert not np.allclose(m, pts)


def test_too_few_anchors_falls_back_to_affine():
    from src.registration_research import correspondence_anchors as CAn
    from src.registration_research import deformation_experiment as DX
    rng = np.random.default_rng(1)
    fixed = rng.uniform(0, 200, (60, 2))
    moving = fixed + 50.0              # nothing within 10 um -> no MNN anchors
    gx, gy, _ = DX.make_grid([fixed, moving])
    out, anchors = DX.run_all(fixed, moving, gx, gy, 4.0, methods=("A", "B'"))
    assert len(anchors["B'"]) < CAn.MIN_ANCHORS
    assert out["B'"]["applied"] is False and out["B'"]["rejection_reason"] == "ANCHORS_INSUFFICIENT"
    assert np.all(out["B'"]["field_x"] == 0)


def test_reranking_rule_is_the_prespecified_one():
    from src.registration_research import candidate_reranking as RR
    recs = [{"candidate": "p#0", "status": "OK", "B2_matched_frac": 0.02, "decoy_B2_matched_frac": 0.01},
            {"candidate": "p#1", "status": "OK", "B2_matched_frac": 0.40, "decoy_B2_matched_frac": 0.01},
            {"candidate": "p#2", "status": "TOO_FEW_NUCLEI"}]
    d = RR.decide(recs)
    assert d["top"] == "p#1" and d["supported"] and d["order"][-1] == "p#2"
    recs[1]["B2_matched_frac"] = 0.05          # excess 0.04 < 0.10 -> not supported
    assert RR.decide(recs)["supported"] is False
    assert RR.RULE == {"min_excess": 0.10, "min_ratio_to_decoy": 3.0, "decoy_floor": 0.01, "min_gap": 0.05}
    assert math.isinf(RR.excess({"status": "TOO_FEW_NUCLEI"}))


def test_explorer_research_tab_is_off_by_default(monkeypatch):
    from src.registration_explorer import region_cell_ui as U
    monkeypatch.delenv(U.ENV, raising=False)
    assert U.enabled() is False
    monkeypatch.setenv(U.ENV, "1")
    assert U.enabled() is True


def test_explorer_page_keeps_existing_tabs_when_disabled():
    src = (REPO / "src/registration_explorer/page.py").read_text()
    assert ('tab_labels = ["対応領域の探索", "Viewer（重ね合わせ・手動調整）", "評価・注釈", "方式比較", "Warp Preview", "新規実行", "実験記録", '
            '"Artifact"]') in src
    assert "tabs = st.tabs(tab_labels + ([rcm_ui.TAB_LABEL] if show_rcm else []))" in src


def test_research_tab_renders_saved_results_read_only(monkeypatch, tmp_path):
    """AppTest: the tab renders the saved study results for 031-6_2 and 031-8_6 without exceptions or writes."""
    from streamlit.testing.v1 import AppTest
    from src.registration_research import legacy
    before = {p: legacy.sha256_file(p) for p in sorted((STUDY / "runs").glob("p1_*.json"))}
    if not before:
        pytest.skip("no saved Phase 1 results yet")
    for sample in ("031-6_2", "031-8_6"):
        def app(sample):
            import sys
            sys.path.insert(0, "/mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu")
            from src.registration_explorer import region_cell_ui as U
            U.render(sample)
        at = AppTest.from_function(app, kwargs={"sample": sample}, default_timeout=120).run()
        assert not at.exception, at.exception
        assert len(at.dataframe) >= 1
    after = {p: legacy.sha256_file(p) for p in before}
    assert before == after
