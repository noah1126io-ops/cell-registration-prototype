"""Analysis-only tests for the Affine global-initialisation study."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import init_lib as L  # noqa: E402


def test_masked_ncc_recovers_known_shift():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 400, (500, 2))
    shape, px, lo = (110, 110), 4.0, np.array([-20.0, -20.0])
    f = L._raster(pts, lo, shape, px, 8.0); mf = L._hull_mask(pts, lo, shape, px)
    shifted = pts - np.array([24.0, -16.0])          # moving = fixed - d  -> best shift of moving is +d
    m = L._raster(shifted, lo, shape, px, 8.0); mm = L._hull_mask(shifted, lo, shape, px)
    ncc, _ = L.masked_ncc(f, mf, m, mm, min_overlap=0.3 * mf.sum())
    k = np.unravel_index(np.nanargmax(np.where(np.isfinite(ncc), ncc, -9)), ncc.shape)
    dy, dx = (k[0] - (shape[0] - 1)) * px, (k[1] - (shape[1] - 1)) * px
    assert (dx, dy) == (24.0, -16.0)


def test_correlogram_peak_at_known_translation():
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(1)
    fixed = rng.uniform(0, 800, (1200, 2))
    q = fixed[:800] - np.array([-30.0, 55.0]) + rng.normal(0, 0.8, (800, 2))
    peaks = L.correlogram_scores(cKDTree(fixed), fixed, q, 120, sigma_um=2.0)
    assert np.allclose(peaks[0][1], [-30.0, 55.0], atol=1.5) and peaks[0][0] > 10


def test_transform_errors_zero_for_identity_and_detect_chirality():
    M = 1.1 * L.rot(0.6) @ np.diag([1, -1.0]); t = np.array([10.0, 900.0])
    e = L.transform_errors(M, t, M, t, 901, 901)
    assert e["transform_rms_error_um"] < 1e-9 and e["chirality_correct"]
    assert not L.transform_errors(1.1 * L.rot(0.6), t, M, t, 901, 901)["chirality_correct"]


def test_generator_truth_consistent():
    p = L.make_pair(7, offset_um=60.0)
    sh = p.true_fixed_index >= 0
    w = L.apply_eff(p.M_true, p.t_true, p.moving_px)
    np.testing.assert_allclose(w, p.moving_world_true, atol=1e-8)
    d = np.linalg.norm(w[sh] - p.fixed[p.true_fixed_index[sh]], axis=1)
    assert np.median(d) < 3.0 and 0.5 < sh.mean() < 0.85
    assert 0.0012 < len(p.fixed) / 1010**2 < 0.0026


def test_detector_fails_unrelated_and_passes_truth():
    p = L.make_pair(8, offset_um=40.0)
    ok = L.detector(p.M_true, p.t_true, p.moving_px, p.fixed)
    assert ok["detector_status"] == "PASS"
    bad = L.detector(p.M_true, p.t_true + np.array([0.0, 45.0]), p.moving_px, p.fixed)
    assert bad["detector_status"] == "FAIL"


@pytest.mark.skipif(not (L.CACHE / "real.json").exists(), reason="not run")
def test_real_outputs_finite_and_a0_reproduces_saved():
    real = json.loads((L.CACHE / "real.json").read_text())
    a0 = next(m for m in real["methods"] if m["method_key"] == "A0")
    assert a0["vs_saved"]["transform_rms_error_um"] < 1e-6        # production re-estimate == saved affine
    for m in real["methods"]:
        assert np.isfinite(m["objective"]) and np.all(np.isfinite(m["M"]))
