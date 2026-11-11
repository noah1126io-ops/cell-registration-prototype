"""Tests for the independent generator, the plan and the freeze (analysis-only). Uses probe keys outside the plan."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import design as D  # noqa: E402
import indep_gen as G  # noqa: E402

ROOT = HERE.parents[1]


@pytest.mark.parametrize("fam", list(G.FAMILIES))
def test_truth_affine_and_persistent_ids(fam):
    p = G.make_pair(fam, "unit-test", 0)
    w = p.moving_px @ p.M_true.T + p.t_true
    np.testing.assert_allclose(w, p.moving_world_true, atol=1e-8)
    sh = p.true_fixed_index >= 0
    np.testing.assert_array_equal(p.fixed_ids[p.true_fixed_index[sh]], p.moving_ids[sh])     # ids, not geometry
    d = np.linalg.norm(w[sh] - p.fixed[p.true_fixed_index[sh]], axis=1)
    assert np.median(d) < 3.5                                   # 2 um deformation + 2 x 0.6 um centroid noise
    assert 0.0015 < p.params["observed_fixed_density"] < 0.0026
    assert 0.5 < sh.mean() < 0.85


def test_deterministic_and_keys_do_not_collide():
    a = G.make_pair("G2", "unit-test", 1); b = G.make_pair("G2", "unit-test", 1)
    np.testing.assert_array_equal(a.fixed, b.fixed)
    # keys sharing a 4-byte prefix must give different streams (regression: truncated seed keys)
    assert G.rng_for("nuclei", "G1").random() != G.rng_for("nuclei-b", "G1").random()
    assert G.rng_for("C03_off100_rot3").random() != G.rng_for("C04_off150_rot6").random()


def test_shared_fraction_and_detection_are_separate_processes():
    lo_q = G.make_pair("G1", "unit-test", 2, D.CONDITIONS["C07_shared_low"]).params
    lo_det = G.make_pair("G1", "unit-test", 2, D.CONDITIONS["C08_detect_loss"]).params
    assert lo_q["q_shared"] == 0.30 and lo_det["q_shared"] == 0.85
    assert abs(lo_q["shared_fraction_of_moving"] - lo_det["shared_fraction_of_moving"]) < 0.12   # matched observed sharing


@pytest.mark.parametrize("mode", ["unrelated", "density_matched"])
def test_no_shared_controls(mode):
    p = G.make_pair("G3", "unit-test", 0, mode=mode)
    assert (p.true_fixed_index < 0).all() and not np.isin(p.fixed_ids, p.moving_ids).any()
    # density-matched draws share the template, so tissue-scale density correlates; unrelated does not
    from scipy.ndimage import gaussian_filter
    w = p.moving_world_true
    h = lambda q: gaussian_filter(np.histogram2d(q[:, 0], q[:, 1], bins=25, range=[[100, 900], [100, 900]])[0], 1.0)  # noqa: E731
    r = np.corrcoef(h(p.fixed).ravel(), h(w).ravel())[0, 1]
    assert (r > 0.4) if mode == "density_matched" else (r < 0.4)


def test_overlap_geometry():
    a = G.rect(0, 0, 10, 10); b = G.rect(5, 5, 10, 10)
    assert G.polygon_area(G.clip_convex(a, b)) == pytest.approx(25.0)
    p = G.make_pair("G1", "unit-test", 3, dict(offset_um=0.0, mov_px=(1230, 1230)))
    assert p.params["overlap_frac_fixed"] == pytest.approx(1.0) and p.params["overlap_min_coverage"] == pytest.approx(1000**2 / (1230 * 1.1) ** 2, rel=1e-3)


def test_nonaffine_field_has_no_affine_component():
    rng = np.random.default_rng(0)
    poly = G.rect(0, 0, 1000, 1000)
    f = G.nonaffine_field(rng, (-100, -100, 1100, 1100), poly, 6.0)
    g = np.column_stack([a.ravel() for a in np.meshgrid(np.linspace(5, 995, 60), np.linspace(5, 995, 60))])
    d = f(g)
    assert np.sqrt(np.mean(np.sum(d**2, 1))) == pytest.approx(6.0, rel=0.1)
    coef, *_ = np.linalg.lstsq(np.column_stack([g, np.ones(len(g))]), d, rcond=None)
    assert np.abs(coef[:2]).max() < 2e-3 and np.abs(coef[2]).max() < 0.6


def test_common_region_error_zero_for_truth_and_matches_offset():
    p = G.make_pair("G4", "unit-test", 0)
    e = G.affine_errors_common(p.M_true, p.t_true, p)
    assert e["rms_common_um"] < 1e-9 and e["n_common_grid"] > 100
    e2 = G.affine_errors_common(p.M_true, p.t_true + np.array([3.0, 4.0]), p)
    assert e2["rms_common_um"] == pytest.approx(5.0)


def test_plan_fixed_size():
    assert len(D.main_cases()) == 112 and len(D.CONTROLS) == 24 and len(D.downstream_jobs()) == 344


def test_frozen_components_unchanged():
    man = json.loads((HERE / "freeze/frozen_components.json").read_text())
    for name, c in man["components"].items():
        assert hashlib.sha256((ROOT / c["path"]).read_bytes()).hexdigest() == c["sha256"], name


def test_generator_independent_of_prior_generator():
    import ast
    tree = ast.parse((HERE / "indep_gen.py").read_text())
    called = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "") for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not called & {"_intensity", "_sample"}            # prior generator internals never called
    imports = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    # init_lib is imported only lazily inside SectionPair.to_init_pair (container type), never for generation
    gen_funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert not any("init_lib" in ast.dump(f) for f in gen_funcs) and "init_lib" in imports
