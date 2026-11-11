"""Analysis-only tests: python -B -m pytest -q test_study.py"""
from __future__ import annotations

import numpy as np

import metrics as MX
import nonrigid as NR
import stages as S

L = S.L


def test_hyp_affine_matches_transform_hyp():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 900, (50, 2))
    h = {"chirality": "reflect", "theta": 7.0, "scale": 1.1, "translation": [3.0, -4.0]}
    M, t = S.hyp_affine(h)
    np.testing.assert_allclose(p @ M.T + t, L._transform_hyp(p, "reflect", 7.0, 1.1) + t, atol=1e-10)


def test_finish_wrapper_is_restored_and_transparent():
    from indep_gen import make_pair  # noqa: F401  (import path check only)
    rng = np.random.default_rng(1)
    fixed = rng.uniform(0, 600, (900, 2))
    M = 1.1 * L.rot(2.0) @ np.diag([1.0, -1.0]); t = np.array([20.0, 640.0])
    mov = np.linalg.solve(M, (fixed - t).T).T + rng.normal(0, 0.3, fixed.shape)
    before = L._finish
    out = S.run_a2_stages(mov, fixed)
    assert L._finish is before
    plain = L.run_method("A2", mov, fixed, None, None)
    np.testing.assert_allclose(out["stages"]["R2"]["M"], plain.M, atol=1e-12)
    np.testing.assert_allclose(out["stages"]["R2"]["t"], plain.t, atol=1e-9)
    # pre-ICP stage is a similarity with the right chirality
    Ma = np.asarray(out["stages"]["R1a"]["M"])
    assert np.linalg.det(Ma) < 0
    assert abs(np.linalg.svd(Ma)[1][0] / np.linalg.svd(Ma)[1][1] - 1) < 1e-9


def test_displacement_is_not_error():
    p = np.array([[0.0, 0.0], [10.0, 0.0]])
    d = MX.displacement_between(np.eye(2), np.zeros(2), np.eye(2), np.array([3.0, 4.0]), p)
    assert d["disp_median_um"] == 5.0


def test_nn_block_perfect_alignment():
    rng = np.random.default_rng(2)
    f = rng.uniform(0, 500, (400, 2))
    m = MX.nn_block(f + 0.1, f)
    assert m["m2f_within_1um"] == 1.0 and m["mutual_nn_rate"] == 1.0


def test_proxy_pairs_mutual_only():
    f = np.array([[0.0, 0.0], [100.0, 0.0]])
    m = np.array([[1.0, 0.0], [1.5, 0.0], [100.0, 3.0]])
    i, j, u = NR.proxy_pairs(m, f, r=4.0)
    assert list(i) == [0, 2] and list(j) == [0, 1]
    np.testing.assert_allclose(u[1], [0.0, -3.0])


def test_autocorr_detects_smooth_field_and_not_noise():
    rng = np.random.default_rng(3)
    pos = rng.uniform(0, 1000, (1500, 2))
    noise = rng.normal(0, 1, pos.shape)
    smooth = np.column_stack([np.sin(pos[:, 0] / 150), np.cos(pos[:, 1] / 150)]) * 2
    a = NR.vector_autocorr(pos, noise); b = NR.vector_autocorr(pos, smooth + noise)
    assert abs(a["rho_0_20"]) < 0.1 and b["rho_0_20"] > 0.5


def test_heldout_skill_null_vs_signal():
    rng = np.random.default_rng(4)
    pos = rng.uniform(0, 1000, (1500, 2))
    noise = rng.normal(0, 1, pos.shape)
    smooth = np.column_stack([np.sin(pos[:, 0] / 150), np.cos(pos[:, 1] / 150)]) * 2
    s0 = NR.heldout_skill(pos, noise, 100.0, 0, np.zeros(2))["skill"]
    s1 = NR.heldout_skill(pos, smooth + noise, 100.0, 0, np.zeros(2))["skill"]
    assert s0 < 0.05 and s1 > 0.5


def test_heldout_never_uses_heldout_vectors_for_fit():
    rng = np.random.default_rng(5)
    pos = rng.uniform(0, 1000, (800, 2))
    vec = rng.normal(0, 1, pos.shape)
    col = NR.checker_colour(pos, 100.0, np.zeros(2))
    a = NR.heldout_skill(pos, vec, 100.0, 0, np.zeros(2))
    vec2 = vec.copy(); vec2[col == 0] += 50.0           # change held-out vectors only
    b = NR.heldout_skill(pos, vec2, 100.0, 0, np.zeros(2))
    assert a["bandwidth"] == b["bandwidth"] and a["n_fit"] == b["n_fit"]


def test_semisynthetic_zero_amplitude_has_no_field():
    rng = np.random.default_rng(6)
    f = rng.uniform(0, 800, (1500, 2))
    mv, d = NR.semisynthetic(f, None, q_pair=0.7, sigma_pair=0.6, n_moving=1400, amp_um=0.0, rng=rng)
    assert len(mv) == 1400 and np.all(d == 0)


def test_decompose_reflection_and_scale():
    M = 1.1 * L.rot(2.0) @ np.diag([1.0, -1.0])
    d = MX.decompose(M, np.zeros(2))
    assert d["determinant"] < 0 and abs(d["scale_sqrt_absdet"] - 1.1) < 1e-12
    assert abs(d["rotation_deg_after_chirality"] - 2.0) < 1e-9
