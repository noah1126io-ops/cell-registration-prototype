from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: runs the frozen pipeline on real point sets (minutes)")


def _rot(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def make_synthetic(seed=7, size=300.0, spacing=12.0, shared=0.85, extra=0.10, amp=3.0):
    """IF nuclei on a jittered grid; HE = shared subset (+unique nuclei) seen through a known global affine and a smooth
    local displacement.  Truth: he_true_if[i] = IF index of HE nucleus i, or -1.  Used only for evaluation, never as input."""
    rng = np.random.default_rng(seed)
    g = np.arange(spacing / 2, size, spacing)
    gx, gy = np.meshgrid(g, g)
    if_pts = np.column_stack([gx.ravel(), gy.ravel()]) + rng.normal(0, spacing * 0.22, (gx.size, 2))
    n_if = len(if_pts)
    keep = rng.random(n_if) < shared
    sh = np.flatnonzero(keep)
    # true forward HE -> IF:  x_IF = A(x_HE) + d(A(x_HE)),  d smooth (amplitude `amp` um)
    M, t = 1.02 * _rot(9.0), np.array([40.0, -25.0])
    d = lambda p: amp * np.column_stack([np.sin(p[:, 1] / 70.0), np.cos(p[:, 0] / 85.0)])  # noqa: E731
    target = if_pts[sh]
    q = target.copy()
    for _ in range(50):                       # invert x + d(x) = target
        q = target - d(q)
    Mi = np.linalg.inv(M)
    he_shared = (q - t) @ Mi.T + rng.normal(0, 0.3, (len(sh), 2))
    n_extra = int(extra * len(sh))
    lo, hi = he_shared.min(0), he_shared.max(0)
    he_extra = rng.uniform(lo, hi, (n_extra, 2))
    he_pts = np.vstack([he_shared, he_extra])
    perm = rng.permutation(len(he_pts))
    he_pts = he_pts[perm]
    true = np.concatenate([sh, -np.ones(n_extra, int)])[perm]
    he_ids = np.asarray([f"he-{1000 + i}" for i in range(len(he_pts))], dtype=object)    # IDs that are NOT the indices
    if_ids = np.arange(n_if) * 3 + 50000
    return {"he_pts": he_pts, "he_ids": he_ids, "if_pts": if_pts, "if_ids": if_ids, "M": M, "t": t, "true": true, "d": d,
            "roi": [10.0, 10.0, size - 10.0, size - 10.0]}


@pytest.fixture(scope="session")
def syn():
    return make_synthetic()


@pytest.fixture(scope="session")
def syn_auto(syn):
    from src.registration_research import cell_review as CR
    return CR.compute_auto(syn["he_pts"], syn["he_ids"], syn["if_pts"], syn["if_ids"], syn["M"], syn["t"], roi=syn["roi"],
                           meta={"sample_id": "SYN"})
