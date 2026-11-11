"""Researcher-specified reference transforms (2026-10-27): landmark fitting with outlier rejection and held-out error,
and localization metrics of automatic candidates against a reference.

A reference transform is a provisional, researcher-specified transform (landmarks or manual adjustment).  It is
never treated as ground truth: it is stored MANUAL_UNVERIFIED, is never accepted, and is only used to *diagnose*
the automatic search (is the reference region searched, where does it rank, how far is each candidate).
Convention: fixed = M @ moving + t.
"""
from __future__ import annotations

import numpy as np

from .workflow import similarity_from_landmarks

LANDMARK_TYPES = {"vessel": "血管", "duct": "腺管", "lumen": "腔構造", "boundary": "組織の境界", "cleft": "間隙・裂隙",
                  "other": "その他の形態的構造"}
CONFIDENCE = {"high": "高", "medium": "中", "low": "低"}


def fit_reference(fixed_xy, moving_xy, *, k_mad=3.0, floor_um=15.0, max_iter=5):
    """Similarity from >= 2 pairs with iterative outlier rejection (residual > max(k * 1.4826 * MAD, floor)).

    Returns dict: M, t, inliers (bool list), residual_um (all pairs, final fit), loo_um (leave-one-out error of each
    inlier pair, needs >= 3 inliers), holdout (every 3rd inlier held out when >= 6 inliers: fit on the rest).
    """
    Q, P = np.asarray(fixed_xy, float), np.asarray(moving_xy, float)
    n = len(P)
    if n < 2:
        raise ValueError("at least 2 landmark pairs are needed")
    inl = np.ones(n, bool)
    for _ in range(max_iter):
        M, t, _ = similarity_from_landmarks(Q[inl], P[inl])
        res = np.linalg.norm(P @ M.T + t - Q, axis=1)
        if inl.sum() < 4:
            break
        r_in = res[inl]
        thr = max(k_mad * 1.4826 * float(np.median(np.abs(r_in - np.median(r_in)))) + float(np.median(r_in)), floor_um)
        new = res <= thr
        if new.sum() < 3 or (new == inl).all():
            break
        inl = new
    M, t, _ = similarity_from_landmarks(Q[inl], P[inl])
    res = np.linalg.norm(P @ M.T + t - Q, axis=1)
    idx = np.flatnonzero(inl)
    loo = []
    if len(idx) >= 3:
        for i in idx:
            keep = idx[idx != i]
            Mi, ti, _ = similarity_from_landmarks(Q[keep], P[keep])
            loo.append(float(np.linalg.norm(Mi @ P[i] + ti - Q[i])))
    hold = None
    if len(idx) >= 6:
        test = idx[2::3]; train = np.setdiff1d(idx, test)
        Mh, th, _ = similarity_from_landmarks(Q[train], P[train])
        e = np.linalg.norm(P[test] @ Mh.T + th - Q[test], axis=1)
        hold = {"n_train": int(len(train)), "n_test": int(len(test)), "test_index": test.tolist(),
                "error_um": e.tolist(), "median_um": float(np.median(e)), "max_um": float(e.max())}
    return {"M": M, "t": t, "inliers": inl.tolist(), "n": int(n), "n_inliers": int(inl.sum()), "residual_um": res.tolist(),
            "residual_median_um": float(np.median(res[inl])), "loo_um": loo,
            "loo_median_um": float(np.median(loo)) if loo else None, "holdout": hold, "params": params(M)}


def params(M):
    M = np.asarray(M, float)
    det = float(np.linalg.det(M))
    C = np.diag([1.0, -1.0]) if det < 0 else np.eye(2)
    sR = M @ C
    sv = np.linalg.svd(M)[1]
    return {"scale": float(np.sqrt(abs(det))), "rotation_deg": float(np.degrees(np.arctan2(sR[1, 0], sR[0, 0]))),
            "reflect": det < 0, "anisotropy": float(sv[0] / sv[1])}


def probe_points(moving_extent, r=400.0):
    c = np.asarray(moving_extent, float) / 2.0
    return c + np.array([[0, 0], [r, 0], [0, r], [-r, 0], [0, -r]], float)


def compare(M, t, M_ref, t_ref, moving_extent, landmarks=None):
    """Localization metrics of one transform against a reference (fixed-frame um)."""
    pr = probe_points(moving_extent)
    d = np.linalg.norm(pr @ np.asarray(M).T + np.asarray(t) - (pr @ np.asarray(M_ref).T + np.asarray(t_ref)), axis=1)
    pa, pb = params(M), params(M_ref)
    out = {"probe_median_um": float(np.median(d)), "centre_um": float(d[0]), "probe_max_um": float(d.max()),
           "d_rotation_deg": float((pa["rotation_deg"] - pb["rotation_deg"] + 180) % 360 - 180),
           "scale_ratio": pa["scale"] / pb["scale"], "same_chirality": pa["reflect"] == pb["reflect"]}
    if landmarks:
        Q = np.asarray([p["fixed"] for p in landmarks], float); P = np.asarray([p["moving"] for p in landmarks], float)
        e = np.linalg.norm(P @ np.asarray(M).T + np.asarray(t) - Q, axis=1)
        out["landmark_error_median_um"] = float(np.median(e)); out["landmark_error_max_um"] = float(e.max())
    return out
