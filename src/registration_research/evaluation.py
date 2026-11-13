"""Evaluation helpers (ground truth enters ONLY here; matching / reranking / deformation never receive it)."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from src.registration_evaluation import sample_displacement_field

from .legacy import modules


def q(values, prefix):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {f"{prefix}_median": np.nan, f"{prefix}_p95": np.nan, f"{prefix}_n": 0}
    return {f"{prefix}_median": float(np.median(v)), f"{prefix}_p95": float(np.percentile(v, 95)), f"{prefix}_n": int(len(v))}


def truth_index_by_coordinates(moving, fixed, matched_moving, matched_fixed, tol=1e-6):
    """True fixed index per moving point for generators that only return matched coordinate pairs (SYN-FIELD)."""
    tfi = np.full(len(moving), -1, int)
    if not len(matched_moving):
        return tfi
    dm, im = cKDTree(moving).query(matched_moving)
    df, jf = cKDTree(fixed).query(matched_fixed)
    ok = (dm <= tol) & (df <= tol)
    tfi[im[ok]] = jf[ok]
    return tfi


def field_at(points, res, bounds, spacing, key="field"):
    return sample_displacement_field(np.asarray(points, float), res[f"{key}_x"], res[f"{key}_y"], bounds=bounds, spacing=spacing)


def nuclei_tre(res, moving, fixed, true_fixed_index, bounds, spacing, exclude=None):
    """TRE of moving nuclei with a known partner after the applied field; exclude = moving indices used as anchors."""
    tfi = np.asarray(true_fixed_index, int)
    sel = tfi >= 0
    if exclude is not None and len(exclude):
        ex = np.zeros(len(moving), bool); ex[np.asarray(exclude, int)] = True
        sel &= ~ex
    idx = np.flatnonzero(sel)
    if not len(idx):
        return np.empty(0)
    after = moving[idx] + field_at(moving[idx], res, bounds, spacing)
    return np.linalg.norm(after - fixed[tfi[idx]], axis=1)


def summary_row(res, spacing):
    _, _, V = modules()
    fs = V.field_stats(res["field_x"], res["field_y"], spacing)
    afs = V.field_stats(res["attempted_field_x"], res["attempted_field_y"], spacing)
    a = res.get("audit") or {}
    return {"applied": bool(res["applied"]), "rejection_reason": res.get("rejection_reason"),
            "hard_safety_pass": a.get("hard_safety_pass"), "topology_pass": a.get("topology_percentiles_pass"),
            "point_gate_pass": a.get("point_improvement_gate_pass"), "anchors_n": (res.get("diagnostics") or {}).get("anchors_n"),
            **{f"applied_{k}": v for k, v in fs.items()}, **{f"attempted_{k}": v for k, v in afs.items()},
            "runtime_s": res.get("runtime_seconds")}


def epe_on_grid(res, true_x, true_y, mask, key="field"):
    err = np.hypot(res[f"{key}_x"] - true_x, res[f"{key}_y"] - true_y)[mask]
    return err
