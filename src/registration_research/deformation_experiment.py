"""Local deformation from nuclear correspondences, global pose fixed (experiment_protocol.md §4).

  A   Affine only (zero field)
  B   existing app fine warp: src.pointset_registration.matched_nuclei_rbf_fine_warp (defaults; MNN + TPS, own guards)
  B'  MNN anchors        + linear RBF of UOT conservative (validation_lib.fit_linear_rbf)
  C   B2 matched anchors + same linear RBF
  D   UOT conservative, validation_lib.run_uot unchanged
  E   B2 anchors x confidence + same linear RBF
Every field goes through the unchanged validation_lib.common_audit (existing SAFETY thresholds); a failed audit or fewer
than MIN_ANCHORS anchors -> the Affine result (applied = False).  Research preview only: never exported to Workflow D.
Result dicts have the shape validation_lib.evaluate_synthetic expects.
"""
from __future__ import annotations

import time

import numpy as np

from src.pointset_registration import matched_nuclei_rbf_fine_warp
from src.registration_evaluation import sample_displacement_field

from . import correspondence_anchors as CAn
from .legacy import modules

METHODS = ("A", "B", "B'", "C", "D", "E")
LABELS = {"A": "Affine only", "B": "MNN + TPS (existing app fine warp)", "B'": "MNN + linear RBF", "C": "B2 pairs + linear RBF",
          "D": "UOT conservative (existing)", "E": "B2 pairs x confidence + linear RBF"}


def make_grid(points_list, spacing=4.0, pad=20.0):
    P = np.vstack([np.asarray(p, float) for p in points_list])
    lo, hi = P.min(0) - pad, P.max(0) + pad
    xs, ys = np.arange(lo[0], hi[0] + spacing, spacing), np.arange(lo[1], hi[1] + spacing, spacing)
    gx, gy = np.meshgrid(xs, ys)
    return gx, gy, (float(lo[0]), float(lo[1]), float(xs[-1]), float(ys[-1]))


def _affine(method, fixed, moving, gx, gy, spacing, reason=None, attempted=None, extra=None):
    _, _, V = modules()
    zeros = np.zeros_like(gx)
    audit = {"common_audit_pass": True, "hard_safety_pass": True, "topology_percentiles_pass": True,
             "point_improvement_gate_pass": None, "rejection_reason": None, **V.field_stats(zeros, zeros, spacing)}
    out = {"method": method, "points": np.asarray(moving, float).copy(), "field_x": zeros, "field_y": zeros.copy(),
           "attempted_points": np.asarray(moving, float).copy(), "attempted_field_x": zeros, "attempted_field_y": zeros.copy(),
           "applied": method == "A", "rejection_reason": reason, "audit": audit, "diagnostics": dict(extra or {}), "runtime_seconds": 0.0}
    if attempted is not None:
        out.update(attempted)
    return out


def _audited(method, fixed, moving, pts, fx, fy, spacing, diag, t0):
    _, _, V = modules()
    audit = V.common_audit(fixed, moving, pts, fx, fy, spacing)
    res = {"method": method, "attempted_points": pts, "attempted_field_x": fx, "attempted_field_y": fy, "audit": audit,
           "diagnostics": diag, "runtime_seconds": time.perf_counter() - t0}
    if audit["common_audit_pass"]:
        return {**res, "points": pts, "field_x": fx, "field_y": fy, "applied": True, "rejection_reason": None}
    z = np.zeros_like(fx)
    return {**res, "points": np.asarray(moving, float).copy(), "field_x": z, "field_y": z.copy(), "applied": False,
            "rejection_reason": audit["rejection_reason"]}


def _rbf_from_anchors(method, a: CAn.Anchors, fixed, moving, gx, gy, spacing):
    _, _, V = modules()
    t0 = time.perf_counter()
    diag = {"anchors_n": len(a), "anchor_source": a.name}
    if len(a) < CAn.MIN_ANCHORS:
        return _affine(method, fixed, moving, gx, gy, spacing, reason="ANCHORS_INSUFFICIENT", extra=diag)
    cfg = V.UOT_CONFIGS["UOT conservative"]
    pts, fx, fy = V.fit_linear_rbf(a.positions, a.displacements, moving, gx, gy, smoothing=cfg["field_smoothing"],
                                   neighbors=cfg["field_neighbors"])
    return _audited(method, fixed, moving, pts, fx, fy, spacing, diag, t0)


def run_all(fixed, moving, gx, gy, spacing, b2_frame=None, methods=METHODS):
    """All methods on one (fixed, moving-after-global) pair.  Returns {method: result dict, ...}, {method: Anchors}."""
    _, _, V = modules()
    fixed, moving = np.asarray(fixed, float), np.asarray(moving, float)
    out, anchors = {}, {}
    if "A" in methods:
        out["A"] = _affine("A", fixed, moving, gx, gy, spacing)
    if "B" in methods:
        t0 = time.perf_counter()
        fr = matched_nuclei_rbf_fine_warp(moving, fixed)
        q = np.column_stack([gx.ravel(), gy.ravel()])
        ax = fr.attempted_displacement_x if fr.attempted_displacement_x is not None else fr.displacement_x
        ay = fr.attempted_displacement_y if fr.attempted_displacement_y is not None else fr.displacement_y
        ap = fr.attempted_transformed_points if fr.attempted_transformed_points is not None else fr.transformed_points
        own = bool(fr.applied if fr.applied is not None else fr.success)
        sx = sample_displacement_field(q, ax, ay, bounds=fr.bounds, spacing=fr.grid_spacing)
        fx, fy = sx[:, 0].reshape(gx.shape), sx[:, 1].reshape(gx.shape)
        diag = {"anchors_n": int(fr.n_filtered_pairs), "own_applied": own, "own_reason": fr.rejection_reason or fr.message}
        r = _audited("B", fixed, moving, np.asarray(ap, float), fx, fy, spacing, diag, t0)
        if not own and r["applied"]:     # the existing guards rejected it: keep that decision
            z = np.zeros_like(fx)
            r.update({"points": moving.copy(), "field_x": z, "field_y": z.copy(), "applied": False,
                      "rejection_reason": f"existing guard: {fr.rejection_reason or fr.message}"})
        out["B"] = r
        anchors["B"] = CAn.mnn(moving, fixed)    # its internal candidate pairs (before its own filtering), for TRE exclusion
    if "B'" in methods:
        anchors["B'"] = CAn.mnn(moving, fixed)
        out["B'"] = _rbf_from_anchors("B'", anchors["B'"], fixed, moving, gx, gy, spacing)
    if b2_frame is not None:
        if "C" in methods:
            anchors["C"] = CAn.b2(moving, fixed, b2_frame)
            out["C"] = _rbf_from_anchors("C", anchors["C"], fixed, moving, gx, gy, spacing)
        if "E" in methods:
            anchors["E"] = CAn.b2(moving, fixed, b2_frame, weighted=True)
            out["E"] = _rbf_from_anchors("E", anchors["E"], fixed, moving, gx, gy, spacing)
    if "D" in methods:
        try:
            out["D"] = V.run_uot(fixed, moving, gx, gy, spacing, "UOT conservative")
            out["D"]["diagnostics"]["anchors_n"] = out["D"]["diagnostics"].get("anchors_kept")
        except RuntimeError as exc:          # e.g. fewer than three anchors retained
            out["D"] = _affine("D", fixed, moving, gx, gy, spacing, reason=f"UOT failed: {exc}")
    return out, anchors
