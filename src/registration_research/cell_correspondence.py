"""Nuclear correspondence for one global candidate, by the frozen Graph / UOT / B2 implementations.

For a candidate pose the moving nuclei are mapped into the fixed frame, both sides are restricted to the imaged
overlap, and the unchanged 2026-10-14 pipeline is applied:
  context  = experiment_lib.build_fixed_context(fixed, moving)       (regional NCC, fingerprints, 30 candidates, graph)
  B0       = Position + Graph UOT        B1 = relative-cost Graph-UOT        B3 = full contextual UOT
  B2       = B1 + frozen synthetic-trained adoption models (match / unmatched thresholds from Validation)
Nothing is refitted.  UOT mass is NOT a probability of identity; B2 confidence is synthetic-derived, not calibrated
on real data.  A mirror decoy (moving overlap reflected about its centroid, as in the C3 mirror null) is run through
the same pipeline so that a candidate's adoption can be compared with what the same density/overlap gives by chance.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.registration_core import metrics as mt
from src.registration_core.pipeline import footprint

from .legacy import frozen_b2, modules

MIN_POINTS = 40            # prespecified: fewer nuclei on either side of the overlap -> not evaluated (30 candidates/row, kNN 20)
METHODS = ("B0", "B1", "B3", "B2")
GRID_UM = 100.0            # spatial-coverage grid
NEIGH_UM = 60.0            # local-consistency neighbourhood


def map_points(M, t, pts):
    return np.asarray(pts, float) @ np.asarray(M, float).T + np.asarray(t, float)


def overlap_subset(M, t, fixed_pts, fixed_extent, moving_pts, moving_extent):
    """Indices of nuclei inside the imaged overlap of the two fields under the pose, and the overlap geometry."""
    fixed_pts, moving_pts = np.asarray(fixed_pts, float), np.asarray(moving_pts, float)
    mw = map_points(M, t, moving_pts)
    fp_m = map_points(M, t, footprint(moving_extent))
    fp_f = footprint(fixed_extent)
    fix_sel = np.flatnonzero(mt.inside_poly(fixed_pts, fp_m, 0.0) & mt.inside_poly(fixed_pts, fp_f, 0.0))
    mov_sel = np.flatnonzero(mt.inside_poly(mw, fp_f, 0.0))
    fe = np.asarray(fixed_extent, float)
    res = 4.0
    g = np.stack(np.meshgrid(np.arange(res / 2, fe[0], res), np.arange(res / 2, fe[1], res)), -1).reshape(-1, 2)
    area = float(mt.inside_poly(g, fp_m, 0.0).sum() * res * res / 1e6)
    return {"fixed_idx": fix_sel, "moving_idx": mov_sel, "moving_mapped": mw, "overlap_area_mm2": area,
            "overlap_frac_of_moving": area / (np.prod(moving_extent) / 1e6), "overlap_polygon": fp_m}


def _sample(fixed, moving, true_fixed_index, seed, label):
    """Same SimpleNamespace shape that the 2026-10-14 study used for its real application."""
    n = len(moving)
    tfi = np.full(n, -1, int) if true_fixed_index is None else np.asarray(true_fixed_index, int)
    return SimpleNamespace(fixed_points=np.asarray(fixed, float), moving_points=np.asarray(moving, float),
                           fixed_ids=np.asarray([f"f{i}" for i in range(len(fixed))], dtype=object),
                           moving_ids=np.asarray([f"m{i}" for i in range(n)], dtype=object), true_fixed_index=tfi,
                           common_moving_mask=tfi >= 0, seed=int(seed), difficulty=label)


def run_pipeline(fixed, moving, true_fixed_index=None, seed=0, label="candidate"):
    """Frozen B0/B1/B3 + B2 on two point sets that are already in one frame.  Returns per-nucleus frames + diagnostics."""
    E, _, _ = modules()
    t0 = time.perf_counter()
    ctx = E.build_fixed_context(np.asarray(fixed, float), np.asarray(moving, float))
    t_ctx = time.perf_counter() - t0
    runs = E.method_runs(ctx)
    sample = _sample(fixed, moving, true_fixed_index, seed, label)
    scen = {"scenario_id": label, "split": "Research", "strength": label, "template_group": label}
    frames = {}
    for key, name in (("B0", E.METHOD_B0), ("B1", E.METHOD_B1), ("B3", E.METHOD_B3)):
        frames[key] = E.result_frame(sample, scen, ctx, runs[name], method=name)
    b2 = frozen_b2()
    frames["B2"] = E.apply_separated_adoption(frames["B1"], b2["match_model"], b2["unmatched_model"], b2["match_threshold"],
                                              b2["unmatched_threshold"], method=E.METHOD_B2)
    uot = {k: runs[name].metrics for k, name in (("B0", E.METHOD_B0), ("B1", E.METHOD_B1), ("B3", E.METHOD_B3))}
    return {"frames": frames, "uot": uot, "context": ctx, "runtime_s": time.perf_counter() - t0, "context_runtime_s": t_ctx}


def matched_pairs(frame: pd.DataFrame):
    m = frame["status"].to_numpy(str) == "matched"
    return frame["moving_index"].to_numpy(int)[m], frame["top1_fixed_index"].to_numpy(int)[m]


def spatial_summary(fixed, moving, mi, fi, area_poly=None):
    """Coverage of matched pairs over a 100 um grid and local displacement consistency (no ground truth needed)."""
    out = {"pairs_n": int(len(mi))}
    if len(mi) == 0:
        return {**out, "coverage_cells": 0.0, "local_dev_median_um": None, "local_dev_le3_frac": None, "disp_median_um": None,
                "duplicate_destinations": 0}
    d = fixed[fi] - moving[mi]
    cells_all = {tuple(c) for c in np.floor(moving / GRID_UM).astype(int)}
    occupied = np.unique(np.floor(moving / GRID_UM).astype(int), axis=0, return_counts=True)
    eligible = {tuple(c) for c, n in zip(*occupied) if n >= 3}
    with_pair = {tuple(c) for c in np.floor(moving[mi] / GRID_UM).astype(int)}
    tree = cKDTree(moving[mi])
    dev = []
    for k, nb in enumerate(tree.query_ball_point(moving[mi], NEIGH_UM)):
        nb = [j for j in nb if j != k]
        if len(nb) >= 3:
            dev.append(float(np.linalg.norm(d[k] - np.median(d[nb], axis=0))))
    dev = np.asarray(dev)
    return {**out, "coverage_cells": len(with_pair & eligible) / max(len(eligible), 1), "cells_eligible": len(eligible),
            "cells_total": len(cells_all), "local_dev_median_um": float(np.median(dev)) if len(dev) else None,
            "local_dev_le3_frac": float(np.mean(dev <= 3.0)) if len(dev) else None, "local_dev_n": int(len(dev)),
            "disp_median_um": float(np.median(np.linalg.norm(d, axis=1))),
            "duplicate_destinations": int(len(fi) - len(np.unique(fi)))}


def truth_metrics(frame: pd.DataFrame):
    """Matching precision/recall/F1 and unmatched detection against known IDs (synthetic only)."""
    common = frame["is_common_cell"].to_numpy(bool)
    correct = frame["top1_correct"].to_numpy(bool)
    st = frame["status"].to_numpy(str)
    matched, unm = st == "matched", st == "unmatched"
    tp = int(np.sum(matched & correct))
    p = tp / max(int(matched.sum()), 1) if matched.any() else float("nan")
    r = tp / max(int(common.sum()), 1) if common.any() else float("nan")
    f1 = 2 * p * r / (p + r) if (p == p and r == r and p + r > 0) else float("nan")
    up = float(np.sum(unm & ~common) / max(unm.sum(), 1)) if unm.any() else float("nan")
    ur = float(np.sum(unm & ~common) / max((~common).sum(), 1)) if (~common).any() else float("nan")
    return {"common_n": int(common.sum()), "matched_tp": tp, "false_matches": int(np.sum(matched & ~correct)), "precision": p,
            "recall": r, "f1": f1, "top1_acc_common": float(np.mean(correct[common])) if common.any() else float("nan"),
            "unmatched_precision": up, "unmatched_recall": ur}


def summarize(res, fixed, moving, has_truth=False):
    out = {"runtime_s": res["runtime_s"], "n_fixed": int(len(fixed)), "n_moving": int(len(moving))}
    for key in METHODS:
        f = res["frames"][key]
        st = f["status"].value_counts()
        n = len(f)
        out[f"{key}_matched_n"] = int(st.get("matched", 0))
        out[f"{key}_ambiguous_n"] = int(st.get("ambiguous", 0))
        out[f"{key}_unmatched_n"] = int(st.get("unmatched", 0))
        out[f"{key}_matched_frac"] = out[f"{key}_matched_n"] / max(n, 1)
        if has_truth:
            out.update({f"{key}_{k}": v for k, v in truth_metrics(f).items()})
    b0, b1 = res["frames"]["B0"], res["frames"]["B1"]
    out["graph_support_median"] = float(np.median(b0["top1_graph_support"]))
    out["graph_support_ge05_frac"] = float(np.mean(b0["top1_graph_support"] >= 0.5))
    u = res["uot"]["B1"]
    out.update({"uot_mass_ratio_to_source": u["total_mass_ratio_to_source"], "uot_row_mass_ratio_median": u["row_mass_ratio_median"],
                "uot_entropy_median": u["normalized_row_entropy_median"], "uot_objective": u["solver_final_objective"],
                "uot_converged": bool(u["solver_converged"]), "uot_iterations": u["solver_iterations"],
                "uot_objective_per_row": u["solver_final_objective"] / max(len(moving), 1)})
    out["b2_confidence_median"] = float(np.median(res["frames"]["B2"]["calibrated_confidence"]))
    mi, fi = matched_pairs(res["frames"]["B2"])
    out.update({f"B2_{k}": v for k, v in spatial_summary(fixed, moving, mi, fi).items()})
    return out


def mirror_decoy(moving):
    """Reflect about the centroid x (registration_core.metrics.mirror_null convention): density kept, identity destroyed."""
    m = np.asarray(moving, float).copy()
    c = m.mean(0)
    m[:, 0] = 2 * c[0] - m[:, 0]
    return m


def evaluate_candidate(cand, fixed_pts, fixed_extent, moving_pts, moving_extent, *, which="final", truth=None, seed=0,
                       with_decoy=True, keep_frames=False):
    """Phase 1 record for one candidate.  truth = (shared_moving_idx, shared_fixed_idx) in the full point sets (synthetic)."""
    M, t = cand.T(which)
    ov = overlap_subset(M, t, fixed_pts, fixed_extent, moving_pts, moving_extent)
    fi_sel, mi_sel = ov["fixed_idx"], ov["moving_idx"]
    rec = {"candidate": cand.id, "source": cand.source, "c1_rank": cand.rank, "c1_score": cand.score, "c1_verdict": cand.verdict,
           "pose": which, "overlap_area_mm2": ov["overlap_area_mm2"], "overlap_frac_of_moving": ov["overlap_frac_of_moving"],
           "n_fixed_overlap": int(len(fi_sel)), "n_moving_overlap": int(len(mi_sel)),
           "c3_le_3um": cand.c3.get("le_3um"), "c3_inlier_over_null": cand.c3.get("inlier_over_null")}
    if len(fi_sel) < MIN_POINTS or len(mi_sel) < MIN_POINTS:
        return {**rec, "status": "TOO_FEW_NUCLEI"}, None
    fixed = np.asarray(fixed_pts, float)[fi_sel]
    moving = ov["moving_mapped"][mi_sel]
    tfi = None
    if truth is not None:
        sm, sf = truth
        f_pos = {int(j): k for k, j in enumerate(fi_sel)}
        partner = {int(m): int(f) for m, f in zip(sm, sf) if f >= 0}
        tfi = np.asarray([f_pos.get(partner.get(int(m), -1), -1) for m in mi_sel], int)
    res = run_pipeline(fixed, moving, tfi, seed, cand.id)
    rec.update({"status": "OK", **summarize(res, fixed, moving, has_truth=truth is not None)})
    if truth is not None:
        rec["true_pairs_in_overlap"] = int(np.sum(tfi >= 0))
    if with_decoy:
        dec = run_pipeline(fixed, mirror_decoy(moving), None, seed, cand.id + ":decoy")
        ds = summarize(dec, fixed, mirror_decoy(moving))
        rec.update({f"decoy_{k}": ds[k] for k in ("B2_matched_frac", "B1_matched_frac", "B0_matched_frac", "graph_support_median",
                                                  "uot_mass_ratio_to_source", "uot_objective_per_row")})
    detail = None
    if keep_frames:
        detail = {"fixed_idx": fi_sel, "moving_idx": mi_sel, "fixed": fixed, "moving_mapped": moving, "frames": res["frames"],
                  "overlap_polygon": ov["overlap_polygon"]}
    return rec, detail
