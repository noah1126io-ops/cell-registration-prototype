"""E0x ablation (X0-X3) and top-candidate diagnostics, built on the 2026-10-19 research module `oir` (imported unmodified).

  X0  canvas = fixed nuclei bbox (+40 um, A2-style)   ICP on all points (A2's production refine)
  X1  canvas = union of both imaged extents (E0x)      ICP on all points
  X2  canvas = fixed nuclei bbox                       ICP restricted to the other side's mapped footprint (E0x)
  X3  canvas = union (E0x)                             ICP restricted (E0x)  == E0x

Everything else (pose grid, masked NCC, robust z, distinct-candidate rule, verdict rule, acceptance) is the E0x setting.
Transform convention: fixed_world = A @ moving_world + t.
"""
from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OIR_DIR = HERE.parent / "optional-image-assisted-registration-20261019"
for p in (str(OIR_DIR), str(HERE.parent / "affine-global-initialization-20261014")):
    if p not in sys.path:
        sys.path.insert(0, p)

import init_lib  # noqa: E402
from oir import metrics as mt  # noqa: E402
from oir import search as osearch  # noqa: E402
from oir.frames import Affine2D  # noqa: E402
from oir.pipeline import ACCEPT, RunResult, Side, cropped_icp, footprint, pose_change  # noqa: E402
from oir.search import Canvas, Layer, PoseSearch, SearchConfig  # noqa: E402

VARIANTS = {"X0": ("bbox", "all"), "X1": ("union", "all"), "X2": ("bbox", "crop"), "X3": ("union", "crop")}


class BBoxCanvas(Canvas):
    """Square canvas covering only the fixed nuclei bbox (+40 um), the A2 convention (no extension)."""

    def __init__(self, fixed_points, res_um, pad_um=40.0):
        lo = fixed_points.min(0) - pad_um
        hi = fixed_points.max(0) + pad_um
        L = float(max(hi - lo))
        self.n = int(np.ceil(L / res_um))
        self.res = res_um
        self.center = (lo + hi) / 2.0
        self.lo = self.center - self.n * res_um / 2


class AblationSearch(PoseSearch):
    """PoseSearch with a selectable canvas; the search body is PoseSearch.run with the canvas factory swapped."""

    def __init__(self, layers, fixed_extent, moving_extent, fixed_center, moving_center, cfg, canvas="union", fixed_points=None):
        super().__init__(layers, fixed_extent, moving_extent, fixed_center, moving_center, cfg)
        self.canvas_kind, self.fixed_points = canvas, fixed_points

    def run(self):
        if self.canvas_kind == "union":
            return super().run()
        fp = self.fixed_points
        orig = osearch.Canvas
        osearch.Canvas = lambda fe, me, fc, res, max_scale=1.1: BBoxCanvas(fp, res)   # noqa: E731
        try:
            return super().run()
        finally:
            osearch.Canvas = orig


def full_icp(cand, moving, fixed):
    """A2's production refine (icp_similarity -> affine_icp on all points)."""
    h = init_lib.Hypothesis(cand.chirality, cand.theta, cand.scale, np.asarray(cand.t, float), cand.score, "X")
    M, t, err, n = init_lib.refine(h, moving, fixed)
    return {"M": M, "t": t, "objective": err, "n_pairs": n, "n_moving_used": len(moving), "n_fixed_used": len(fixed)}


def refine(variant, cand, moving: Side, fixed: Side):
    """None when ICP cannot run (e.g. the similarity step leaves no points inside the other footprint)."""
    try:
        if VARIANTS[variant][1] == "crop":
            return cropped_icp(cand.A(), cand.t, cand.chirality, moving.points, fixed.points, moving.extent, fixed.extent)
        return full_icp(cand, moving.points, fixed.points)
    except ValueError:
        return None


def geometric_eval(T: Affine2D, moving: Side, fixed: Side):
    """Proximity of fixed nuclei inside the mapped moving footprint to mapped moving nuclei (fixed frame)."""
    mw = T.apply(moving.points)
    mov_fp = T.apply(footprint(moving.extent))
    fix_mask = mt.inside_poly(fixed.points, mov_fp, 10.0) & mt.inside_poly(fixed.points, footprint(fixed.extent), 0.0)
    m_in = mt.inside_poly(mw, footprint(fixed.extent), -10.0)
    if fix_mask.sum() < 5 or m_in.sum() < 5:
        return {"n_eval": int(fix_mask.sum()), "le_3um": 0.0, "null_le_3um": 0.0, "inlier_over_null": 0.0, "nn_median_um": np.nan,
                "geometric_pass": False}
    pr = mt.proximity(mw[m_in], fixed.points, fix_mask)
    nl = mt.mirror_null(mw[m_in], fixed.points, fix_mask)
    ion = pr["le_3um"] / max(nl["le_3um"], 1e-6)
    return {"n_eval": int(fix_mask.sum()), **pr, "null_le_3um": nl["le_3um"], "inlier_over_null": ion,
            "geometric_pass": bool(pr["le_3um"] >= ACCEPT["min_inlier"] and ion >= ACCEPT["min_inlier_over_null"])}


def overlap_stats(T: Affine2D, moving: Side, fixed: Side):
    """Shared imaged area (polygon intersection on a 4 um raster) and nuclear densities inside it."""
    mov_fp = T.apply(footprint(moving.extent))
    fe = np.asarray(fixed.extent)
    res = 4.0
    xs, ys = np.arange(res / 2, fe[0], res), np.arange(res / 2, fe[1], res)
    g = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    inside = mt.inside_poly(g, mov_fp, 0.0)
    area_mm2 = inside.sum() * res * res / 1e6
    f_in = mt.inside_poly(fixed.points, mov_fp, 0.0)
    m_in = mt.inside_poly(T.apply(moving.points), footprint(fixed.extent), 0.0)
    return {"overlap_area_mm2": float(area_mm2), "overlap_frac_of_moving": float(area_mm2 / (np.prod(moving.extent) / 1e6)),
            "n_fixed_in_overlap": int(f_in.sum()), "n_moving_in_overlap": int(m_in.sum()),
            "fixed_density_per_mm2": float(f_in.sum() / max(area_mm2, 1e-9)), "moving_density_per_mm2": float(m_in.sum() / max(area_mm2, 1e-9)),
            "overlap_polygon_fixed_um": mov_fp.tolist()}


def run_variant(sample, variant, fixed: Side, moving: Side, cfg: SearchConfig | None = None, n_candidates=6, verdict_only=False):
    """Run one ablation variant; returns the RunResult-like dict plus per-candidate ICP diagnostics (top-n)."""
    cfg = cfg or SearchConfig()
    canvas, icp_kind = VARIANTS[variant]
    tracemalloc.start(); t0 = time.perf_counter()
    layer = Layer("points", "points", fixed_points=fixed.points, moving_points=moving.points, res_um=cfg.res_fine)
    ps = AblationSearch([layer], fixed.extent, moving.extent, np.asarray(fixed.extent) / 2, np.asarray(moving.extent) / 2, cfg,
                        canvas=canvas, fixed_points=fixed.points)
    out = ps.run()
    search_rt = time.perf_counter() - t0
    fname, mname = f"{sample}:{fixed.key}", f"{sample}:{moving.key}"
    cands = []
    for rank, c in enumerate(out["fine"][:n_candidates]):
        Ts = Affine2D(c.A(), c.t, mname, fname)
        rec = {"rank": rank, "chirality": c.chirality, "theta_deg": c.theta, "scale": c.scale, "t_um": c.t.tolist(),
               "score_z": c.score, "ncc": c.layer_ncc.get("points"), "relative_gap_to_top": (out["fine"][0].score - c.score) / abs(out["fine"][0].score),
               "T_search": Ts.to_dict(), "search_eval": geometric_eval(Ts, moving, fixed)}
        if not verdict_only or rank == 0:
            ref = refine(variant, c, moving, fixed)
            if ref is None:
                rec["icp"] = None; rec["final_eval"] = rec["search_eval"]; rec["T_final"] = Ts.to_dict()
                rec["icp_note"] = "ICP not run: too few or no points inside the other footprint"
            else:
                Tf = Affine2D(ref["M"], ref["t"], mname, fname)
                rec["T_final"] = Tf.to_dict()
                rec["icp"] = {"pose_change": pose_change(Ts, Tf, footprint(moving.extent)), "n_moving_used": ref["n_moving_used"],
                              "n_fixed_used": ref["n_fixed_used"], "objective": ref["objective"]}
                rec["final_eval"] = geometric_eval(Tf, moving, fixed)
            rec["overlap"] = overlap_stats(Affine2D.from_dict(rec["T_final"]), moving, fixed)
        cands.append(rec)
    runtime = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()
    top = cands[0] if cands else None
    accepted = bool(top and out["verdict"] == "UNIQUE" and top["final_eval"]["geometric_pass"])
    return {"sample": sample, "variant": variant, "canvas": canvas, "icp": icp_kind, "fixed": fixed.key, "moving": moving.key,
            "verdict": out["verdict"], "margin": out["margin"], "n_poses": out["n_poses"], "search_runtime_s": search_rt,
            "runtime_s": runtime, "peak_mem_mb": peak, "accepted": accepted,
            "safety": ("UNIQUE / Accepted" if accepted else (out["verdict"] if out["verdict"] != "UNIQUE" else "FAIL (nuclear geometry)")),
            "candidates": cands, "coarse_top": [c.as_dict() for c in out["coarse_top"][:10]]}
