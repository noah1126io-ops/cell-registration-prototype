"""Post-hoc cause diagnosis of A2 failures (diagnostic only; nothing here is adopted or used to re-score A2).

Uses only the frozen A2 building blocks (`scale_candidates`, `_raster`, `_hull_mask`, `masked_ncc`) to evaluate the
A2 NCC surface at (a) the hypothesis A2 chose and (b) the TRUE chirality/rotation/scale, for the containment conditions
(C09/C10). A pure-outline control replaces the nuclei by uniform random points inside the same point-cloud hulls: if
NCC there still prefers the wrong scale, the preference comes from the field-of-view outline, not from tissue pattern.
Writes cache/diagnose_containment.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MPath
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "affine-global-initialization-20261014"))
import design as D  # noqa: E402
import indep_gen as G  # noqa: E402
import init_lib as L  # noqa: E402


def ncc_at(moving_px, fixed, chir, th, s, px=4.0, sigma=8.0, R=L.SEARCH_RADIUS_UM):
    """Identical arithmetic to A2's inner `score` (frozen), returning best NCC and its translation in the +-R window."""
    lo = fixed.min(0) - 40; hi = fixed.max(0) + 40
    shape = (int(np.ceil((hi[1] - lo[1]) / px)), int(np.ceil((hi[0] - lo[0]) / px)))
    f = L._raster(fixed, lo, shape, px, sigma); mf = L._hull_mask(fixed, lo, shape, px)
    f = f / max(f[mf > 0].mean(), 1e-12)
    q = L._transform_hyp(moving_px, chir, th, s); tc = fixed.mean(0) - q.mean(0); qq = q + tc
    m = L._raster(qq, lo, shape, px, sigma); mm = L._hull_mask(qq, lo, shape, px)
    m = m / max(m[mm > 0].mean(), 1e-12)
    ncc, _ = L.masked_ncc(f, mf, m, mm, min_overlap=0.3 * min(mf.sum(), mm.sum()))
    rmax = int(R / px); cy, cx = shape[0] - 1, shape[1] - 1
    win = ncc[cy - rmax: cy + rmax + 1, cx - rmax: cx + rmax + 1]
    k = np.unravel_index(np.argmax(np.where(np.isfinite(win), win, -9)), win.shape)
    dy, dx = (np.array(k) - rmax) * px
    return float(win[k]), tc + np.array([dx, dy])


def uniform_in_hull(points, rng, n):
    h = ConvexHull(points); path = MPath(points[h.vertices])
    lo, hi = points.min(0), points.max(0)
    out = []
    while sum(len(o) for o in out) < n:
        c = rng.uniform(lo, hi, (n, 2)); out.append(c[path.contains_points(c)])
    return np.concatenate(out)[:n]


def true_params(pair):
    chir = "reflect" if np.linalg.det(pair.M_true) < 0 else "proper"
    A = pair.M_true @ L.CHIRALITY[chir]          # = s R(theta)
    s = float(np.sqrt(abs(np.linalg.det(A)))); th = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
    return chir, th, s


rows = []
for i, (fam, cond, rep) in enumerate(D.main_cases()):
    if cond not in ("C09_overlap_mid_contained", "C10_overlap_low_smallfixed", "C01_base") or rep != 0:
        continue
    pair = G.make_pair(fam, cond, rep, D.CONDITIONS[cond])
    rec = next(m for m in json.loads((HERE / f"cache/main_{i:03d}.json").read_text())["methods"] if m["method_key"] == "A2")
    chosen = rec["basins"][0][3]
    chir_t, th_t, s_t = true_params(pair)
    sc = L.scale_candidates(pair.fixed, pair.moving_px)
    n_true, t_true_ncc = ncc_at(pair.moving_px, pair.fixed, chir_t, th_t, s_t)
    n_chosen, _ = ncc_at(pair.moving_px, pair.fixed, chosen["chirality"], chosen["theta"], chosen["scale"])
    # where does the true-parameter NCC peak put the moving centre vs the truth? (translation check)
    q = L._transform_hyp(pair.moving_px, chir_t, th_t, s_t)
    A_true_t = (pair.M_true @ np.array([pair.width / 2, pair.height / 2]) + pair.t_true)
    est_centre = L._transform_hyp(np.array([[pair.width / 2, pair.height / 2]]), chir_t, th_t, s_t)[0] + t_true_ncc
    rng = np.random.default_rng(7)
    uf = uniform_in_hull(pair.fixed, rng, len(pair.fixed)); um = uniform_in_hull(pair.moving_px, rng, len(pair.moving_px))
    u_true, _ = ncc_at(um, uf, chir_t, th_t, s_t)
    u_chosen, _ = ncc_at(um, uf, chosen["chirality"], chosen["theta"], chosen["scale"])
    rows.append({"case": pair.name, "true_scale": s_t, "scale_candidates": json.dumps([round(x, 4) for x in sc]),
                 "nn_scale_added": len(sc) > 1, "chosen_scale": chosen["scale"], "chosen_theta": chosen["theta"], "true_theta": th_t,
                 "ncc_at_true_params": n_true, "ncc_at_chosen": n_chosen, "true_params_translation_error_um": float(np.linalg.norm(est_centre - A_true_t)),
                 "outline_only_ncc_true_params": u_true, "outline_only_ncc_chosen": u_chosen,
                 "fov_area_ratio_fixed_over_moving": pair.params["overlap_area_um2"] / G.polygon_area(pair.moving_fov_world)})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(HERE / "cache/diagnose_containment.csv", index=False)
