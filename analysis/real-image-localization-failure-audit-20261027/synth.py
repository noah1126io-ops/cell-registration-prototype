"""Known-transform validation of the pose search (engine Canvas / masked NCC / z / zsum / distinct rule; coarse stage).

Test 1  same image   : moving = a 1026 um window of the HE feature maps, resampled under a known similarity (GT).
Test 2  partial      : as Test 1, but the window hangs over the HE field edge (overlap 75 % / 50 %).
Test 3  ablation     : every condition is scored for every feature combination (base, tissue, nuclear, occ, cleft, ...).
Test X  cross-modal  : moving = the real IF maps resampled under an extra known transform E (and cropped);
                       GT = reference o E.  031-6_2 reference = the accepted 10-22 transform (independent of occ/cleft);
                       031-8_6 reference = the manual MANUAL_UNVERIFIED transform (approximate; reported separately).
Success = top-1 distinct candidate within 40 um (probe median) of GT.  Output: synthetic_validation.csv
"""
from __future__ import annotations

import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.ndimage import affine_transform

from common import features, probe, probe_dist, side, store_transform
from src.registration_core.features import coarsen
from src.registration_core.search import Canvas, Layer, PoseSearch, SearchConfig, pose_matrix, rot

RES = 4.0
MAPS = ("tissue", "nuclear", "occ", "cleft")
COMBOS = {"base(tissue+nuclear)": ["img_tissue", "img_nuclear"], "tissue": ["img_tissue"], "nuclear": ["img_nuclear"],
          "points": ["points"], "occ": ["img_occ"], "cleft": ["img_cleft"], "occ+cleft": ["img_occ", "img_cleft"]}


def resample(src, src_res, A, t, out_shape, out_res, order=1, cval=0.0):
    """out(p) = src(A p + t), p = moving world (um) on an out_res grid; src on a src_res grid (world = (col,row)*res)."""
    P = np.array([[0, 1], [1, 0]], float)
    mat = P @ A @ P * (out_res / src_res)
    off = (P @ t) / src_res
    return affine_transform(src.astype(np.float32), mat, offset=off, output_shape=out_shape, order=order, cval=cval)


def maps_of(sample, key):
    fm = coarsen(features(sample, key), RES)
    return {k: fm.maps[k] for k in MAPS}, np.ones(fm.maps["tissue"].shape, np.float32)


def condition(c):
    """Build fixed / moving layers for one condition; returns (F_ext, M_ext, layers, A_gt, t_gt)."""
    sample = c["sample"]
    F = side(sample, "HE"); fmaps, fmask = maps_of(sample, "HE")
    ext = (1026.0, 1026.0) if c["test"] != "X" else tuple(c["crop"])
    n = int(round(ext[0] / RES))
    if c["test"] in ("1", "2"):
        A = c["scale"] * rot(c["theta"]); t = np.asarray(c["t"], float)
        mmaps = {k: resample(v, RES, A, t, (n, n), RES) for k, v in fmaps.items()}
        mmask = resample(fmask, RES, A, t, (n, n), RES, order=0)
        q = np.linalg.solve(A, (F.points - t).T).T
        mpts = q[(q[:, 0] >= 0) & (q[:, 0] < ext[0]) & (q[:, 1] >= 0) & (q[:, 1] < ext[1])]
        A_gt, t_gt = A, t
    else:   # cross-modal: virtual IF = real IF under E (virtual -> real IF world: p_if = E p + e)
        Mi = side(sample, "IF"); imaps, imask = maps_of(sample, "IF")
        E = c["scale"] * rot(c["theta"]); e = np.asarray(c["t"], float)
        mmaps = {k: resample(v, RES, E, e, (n, n), RES) for k, v in imaps.items()}
        mmask = resample(imask, RES, E, e, (n, n), RES, order=0)
        q = np.linalg.solve(E, (Mi.points - e).T).T
        mpts = q[(q[:, 0] >= 0) & (q[:, 0] < ext[0]) & (q[:, 1] >= 0) & (q[:, 1] < ext[1])]
        Mr, tr = c["ref"]
        A_gt, t_gt = Mr @ E, Mr @ e + tr
    layers = [Layer(f"img_{k}", "image", fixed_map=fmaps[k], fixed_mask=fmask, moving_map=mmaps[k] + (0.0 if k != "cleft" else 0.0),
                    moving_mask=(mmask > 0.5).astype(np.float32), res_um=RES) for k in MAPS]
    layers.append(Layer("points", "points", fixed_points=F.points, moving_points=mpts, res_um=RES))
    return F.extent, ext, layers, A_gt, t_gt, float((mmask > 0.5).mean()), len(mpts)


def work(c):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    fe, me, layers, A_gt, t_gt, ov, npts = condition(c)
    cfg = SearchConfig()
    ps = PoseSearch(layers, fe, me, np.asarray(fe) / 2, np.asarray(me) / 2, cfg)
    sub = {k: PoseSearch([l for l in layers if l.name in v], fe, me, ps.fc, ps.mc, cfg) for k, v in COMBOS.items()}
    cv = Canvas(fe, me, ps.fc, cfg.res_coarse, 1.15)
    cache, peaks = {}, {k: [] for k in COMBOS}
    for th in np.arange(0, 360, cfg.rot_step):
        for s in cfg.scales:
            maps, t0 = ps._maps(cv, "proper", th, s, cache)
            if maps is None:
                continue
            for k, sp in sub.items():
                peaks[k] += sp._peaks(cv, {n: maps[n] for n in COMBOS[k]}, t0, "proper", th, s, cfg.peaks_per_pose)
    pr = probe(me)
    rows = []
    for k, sp in sub.items():
        d = sp.distinct(peaks[k])
        errs = [probe_dist(x.A(), x.t, A_gt, t_gt, pr) for x in d]
        rank = next((i + 1 for i, e in enumerate(errs) if e < 40.0), None)
        margin = (d[0].score - d[1].score) / abs(d[0].score) if len(d) > 1 and d[0].score else float("inf")
        rows.append({**{kk: vv for kk, vv in c.items() if kk not in ("ref",)}, "overlap_frac_of_moving": round(ov, 3), "n_moving_pts": npts,
                     "combo": k, "top1_err_um": round(errs[0], 1) if errs else None, "success_top1": bool(errs and errs[0] < 40.0),
                     "gt_rank": rank, "margin": round(margin, 3), "verdict_rule": "UNIQUE" if margin >= cfg.ambiguous_margin else "AMBIGUOUS",
                     "top1_theta": d[0].theta if d else None, "top1_scale": d[0].scale if d else None})
    return rows


def conditions():
    C = []
    he_centre = {"031-8_6": np.array([1058.0, 1075.0]), "031-6_2": np.array([1000.0, 1000.0])}   # 6_2: tissue blob centre
    for sample in ("031-8_6", "031-6_2"):
        for th, s in ((37.0, 1.0), (200.0, 0.97), (110.0, 1.04)):
            A = s * rot(th); t = he_centre[sample] - A @ np.array([513.0, 513.0])
            C.append({"test": "1", "sample": sample, "theta": th, "scale": s, "t": t.tolist(), "crop": [1026, 1026]})
        for ov in (0.75, 0.5):     # shift the window so (1 - ov) of it hangs over the left HE edge
            A = rot(37.0); fe_c = np.array([513.0 * (2 * ov - 1), he_centre[sample][1]]) if sample == "031-8_6" else np.array([1026.0 * ov - 513.0 + 0.0, 1000.0])
            t = fe_c - A @ np.array([513.0, 513.0])
            C.append({"test": "2", "sample": sample, "theta": 37.0, "scale": 1.0, "t": t.tolist(), "crop": [1026, 1026], "target_overlap": ov})
    M8, t8, _ = store_transform()
    d = json.load(open("cache/ref_6_2_accepted.json")); M6, t6 = np.array(d["M"]), np.array(d["t"])
    for sample, ref in (("031-6_2", (M6, t6)), ("031-8_6", (M8, t8))):
        for th, s, crop in ((30.0, 1.0, 1026), (-75.0, 0.97, 800), (150.0, 1.04, 800)):
            E = s * rot(th); cc = np.array([513.0, 513.0])
            e = cc - E @ np.array([crop / 2, crop / 2])       # virtual window centred on the IF centre
            C.append({"test": "X", "sample": sample, "theta": th, "scale": s, "t": e.tolist(), "crop": [crop, crop], "ref": ref,
                      "ref_kind": "accepted 10-22" if sample == "031-6_2" else "manual (MANUAL_UNVERIFIED)"})
    return C


if __name__ == "__main__":
    C = conditions()
    for s in ("031-8_6", "031-6_2"):
        for k in ("HE", "IF"):
            features(s, k)
    rows = []
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as ex:
        for r in ex.map(work, C):
            rows += r
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open("synthetic_validation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("rows", len(rows))
