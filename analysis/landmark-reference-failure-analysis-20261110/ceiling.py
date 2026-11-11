"""Q1: can nuclei correspond at all?  Best attainable <=3 um proximity in a local neighbourhood of the reference pose,
compared with the same local maximisation around 20 far-away (wrong) poses (selection-matched null).

Output: ceiling.csv, ceiling.json
"""
import json

import numpy as np
from scipy.spatial import cKDTree

from common2 import C27, reference
from src.registration_core import metrics as mt
from src.registration_core.pipeline import footprint

rng = np.random.default_rng(0)
SH = np.arange(-24, 25, 3.0)                 # um
ROT = np.arange(-3, 3.01, 1.0)               # deg
SC = (0.97, 1.0, 1.03)


def le3(M, t, mov, fix):
    mw = mov.points @ M.T + t
    fmask = mt.inside_poly(fix.points, mw_fp(M, t, mov), 10.0)
    if fmask.sum() < 50:
        return np.nan, 0
    d = cKDTree(mw).query(fix.points[fmask])[0]
    return float(np.mean(d <= 3.0)), int(fmask.sum())


def mw_fp(M, t, mov):
    return footprint(mov.extent) @ M.T + t


def local_max(M0, t0, mov, fix):
    c = np.asarray(mov.extent) / 2
    best = (-1, None)
    for r in ROT:
        a = np.radians(r)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        for s in SC:
            M = s * R @ M0
            tc = M0 @ c + t0 - M @ c                  # rotate/scale about the moving-field centre
            for dx in SH:
                for dy in SH:
                    v, n = le3(M, tc + (dx, dy), mov, fix)
                    if v > best[0]:
                        best = (v, (float(r), s, float(dx), float(dy)), n)
    return best


rows = []
for sample in ("031-6_2", "031-8_6"):
    fix, mov = C27.side(sample, "HE", False), C27.side(sample, "IF", False)
    M0, t0 = reference(sample)
    at_ref, n = le3(M0, t0, mov, fix)
    mw = mov.points @ M0.T + t0
    fmask = mt.inside_poly(fix.points, mw_fp(M0, t0, mov), 10.0)
    null_mirror = mt.mirror_null(mw, fix.points, fmask)["le_3um"]
    best, arg, _ = local_max(M0, t0, mov, fix)
    # wrong poses: random rotation, translation keeping >= 50% of the IF field inside HE
    wrong = []
    fe = np.asarray(fix.extent); me = np.asarray(mov.extent)
    while len(wrong) < 20:
        a = rng.uniform(0, 2 * np.pi); R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        Mw = R * np.sqrt(abs(np.linalg.det(M0)))
        ctr = rng.uniform(me / 2, fe - me / 2)
        tw = ctr - Mw @ (me / 2)
        if np.linalg.norm(ctr - (M0 @ (me / 2) + t0)) < 300:
            continue
        b, _, nn = local_max(Mw, tw, mov, fix)
        if nn >= 200:
            wrong.append(b)
    rows.append({"sample": sample, "n_fixed_eval": n, "le3_at_reference": at_ref, "mirror_null": null_mirror,
                 "local_max_le3_near_reference": best, "argmax(rot,scale,dx,dy)": arg,
                 "local_max_le3_wrong_poses_median": float(np.median(wrong)), "wrong_p95": float(np.percentile(wrong, 95)),
                 "wrong_max": float(np.max(wrong)), "accept_threshold": 0.25})
    print(rows[-1], flush=True)
json.dump(rows, open("ceiling.json", "w"), indent=1)
import csv
with open("ceiling.csv", "w", newline="") as f:
    w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
