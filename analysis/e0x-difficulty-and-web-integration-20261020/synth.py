"""Point-only synthetic difficulty sweep for E0x (protocol.md, fixed before running).

  python -B synth.py --list               # print the task table
  python -B synth.py --task 7             # run task 7 (a chunk of cases) -> cache/syn_<case>.json
  python -B synth.py --smoke              # dev seed 0, two extreme conditions (not reported)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e0xlab  # noqa: E402
from oir import metrics as mt  # noqa: E402
from oir.frames import Affine2D, json_safe  # noqa: E402
from oir.pipeline import Side, footprint  # noqa: E402
from oir.search import rot  # noqa: E402

CACHE = HERE / "cache"
WF = 2100.0
BASE = {"ratio": 0.5, "overlap": 1.0, "nonshared": 0.3, "missed": 0.1, "rotation": 30.0, "offset": 0.2, "scale": 1.0,
        "density": 2000.0, "deform": 0.0, "repeat": 0.0, "absent": False}
LEVELS = {"ratio": [0.25, 0.5, 0.75, 1.0], "overlap": [1.0, 0.75, 0.5, 0.3], "nonshared": [0.0, 0.3, 0.5, 0.7, 0.85],
          "missed": [0.0, 0.1, 0.2, 0.4, 0.6], "rotation": [0.0, 30.0, 45.0, 90.0, 180.0], "offset": [0.0, 0.2, 0.4],
          "scale": [0.9, 0.95, 1.0, 1.05, 1.1], "density": [500.0, 1000.0, 2000.0, 4000.0], "deform": [0.0, 2.0, 5.0, 10.0],
          "repeat": [0.0, 0.5, 0.8, 1.0]}
EVAL_SEEDS = (1001, 1002, 1003)
X0_FACTORS = ("ratio", "overlap", "scale")
XS_FACTORS = ("ratio",)


def conditions():
    """(cond_id, factor, params) with the base condition listed once."""
    out = [("base", "base", dict(BASE))]
    for f, lv in LEVELS.items():
        for v in lv:
            if v == BASE[f]:
                continue
            out.append((f"{f}={v:g}", f, {**BASE, f: v}))
    out.append(("absent", "absent", {**BASE, "absent": True}))
    for r in (0.25, 0.5, 1.0):
        for ns in (0.0, 0.5, 0.85):
            cid = f"grid ratio={r:g} ns={ns:g}"
            out.append((cid, "grid", {**BASE, "ratio": r, "nonshared": ns}))
    return out


def cases():
    rows = []
    for cid, factor, p in conditions():
        methods = ["X3_L"]
        if factor in X0_FACTORS or (factor == "base"):
            methods.append("X0_L")
        if factor in XS_FACTORS or factor == "base":
            methods.append("X3_S")
        for seed in EVAL_SEEDS:
            rows.append({"case": f"{len(rows):04d}", "cond": cid, "factor": factor, "params": p, "seed": seed, "methods": methods})
    return rows


def tasks(n_tasks=int(__import__("os").environ.get("SYN_NTASKS", "60"))):
    cs = cases()
    return [cs[i::n_tasks] for i in range(n_tasks)]


# ----------------------------------------------------------------------------- generator
def smooth_field(rng, shape, sigma_px):
    f = gaussian_filter(rng.standard_normal(shape), sigma_px)
    return (f - f.mean()) / (f.std() + 1e-12)


def tissue_field(rng, lo, size, res, repeat, corr_um=60.0, patch_um=320.0):
    n = int(np.ceil(size / res))
    f = smooth_field(rng, (n, n), corr_um / res)
    if repeat > 0:
        k = int(round(patch_um / res))
        patch = smooth_field(rng, (k * 3, k * 3), corr_um / res)[k:2 * k, k:2 * k]
        per = np.tile(patch, (n // k + 1, n // k + 1))[:n, :n]
        per = (per - per.mean()) / (per.std() + 1e-12)
        f = (1 - repeat) * f + repeat * per
    thr = np.quantile(f, 0.5)
    return f > thr, lo, res


def sample_nuclei(rng, tissue, density, hard_core=4.5, repeat=0.0, patch_um=320.0):
    mask, lo, res = tissue
    n = mask.shape[0]
    area = (n * res) ** 2 / 1e6               # uniform over the whole domain; the tissue mask keeps density x tissue area
    k = rng.poisson(density * area)
    p = rng.uniform(0, n * res, (k, 2))
    if repeat > 0:   # repeated structure also repeats the nuclear arrangement inside the periodic part
        base = rng.uniform(0, patch_um, (int(k * (patch_um / (n * res)) ** 2) + 1, 2))
        reps = np.concatenate([base + np.array([i, j]) * patch_um for i in range(int(n * res // patch_um) + 1)
                               for j in range(int(n * res // patch_um) + 1)])
        take = rng.random(len(reps)) < repeat
        keep = rng.random(len(p)) >= repeat
        p = np.concatenate([p[keep], reps[take]])
    ij = np.clip((p / res).astype(int), 0, n - 1)
    p = p[mask[ij[:, 1], ij[:, 0]]]
    tree = cKDTree(p)
    drop = set()
    for a, b in tree.query_pairs(hard_core):
        if a not in drop and b not in drop:
            drop.add(b)
    keep = np.ones(len(p), bool); keep[list(drop)] = False
    return p[keep] + lo


def inside_frac(A, t, wm, wf, n=41):
    g = np.stack(np.meshgrid(np.linspace(0, wm, n), np.linspace(0, wm, n)), -1).reshape(-1, 2)
    q = g @ A.T + t
    return float(np.mean((q[:, 0] >= 0) & (q[:, 0] <= wf) & (q[:, 1] >= 0) & (q[:, 1] <= wf)))


@dataclass
class Case:
    fixed: np.ndarray
    moving: np.ndarray
    A: np.ndarray
    t: np.ndarray
    wm: float
    shared_moving: np.ndarray   # indices into moving
    shared_fixed: np.ndarray    # matching indices into fixed (-1 = partner not detected)
    overlap_achieved: float


def make_case(p, seed):
    rng = np.random.default_rng(seed)
    wm = p["ratio"] * WF
    s, th = p["scale"], p["rotation"]
    A = s * rot(th)
    mc = np.array([wm, wm]) / 2
    phi = rng.uniform(0, 2 * np.pi)
    u = np.array([np.cos(phi), np.sin(phi)])
    c = np.array([WF, WF]) / 2 + p["offset"] * WF * u
    t = c - A @ mc
    if p["overlap"] >= 1.0:          # pull towards the centre until fully inside (if possible)
        for _ in range(60):
            if inside_frac(A, t, wm, WF) >= 0.999:
                break
            c = c + (np.array([WF, WF]) / 2 - c) * 0.1
            t = c - A @ mc
    else:                            # push outward along u until the target overlap is reached
        lo_d, hi_d = 0.0, 2.0 * WF
        for _ in range(40):
            d = (lo_d + hi_d) / 2
            tt = (np.array([WF, WF]) / 2 + d * u) - A @ mc
            if inside_frac(A, tt, wm, WF) > p["overlap"]:
                lo_d = d
            else:
                hi_d = d
        t = (np.array([WF, WF]) / 2 + lo_d * u) - A @ mc
    ov = inside_frac(A, t, wm, WF)
    pad = 1.6 * wm + 200
    lo = np.array([-pad, -pad]); size = WF + 2 * pad
    tis = tissue_field(rng, lo, size, 4.0, p["repeat"])
    latent = sample_nuclei(rng, tis, p["density"], repeat=p["repeat"])
    fixed_all = latent[(latent[:, 0] >= 0) & (latent[:, 0] <= WF) & (latent[:, 1] >= 0) & (latent[:, 1] <= WF)]
    Ai = np.linalg.inv(A)
    if p["absent"]:
        src = sample_nuclei(np.random.default_rng(seed + 77777), tissue_field(np.random.default_rng(seed + 88888), lo, size, 4.0, p["repeat"]),
                            p["density"], repeat=p["repeat"])
    else:
        src = latent
    mw = (src - t) @ Ai.T
    inwin = (mw[:, 0] >= 0) & (mw[:, 0] <= wm) & (mw[:, 1] >= 0) & (mw[:, 1] <= wm)
    mw, src_in = mw[inwin], src[inwin]
    # non-shared replacement: same tissue, independent nuclear process (a different section's cells)
    n = len(mw)
    rep = rng.random(n) < p["nonshared"]
    other = sample_nuclei(rng, tis, p["density"])
    ow = (other - t) @ Ai.T
    ow = ow[(ow[:, 0] >= 0) & (ow[:, 0] <= wm) & (ow[:, 1] >= 0) & (ow[:, 1] <= wm)]
    n_rep = int(rep.sum())
    foreign = ow[rng.permutation(len(ow))[:n_rep]] if len(ow) else np.zeros((0, 2))
    shared_idx = np.where(~rep)[0]
    moving = np.concatenate([mw[~rep], foreign])
    shared_src = src_in[~rep]
    # local deformation (moving side), jitter
    if p["deform"] > 0:
        g = 8.0; ng = int(np.ceil(wm / g)) + 2
        dx, dy = smooth_field(rng, (ng, ng), 150 / g), smooth_field(rng, (ng, ng), 150 / g)
        nrm = p["deform"] / np.sqrt(np.mean(dx ** 2 + dy ** 2))
        coords = [moving[:, 1] / g, moving[:, 0] / g]
        moving = moving + nrm * np.c_[map_coordinates(dx, coords, order=1), map_coordinates(dy, coords, order=1)]
    moving = moving + rng.normal(0, 0.8, moving.shape)
    # missed detections, independently on both sides
    keep_m = rng.random(len(moving)) >= p["missed"]
    keep_f = rng.random(len(fixed_all)) >= p["missed"]
    fixed = fixed_all[keep_f]
    n_sh = len(shared_idx)
    sh_mask = np.zeros(len(moving), bool); sh_mask[:n_sh] = True
    moving = moving[keep_m]
    sh_mask = sh_mask[keep_m]
    shared_src = shared_src[keep_m[:n_sh]]
    if p["absent"]:
        shared_moving = np.zeros(0, int); shared_fixed = np.zeros(0, int)
    else:
        tree = cKDTree(fixed)
        d, j = tree.query(shared_src)
        shared_moving = np.where(sh_mask)[0]
        shared_fixed = np.where(d < 1e-6, j, -1)
    return Case(fixed, moving, A, t, wm, shared_moving, shared_fixed, ov)


# ----------------------------------------------------------------------------- evaluation
def grid_error(T_est: Affine2D, A, t, wm, moving_is_window=True):
    """Median distance (fixed frame) between estimated and true maps on the window part inside the fixed FOV."""
    g = np.stack(np.meshgrid(np.linspace(0, wm, 11), np.linspace(0, wm, 11)), -1).reshape(-1, 2)
    q = g @ A.T + t
    ins = (q[:, 0] >= 0) & (q[:, 0] <= WF) & (q[:, 1] >= 0) & (q[:, 1] <= WF)
    g, q = (g[ins], q[ins]) if ins.sum() >= 4 else (g, q)
    return float(np.median(np.linalg.norm(T_est.apply(g) - q, axis=1)))


def run_case(row, methods=None):
    p, seed = row["params"], row["seed"]
    C = make_case(p, seed)
    L = Side("L", C.fixed, (WF, WF), None)       # large side = the 2100 um field
    S = Side("S", C.moving, (C.wm, C.wm), None)
    T_true = Affine2D(C.A, C.t, "S", "L")
    out = {"case": row["case"], "cond": row["cond"], "factor": row["factor"], "params": p, "seed": seed,
           "n_fixed": len(C.fixed), "n_moving": len(C.moving), "overlap_achieved": C.overlap_achieved, "results": {}}
    for m in methods or row["methods"]:
        variant, side = m.split("_")
        fixed, moving = (L, S) if side == "L" else (S, L)
        t0 = time.perf_counter()
        rec = e0xlab.run_variant("syn", variant, fixed, moving, n_candidates=3, verdict_only=True)
        top = rec["candidates"][0] if rec["candidates"] else None
        if top is None:
            out["results"][m] = {"verdict": rec["verdict"], "accepted": False}
            continue
        Tf = Affine2D.from_dict(top["T_final"]); Ts = Affine2D.from_dict(top["T_search"])
        if side == "S":   # result maps L -> S; bring to S -> L for the error
            Tf, Ts = Tf.inverse(), Ts.inverse()
        err = grid_error(Tf, C.A, C.t, C.wm); serr = grid_error(Ts, C.A, C.t, C.wm)
        tid = np.nan
        if len(C.shared_moving):
            mwp = Tf.apply(C.moving[C.shared_moving])
            d, j = cKDTree(C.fixed).query(mwp)
            ok = C.shared_fixed >= 0
            tid = float(np.mean((j[ok] == C.shared_fixed[ok]) & (d[ok] <= 3.0))) if ok.any() else np.nan
        fe = top["final_eval"]
        absent = bool(p["absent"])
        acc = bool(rec["accepted"])
        out["results"][m] = {"verdict": rec["verdict"], "margin": rec["margin"], "accepted": acc, "safety": rec["safety"],
                             "err_um": err, "search_err_um": serr, "region_found": (not absent) and serr <= 20.0,
                             "reg_success": (not absent) and err <= 3.0, "correct_accept": acc and (not absent) and err <= 3.0,
                             "false_accept": acc and (absent or err > 20.0), "le_3um": fe.get("le_3um"), "null_le_3um": fe.get("null_le_3um"),
                             "inlier_over_null": fe.get("inlier_over_null"), "true_id": tid, "runtime_s": time.perf_counter() - t0,
                             "peak_mem_mb": rec["peak_mem_mb"], "est_scale": Tf.params()["scale"], "est_rot": Tf.params()["rotation_deg"]}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int); ap.add_argument("--list", action="store_true"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.list:
        cs = cases()
        print(len(cs), "cases;", sum(len(c["methods"]) for c in cs), "runs;", len(tasks()), "tasks")
    elif a.smoke:
        for cid in ("base", "ratio=1"):
            row = next(c for c in cases() if c["cond"] == cid)
            row = {**row, "seed": 0}
            C = make_case(row["params"], 0)
            print(cid, "n_fixed", len(C.fixed), "n_moving", len(C.moving), "overlap", round(C.overlap_achieved, 3), "shared", len(C.shared_moving), flush=True)
            r = run_case(row, methods=["X3_L"])
            print(json.dumps(json_safe(r["results"]), indent=1), flush=True)
    else:
        for row in tasks()[a.task]:
            path = CACHE / f"syn_{row['case']}.json"
            if path.exists():
                continue
            r = run_case(row)
            path.write_text(json.dumps(json_safe(r), indent=1))
            print(row["case"], row["cond"], row["seed"], {k: (v.get("verdict"), round(v.get("err_um", np.nan), 2)) for k, v in r["results"].items()}, flush=True)
