"""Score landscape of the (unchanged) registration_core pose search over an extended rotation x scale grid.

For every pose and every feature combination: the top peaks (as PoseSearch._peaks) and the best score inside a
40 um window around the reference location (where the manual reference puts the moving centre).  Uses the engine's
own Canvas / masked NCC / robust z / zsum combination; only the layer list and the pose grid differ.

usage: landscape.py SAMPLE OUT.csv [--ref TF_ID | --ref-json M,t file] [--res 8]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.ndimage import zoom

from common import features, probe, side, store_transform
from src.registration_core.features import coarsen
from src.registration_core.pipeline import _fov_mask
from src.registration_core.search import Canvas, Layer, PoseSearch, SearchConfig, pose_matrix
from variants import variant_maps

COMBOS = {
    "base(tissue+nuclear)": ["img_tissue", "img_nuclear"],
    "tissue": ["img_tissue"], "nuclear": ["img_nuclear"], "points": ["points"], "edt": ["img_edt"],
    "occ": ["occ"], "occ12": ["occ12"], "cleft": ["cleft"], "occ+cleft": ["occ", "cleft"],
    "occ+nuclear": ["occ", "img_nuclear"], "occ+cleft+nuclear": ["occ", "cleft", "img_nuclear"], "occ+points": ["occ", "points"],
}
RES_FINE = 4.0


def build(sample_id, layers_needed):
    F, Mv = side(sample_id, "HE"), side(sample_id, "IF")
    ff, mf = coarsen(F.feats, RES_FINE), coarsen(Mv.feats, RES_FINE)
    fmask, mmask = _fov_mask(ff), _fov_mask(mf)
    L = []
    for n in layers_needed:
        if n == "points":
            L.append(Layer("points", "points", fixed_points=F.points, moving_points=Mv.points, res_um=RES_FINE))
        elif n.startswith("img_"):
            k = n[4:]
            L.append(Layer(n, "image", fixed_map=ff.maps[k], fixed_mask=fmask, moving_map=mf.maps[k], moving_mask=mmask, res_um=RES_FINE))
        else:
            vf, vm = variant_maps(sample_id, "HE")[n], variant_maps(sample_id, "IF")[n]
            off = 1.0 if n == "cleft" else 0.0          # band-passed map: positive offset (NCC is offset-invariant; _normalise divides by the mean)
            fz = zoom(vf, 2.0 / RES_FINE, order=1, prefilter=False) + off
            mz = zoom(vm, 2.0 / RES_FINE, order=1, prefilter=False) + off
            L.append(Layer(n, "image", fixed_map=fz, fixed_mask=np.ones(fz.shape, np.float32), moving_map=mz,
                           moving_mask=np.ones(mz.shape, np.float32), res_um=RES_FINE))
    return F, Mv, L


def work(args):
    sample_id, scale, thetas, ref, res, combos = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    need = sorted({n for c in combos for n in COMBOS[c]})
    F, Mv, layers = build(sample_id, need)
    cfg = SearchConfig(res_coarse=res)
    ps = PoseSearch(layers, F.extent, Mv.extent, np.asarray(F.extent) / 2, np.asarray(Mv.extent) / 2, cfg)
    sub = {c: PoseSearch([l for l in layers if l.name in COMBOS[c]], F.extent, Mv.extent, ps.fc, ps.mc, cfg) for c in combos}
    cv = Canvas(F.extent, Mv.extent, ps.fc, res, 1.25)
    cache, rows = {}, []
    Mr, tr = ref
    target = Mr @ ps.mc + tr
    for th in thetas:
        maps, t0 = ps._maps(cv, "proper", th, scale, cache)
        if maps is None:
            continue
        A = pose_matrix("proper", th, scale)
        t_ref = target - A @ ps.mc
        for c, sp in sub.items():
            m = {n: maps[n] for n in COMBOS[c]}
            peaks = sp._peaks(cv, m, t0, "proper", th, scale, 3)
            near = sp._peaks(cv, m, t0, "proper", th, scale, 1, window=(t_ref, 40.0))
            for kind, lst in (("peak", peaks), ("ref", near)):
                for p in lst:
                    rows.append({"combo": c, "kind": kind, "theta": th, "scale": scale, "tx": p.t[0], "ty": p.t[1], "score": p.score,
                                 "layer_z": json.dumps({k: round(v, 3) for k, v in p.layer_z.items()})})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample"); ap.add_argument("out")
    ap.add_argument("--ref", default=None, help="research-store transform id")
    ap.add_argument("--ref-json", default=None, help='json file {"M":..,"t":..}')
    ap.add_argument("--res", type=float, default=8.0)
    ap.add_argument("--scales", default="0.90,0.95,1.00,1.05,1.10,1.15,1.20")
    ap.add_argument("--rot-step", type=float, default=4.0)
    ap.add_argument("--combos", default=",".join(COMBOS))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.ref_json:
        d = json.load(open(a.ref_json)); ref = (np.array(d["M"]), np.array(d["t"]))
    else:
        M, t, _ = store_transform(a.ref) if a.ref else store_transform(); ref = (M, t)
    scales = [float(s) for s in a.scales.split(",")]
    thetas = np.arange(0, 360, a.rot_step)
    combos = a.combos.split(",")
    # warm caches once (feature npz) before forking
    for k in ("HE", "IF"):
        features(a.sample, k); variant_maps(a.sample, k)
    chunks = [(a.sample, s, thetas[i::2], ref, a.res, combos) for s in scales for i in range(2)]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(work, chunks):
            rows += r
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("rows", len(rows), "->", a.out)


if __name__ == "__main__":
    main()
