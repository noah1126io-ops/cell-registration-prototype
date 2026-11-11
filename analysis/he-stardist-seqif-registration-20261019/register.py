"""HE (StarDist nuclei) -> seqIF (GeoJSON nuclei) affine registration, analysis-only (2026-10-19).

Method = A2 of affine-global-initialization-20261014 (point-density masked NCC init + production
icp_similarity -> affine_icp), with two changes required by this data:
  * both point sets are in physical um (OME PhysicalSize), so the scale search is centred on 1.0;
  * HE field (~2.1 mm) strictly contains the seqIF field (~1 mm). The NCC grid therefore covers both
    sets, and ICP only uses HE nuclei inside the transformed seqIF hull (+margin), re-cropped each round.
    (2026-10-15 study: un-cropped ICP collapses in scale under a large field mismatch.)
The existing *_seg_he.geojson placement (pure scale+translation) is used only as a comparison, never as input.

world (seqIF level-0 um, image origin top-left) = M @ he_um + t
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from matplotlib.path import Path as MPath
from scipy.ndimage import maximum_filter
from scipy.spatial import ConvexHull, cKDTree

ROOT = Path("/mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis/affine-global-initialization-20261014"))
from src.pointset_registration import SimilarityTransform, affine_icp, icp_similarity  # noqa: E402
from init_lib import CHIRALITY, _hull_mask, _raster, apply_eff, masked_ncc, rot  # noqa: E402

PX, SIGMA = 4.0, 8.0               # NCC raster (same as A2)
COARSE_ROT, FINE_HALF, FINE_STEP = 4.0, 3.0, 1.0
SCALES_COARSE = (1.0,)
SCALES_FINE = (0.97, 1.0, 1.03)
CROP_MARGIN_UM = 20.0
N_REFINE = 6


def ome_px(path):
    with tifffile.TiffFile(path) as t:
        return float(re.search(r'PhysicalSizeX="([^"]+)"', t.ome_metadata).group(1))


def load_geojson(path):
    d = json.loads(Path(path).read_text())
    rows = []
    for f in d["features"]:
        p = f["properties"]
        rows.append((p["segment_id"], p["centroid_x"], p["centroid_y"], f["geometry"]["coordinates"][0]))
    return rows


# ----------------------------------------------------------------------------- NCC search on a union grid
def ncc_search(moving, fixed, chiralities=("proper", "reflect")):
    fc = fixed.mean(0)
    ext = np.ptp(moving, 0).max() * 1.05 + np.ptp(fixed, 0).max()
    lo = fc - ext / 2 - 40
    shape = (int(np.ceil((ext + 80) / PX)),) * 2
    f = _raster(fixed, lo, shape, PX, SIGMA); mf = _hull_mask(fixed, lo, shape, PX)
    f = f / max(f[mf > 0].mean(), 1e-12)
    min_ov = 0.6 * mf.sum()                         # seqIF must lie mostly inside HE

    def score(chir, th, s, npeaks):
        q = s * (moving @ CHIRALITY[chir].T) @ rot(th).T
        tc = fc - q.mean(0)
        qq = q + tc
        m = _raster(qq, lo, shape, PX, SIGMA); mm = _hull_mask(qq, lo, shape, PX)
        m = m / max(m[mm > 0].mean(), 1e-12)
        ncc, _ = masked_ncc(f, mf, m, mm, min_overlap=min_ov)
        loc = (ncc == maximum_filter(ncc, size=5)) & np.isfinite(ncc)
        cand = np.argwhere(loc); vals = ncc[loc]
        fin = ncc[np.isfinite(ncc)]
        med, mad = np.median(fin), np.median(np.abs(fin - np.median(fin))) + 1e-9
        out = []
        for o in np.argsort(vals)[::-1][:npeaks]:
            dy, dx = (cand[o] - (np.array(shape) - 1)) * PX
            out.append(dict(chirality=chir, theta=float(th), scale=float(s), translation=tc + np.array([dx, dy]),
                            ncc=float(vals[o]), robust_z=float((vals[o] - med) / (1.4826 * mad))))
        return out

    coarse = [h for c in chiralities for th in np.arange(0, 360, COARSE_ROT) for s in SCALES_COARSE for h in score(c, th, s, 2)]
    coarse.sort(key=lambda h: -h["ncc"])
    fine, seen = [], set()
    for h in coarse[:8]:
        key = (h["chirality"], round(h["theta"]))
        if key in seen:
            continue
        seen.add(key)
        for th in np.arange(h["theta"] - FINE_HALF, h["theta"] + FINE_HALF + 1e-9, FINE_STEP):
            for s in SCALES_FINE:
                fine.extend(score(h["chirality"], th, s, 2))
    fine.sort(key=lambda h: -h["ncc"])
    return coarse, fine


# ----------------------------------------------------------------------------- cropped ICP refine
def hyp_affine(h):
    M = h["scale"] * rot(h["theta"]) @ CHIRALITY[h["chirality"]]
    return M, np.asarray(h["translation"], float)


def crop_to_fixed(moving, M, t, fixed_path, margin=CROP_MARGIN_UM):
    w = apply_eff(M, t, moving)
    inside = fixed_path.contains_points(w, radius=margin) | fixed_path.contains_points(w, radius=-margin)
    return inside


def refine(h, moving, fixed, fixed_path):
    """Crop -> similarity ICP -> re-crop -> affine ICP -> re-crop -> affine ICP."""
    C = CHIRALITY[h["chirality"]]
    M, t = hyp_affine(h)
    keep = crop_to_fixed(moving, M, t, fixed_path)
    src = moving[keep] @ C.T
    init = SimilarityTransform(scale=h["scale"], rotation_rad=np.deg2rad(h["theta"]), translation=np.asarray(h["translation"]))
    sim, _, _ = icp_similarity(src, fixed, initial=init, max_iter=80, trim_quantile=0.8)
    Ms, ts = sim.to_affine()
    keep = crop_to_fixed(moving, Ms @ C, ts, fixed_path)
    Mi, ti = Ms, ts
    for _ in range(2):
        src = moving[keep] @ C.T
        Mi, ti, err, n = affine_icp(src, fixed, Mi, ti, max_iter=80, trim_quantile=0.7)
        keep = crop_to_fixed(moving, Mi @ C, ti, fixed_path)
    return dict(hyp=h, M_sim=(Ms @ C), t_sim=ts, M=(Mi @ C), t=ti, objective=float(err), n_pairs=int(n), n_crop=int(keep.sum()))


def fit(moving, fixed, fixed_path, chiralities=("proper", "reflect")):
    t0 = time.perf_counter()
    coarse, fine = ncc_search(moving, fixed, chiralities)
    res = [refine(h, moving, fixed, fixed_path) for h in fine[:N_REFINE]]
    res.sort(key=lambda r: r["objective"])
    return dict(best=res[0], refined=res, coarse_top=coarse[:10], fine_top=fine[:10], runtime_s=time.perf_counter() - t0)


# ----------------------------------------------------------------------------- metrics
def sim_params(M):
    det = np.linalg.det(M)
    s = np.sqrt(abs(det))
    C = np.diag([1.0, np.sign(det)])
    R = (M @ C) / s
    return dict(scale=float(s), reflect=bool(det < 0), rotation_deg=float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
                singular_values=np.linalg.svd(M)[1].tolist())


def proximity(world_moving, fixed, fixed_mask, r=(1, 2, 3, 5)):
    d = cKDTree(world_moving).query(fixed[fixed_mask])[0]
    out = {f"frac_le_{x}um": float(np.mean(d <= x)) for x in r}
    out["nn_median_um"] = float(np.median(d))
    return out


def evaluate(M, t, moving, fixed, fixed_path):
    w = apply_eff(M, t, moving)
    # fixed nuclei not too close to the seqIF border (HE covers them all)
    fmask = fixed_path.contains_points(fixed, radius=-2 * CROP_MARGIN_UM) | fixed_path.contains_points(fixed, radius=2 * CROP_MARGIN_UM)
    ev = proximity(w, fixed, np.ones(len(fixed), bool))
    # null: mirror the transformed HE cloud about the seqIF centre (keeps density, destroys identity)
    c = fixed.mean(0)
    null = proximity(np.c_[2 * c[0] - w[:, 0], w[:, 1]], fixed, np.ones(len(fixed), bool))
    ev["null_mirror_x"] = null
    # mutual NN within 3 um
    tr_w, tr_f = cKDTree(w), cKDTree(fixed)
    dfw, ifw = tr_w.query(fixed); dwf, iwf = tr_f.query(w)
    mutual = (iwf[ifw] == np.arange(len(fixed))) & (dfw <= 3)
    ev["mutual_nn_le_3um_frac_fixed"] = float(mutual.mean())
    ev["n_fixed"] = int(len(fixed))
    ev["n_he_in_seqif_field"] = int(crop_to_fixed(moving, M, t, fixed_path, margin=0.0).sum())
    return ev, mutual, ifw


def checker_heldout(moving, fixed, fixed_path, chir, block_um=100.0):
    """Fit on one checkerboard colour of HE nuclei, evaluate the other (2 folds)."""
    col = ((np.floor(moving[:, 0] / block_um) + np.floor(moving[:, 1] / block_um)) % 2).astype(int)
    out = []
    for k in (0, 1):
        r = fit(moving[col == k], fixed, fixed_path, chiralities=(chir,))["best"]
        ev, _, _ = evaluate(r["M"], r["t"], moving[col != k], fixed, fixed_path)
        out.append(dict(fold=k, frac_le_3um=ev["frac_le_3um"], frac_le_2um=ev["frac_le_2um"], nn_median_um=ev["nn_median_um"],
                        null_le_3um=ev["null_mirror_x"]["frac_le_3um"], M=r["M"].tolist(), t=r["t"].tolist()))
    return out


def json_safe(v):
    if isinstance(v, dict):
        return {k: json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    return v


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--data", required=True, help="folder with *_seqif.ome.tif, *_HE*.ome.tif, *_seg_*.geojson")
    ap.add_argument("--nuclei", required=True, help="he_nuclei.csv from detect_he_nuclei.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-prob", type=float, default=0.0)
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s = args.sample
    he_tif = next(p for p in data.glob(f"{s}_HE*.ome.tif"))
    q_tif = data / f"{s}_seqif.ome.tif"
    he_px, q_px = ome_px(he_tif), ome_px(q_tif)

    nuc = pd.read_csv(args.nuclei)
    nuc = nuc[nuc.prob >= args.min_prob].reset_index(drop=True)
    moving = nuc[["centroid_x_px", "centroid_y_px"]].to_numpy() * he_px
    q_rows = load_geojson(data / f"{s}_seg_seqif_level0.geojson")
    seg_ids = np.array([r[0] for r in q_rows])
    fixed = np.array([[r[1], r[2]] for r in q_rows]) * q_px
    hull = ConvexHull(fixed); fixed_path = MPath(fixed[hull.vertices])

    # comparison only: existing placement (seqIF px -> HE px, pure scale + translation)
    h_rows = {r[0]: (r[1], r[2]) for r in load_geojson(data / f"{s}_seg_he.geojson")}
    H = np.array([h_rows[i] for i in seg_ids]); Q = np.array([[r[1], r[2]] for r in q_rows])
    A = np.c_[Q, np.ones(len(Q))]; X = np.linalg.lstsq(A, H, rcond=None)[0]
    Mp_px, tp_px = X[:2].T, X[2]                    # he_px = Mp_px @ q_px + tp_px
    # convert to world = M @ he_um + t
    Mq = np.linalg.inv(Mp_px)
    M_prior = Mq * (q_px / he_px)
    t_prior = -(Mq @ tp_px) * q_px

    print(s, "HE nuclei", len(moving), "seqIF nuclei", len(fixed), flush=True)
    r = fit(moving, fixed, fixed_path)
    best = r["best"]
    ev, mutual, ifw = evaluate(best["M"], best["t"], moving, fixed, fixed_path)
    ev_prior, _, _ = evaluate(M_prior, t_prior, moving, fixed, fixed_path)
    ev_sim, _, _ = evaluate(best["M_sim"], best["t_sim"], moving, fixed, fixed_path)
    h0 = best["hyp"]; M0, t0 = hyp_affine(h0)
    ev_ncc, _, _ = evaluate(M0, t0, moving, fixed, fixed_path)
    print("eval", json.dumps(ev), flush=True)
    held = checker_heldout(moving, fixed, fixed_path, h0["chirality"])

    # displacement between ours and existing placement, on seqIF field
    w_ours = apply_eff(best["M"], best["t"], moving); w_prior = apply_eff(M_prior, t_prior, moving)
    inside = crop_to_fixed(moving, best["M"], best["t"], fixed_path, margin=0.0)
    disp = np.linalg.norm(w_ours - w_prior, axis=1)[inside]

    # transformed HE nuclei CSV (seqIF world um) + seqIF polygons mapped into HE pixel coordinates
    nuc_out = nuc.copy()
    nuc_out["seqif_x_um"], nuc_out["seqif_y_um"] = w_ours[:, 0], w_ours[:, 1]
    nuc_out["seqif_x_px"], nuc_out["seqif_y_px"] = w_ours[:, 0] / q_px, w_ours[:, 1] / q_px
    nuc_out["in_seqif_field"] = inside
    nuc_out.to_csv(out / "he_nuclei_in_seqif_coords.csv", index=False)
    Minv = np.linalg.inv(best["M"])

    def q2he(xy_px):
        return (np.asarray(xy_px, float) * q_px - best["t"]) @ Minv.T / he_px
    feats = []
    for (sid, cx, cy, ring) in q_rows:
        ring_he = q2he(ring)
        c_he = q2he([[cx, cy]])[0]
        feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring_he.round(2).tolist()]},
                      "properties": {"segment_id": sid, "segment_id_str": str(sid), "centroid_x": round(float(c_he[0]), 2),
                                     "centroid_y": round(float(c_he[1]), 2)}})
    (out / f"{s}_seg_he_registered.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    pairs = pd.DataFrame({"segment_id": seg_ids, "seqif_x_um": fixed[:, 0], "seqif_y_um": fixed[:, 1],
                          "he_point_id": nuc.point_id.to_numpy()[ifw],
                          "dist_um": np.linalg.norm(w_ours[ifw] - fixed, axis=1), "mutual_le_3um": mutual})
    pairs.to_csv(out / "seqif_to_he_nearest.csv", index=False)

    summary = dict(
        sample=s, he_image=he_tif.name, he_pixel_um=he_px, seqif_pixel_um=q_px,
        n_he_nuclei=len(moving), n_seqif_nuclei=len(fixed), min_prob=args.min_prob,
        transform=dict(convention="world_seqif_um = M @ he_um + t ; he_um = he_px * he_pixel_um",
                       M=best["M"], t=best["t"], params=sim_params(best["M"]),
                       M_he_px_to_seqif_px=best["M"] * he_px / q_px, t_he_px_to_seqif_px=best["t"] / q_px),
        stages=dict(ncc_init=dict(params=sim_params(M0), ncc=h0["ncc"], robust_z=h0["robust_z"], eval=ev_ncc),
                    similarity_icp=dict(params=sim_params(best["M_sim"]), eval=ev_sim),
                    affine_icp=dict(params=sim_params(best["M"]), objective_um=best["objective"], n_pairs=best["n_pairs"],
                                    n_he_cropped=best["n_crop"], eval=ev)),
        safeguard=dict(scale_change_ncc_to_final=sim_params(best["M"])["scale"] / h0["scale"] - 1,
                       basin_objectives=[x["objective"] for x in r["refined"]],
                       second_basin=[dict(objective=x["objective"], params=sim_params(x["M"]),
                                          median_diff_um=float(np.median(np.linalg.norm(apply_eff(x["M"], x["t"], moving[inside]) - w_ours[inside], axis=1))))
                                     for x in r["refined"][1:]]),
        heldout_checker_100um=held,
        existing_placement_seg_he=dict(params=sim_params(M_prior), eval=ev_prior,
                                       displacement_vs_ours_um=dict(median=float(np.median(disp)), p95=float(np.percentile(disp, 95)), max=float(disp.max()))),
        ncc_top=[dict(h, translation=list(h["translation"])) for h in r["fine_top"][:5]],
        runtime_s=r["runtime_s"],
    )
    (out / "registration_summary.json").write_text(json.dumps(json_safe(summary), indent=2))
    np.savez(out / "registration_transform.npz", M=best["M"], t=best["t"], M_prior=M_prior, t_prior=t_prior,
             he_pixel_um=he_px, seqif_pixel_um=q_px)
    print(json.dumps(json_safe({k: summary[k] for k in ("transform", "safeguard", "heldout_checker_100um")}), indent=1)[:3000])


if __name__ == "__main__":
    main()
