"""Q2: which tissue-level image measures separate the reference pose from wrong poses, on both samples?

For each sample: reference pose, the engine's image/point candidates (T_final), and 300 random poses (>= 50 % of the IF
field inside HE).  Measures in the IF frame at 2 um (HE maps warped by the inverse): masked correlation of each map,
tissue Dice, envelope-edge chamfer.  Search features (occ, cleft) are marked: they are NOT independent of the search.

Output: imgnull.csv (one row per pose), imgnull_summary.json (reference z-score vs random, rank among candidates)
"""
import json

import numpy as np
import pandas as pd

from common2 import C27, reference
from src.registration_core import metrics as mt
from src.registration_core.frames import Affine2D
from src.registration_core.pipeline import footprint

ARTS = {"031-8_6": C27.HERE / "cache/runs/auto_031-8_6_occ+cleft/artifact.json",
        "031-6_2": C27.HERE / "cache/runs/auto_031-6_2_occ+cleft/artifact.json"}
MAPS = ("occ", "cleft", "tissue", "nuclear", "edt")
rng = np.random.default_rng(1)


def measures(M, t, he, iff, res=2.0):
    T = Affine2D(np.asarray(M), np.asarray(t), "IF", "HE").inverse()          # HE -> IF
    shp = iff.shape
    valid = np.isfinite(mt.warp_map_into(np.ones(he.shape, np.float32), he.res_um, T, shp, res, order=0))
    out = {"valid_frac": float(valid.mean())}
    for k in MAPS:
        w = mt.warp_map_into(he.maps[k], he.res_um, T, shp, res)
        out[f"{k}_corr"] = mt.masked_corr(w, iff.maps[k], valid)
    wt = mt.warp_map_into(he.maps["tissue_bin"].astype(np.float32), he.res_um, T, shp, res)
    out["tissue_dice"] = mt.dice(wt, iff.maps["tissue_bin"].astype(np.float32), valid)
    we = mt.warp_map_into(he.maps["envelope"].astype(np.float32), he.res_um, T, shp, res, order=0) > 0.5
    out["env_chamfer_um"], out["n_env_px"] = mt.boundary_chamfer(iff.maps["envelope"], we, valid, res)
    return out


rows, summ = [], {}
for sample in ("031-6_2", "031-8_6"):
    he, iff = C27.features(sample, "HE"), C27.features(sample, "IF")
    fe, me = np.array(he.shape[::-1]) * he.res_um, np.array(iff.shape[::-1]) * iff.res_um
    M0, t0 = reference(sample)
    s0 = np.sqrt(abs(np.linalg.det(M0)))
    ctr0 = M0 @ (me / 2) + t0
    poses = [("reference", M0, t0)]
    art = json.load(open(ARTS[sample]))
    for c in art["candidates"]:
        poses.append((c["id"], np.array(c["T_final"]["M"]), np.array(c["T_final"]["t"])))
    while len(poses) < 1 + len(art["candidates"]) + 300:
        a = rng.uniform(0, 2 * np.pi); M = s0 * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        ctr = rng.uniform(0, fe)
        if np.linalg.norm(ctr - ctr0) < 200:
            continue
        t = ctr - M @ (me / 2)
        poses.append(("random", M, t))
    for name, M, t in poses:
        r = measures(M, t, he, iff)
        if name == "random" and r["valid_frac"] < 0.5:
            continue
        d = float(np.median(np.linalg.norm(C27.probe(me) @ (M - M0).T + (t - t0), axis=1)))
        rows.append({"sample": sample, "pose": name, "dist_to_ref_um": d, **r})
    df = pd.DataFrame([r for r in rows if r["sample"] == sample])
    rnd = df[df.pose == "random"]; ref = df[df.pose == "reference"].iloc[0]
    wrong_c = df[(df.pose.str.contains("#")) & (df.dist_to_ref_um > 100)]
    s = {"n_random": int(len(rnd))}
    for k in [f"{m}_corr" for m in MAPS] + ["tissue_dice", "env_chamfer_um"]:
        sign = -1 if k == "env_chamfer_um" else 1
        mu, sd = rnd[k].mean(), rnd[k].std()
        s[k] = {"reference": float(ref[k]), "random_mean": float(mu), "random_sd": float(sd),
                "z_vs_random": float(sign * (ref[k] - mu) / sd) if sd > 0 else None,
                "frac_random_beating_ref": float(np.mean(sign * rnd[k] >= sign * ref[k])),
                "best_wrong_candidate": float((wrong_c[k].min() if sign < 0 else wrong_c[k].max())) if len(wrong_c) else None,
                "independent_of_search": k.split("_")[0] not in ("occ", "cleft")}
    summ[sample] = s
    print(sample, json.dumps(s, indent=0), flush=True)
pd.DataFrame(rows).to_csv("imgnull.csv", index=False)
json.dump(summ, open("imgnull_summary.json", "w"), indent=1)
