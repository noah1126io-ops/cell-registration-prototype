"""Q3 (prototype of the new independent check): quadrant consistency of an image-search candidate.

The IF field is split 2 x 2.  Each quadrant alone is re-localised by a local image search (same structural maps and NCC
as the search, but only that quadrant's tissue) around the candidate pose.  If the candidate is the true region,
the four independent quadrant optima agree with it; a wrong local optimum of the full-field score has no reason to be
reproduced by each quarter of the tissue separately.

Evaluated on both samples for: reference pose, every image candidate (T_search), and 8 random poses.
Output: quadrants.csv, logs.
"""
import json
import sys
import time

import numpy as np
import pandas as pd

from common2 import C27, reference
from src.registration_core.features import FeatureMaps
from src.registration_core.pipeline import Side, build_layers
from src.registration_core.search import SearchConfig
from src.registration_core.workflow import WorkflowConfig, _local_search

ARTS = {"031-8_6": C27.HERE / "cache/runs/auto_031-8_6_occ+cleft/artifact.json",
        "031-6_2": C27.HERE / "cache/runs/auto_031-6_2_occ+cleft/artifact.json"}
WCFG = WorkflowConfig(local_half_deg=4.0, local_step_deg=1.0, local_scales=(0.97, 1.0, 1.03), local_window_um=100.0)
SCFG = SearchConfig(combine="zsum")


def crop(fm: FeatureMaps, x0, y0, w, h):
    r = fm.res_um
    sl = (slice(int(round(y0 / r)), int(round((y0 + h) / r))), slice(int(round(x0 / r)), int(round((x0 + w) / r))))
    return FeatureMaps(fm.frame, fm.modality, r, {k: v[sl] for k, v in fm.maps.items()}, fm.fov[sl], fm.quality, 0, fm.source_level)


def quadrant_check(M, t, fixed: Side, moving: Side, feature="occ+cleft"):
    me = np.asarray(moving.extent, float)
    out = []
    for qi, (ox, oy) in enumerate([(0, 0), (me[0] / 2, 0), (0, me[1] / 2), (me[0] / 2, me[1] / 2)]):
        o = np.array([ox, oy]); qe = me / 2
        mq = Side(moving.key, moving.points, tuple(qe), crop(moving.feats, ox, oy, *qe))
        # fixed side cropped to the neighbourhood of the mapped quadrant (speed; fixed_world = fixed_local + f0)
        fp = (np.array([[0, 0], [qe[0], 0], [0, qe[1]], qe]) + o) @ M.T + t
        pad = WCFG.local_window_um + 0.5 * np.linalg.norm(qe) + 20.0
        fe = np.asarray(fixed.extent, float)
        f0 = np.clip(fp.min(0) - pad, 0, fe); f1 = np.clip(fp.max(0) + pad, 0, fe)
        f0 = np.floor(f0 / fixed.feats.res_um) * fixed.feats.res_um
        fq = Side(fixed.key, fixed.points, tuple(f1 - f0), crop(fixed.feats, f0[0], f0[1], *(f1 - f0)))
        layers, _ = build_layers("E2", fq, mq, SCFG.res_fine, feature)
        res = _local_search(layers, fq, mq, SCFG, M, M @ o + t - f0, WCFG)
        top = res["top"]
        if top is None:
            out.append({"q": qi, "ok": False}); continue
        Mq, tq = top.A(), top.t + f0 - top.A() @ o      # back to full-IF / full-HE coordinates
        cq = o + qe / 2
        d = float(np.linalg.norm(Mq @ cq + tq - (M @ cq + t)))
        dth = float((top.theta - np.degrees(np.arctan2(M[1, 0], M[0, 0])) + 180) % 360 - 180)
        out.append({"q": qi, "ok": True, "shift_um": d, "d_theta_deg": dth, "scale": top.scale / np.sqrt(abs(np.linalg.det(M))),
                    "z": top.score, "second_z": res["second"].score if res["second"] else None})
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(2)
    rows = []
    samples = sys.argv[1:] or ["031-6_2", "031-8_6"]
    for sample in samples:
        fixed, moving = C27.side(sample, "HE"), C27.side(sample, "IF")
        M0, t0 = reference(sample)
        art = json.load(open(ARTS[sample]))
        poses = [("reference", M0, t0, 0.0)]
        me = np.asarray(moving.extent); fe = np.asarray(fixed.extent)
        for c in art["candidates"]:
            if c["source"] == "image":
                M, t = np.array(c["T_search"]["M"]), np.array(c["T_search"]["t"])
                poses.append((c["id"], M, t, float(np.median(np.linalg.norm(C27.probe(me) @ (M - M0).T + (t - t0), axis=1)))))
        s0 = np.sqrt(abs(np.linalg.det(M0)))
        while len(poses) < 7 + 8:
            a = rng.uniform(0, 2 * np.pi); M = s0 * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            ctr = rng.uniform(me / 2, fe - me / 2)
            if np.linalg.norm(ctr - (M0 @ (me / 2) + t0)) < 300:
                continue
            poses.append(("random", M, ctr - M @ (me / 2), None))
        for name, M, t, dref in poses:
            t1 = time.perf_counter()
            q = quadrant_check(M, t, fixed, moving)
            sh = [x.get("shift_um", np.inf) for x in q]
            row = {"sample": sample, "pose": name, "dist_to_ref_um": dref,
                   "quadrant_shifts_um": json.dumps([round(s, 1) for s in sh]),
                   "quadrant_dtheta": json.dumps([round(x.get("d_theta_deg", np.nan), 1) for x in q]),
                   "quadrant_z": json.dumps([round(x.get("z", np.nan), 1) for x in q]),
                   "n_within_30um": int(sum(s <= 30 for s in sh)), "median_shift_um": float(np.median(sh)), "runtime_s": time.perf_counter() - t1}
            rows.append(row); print(row, flush=True)
            pd.DataFrame(rows).to_csv(f"quadrants_{'_'.join(samples)}.csv", index=False)
