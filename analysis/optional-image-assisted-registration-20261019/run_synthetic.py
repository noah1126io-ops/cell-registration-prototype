"""Semi-synthetic known-truth cases (one case per call, all methods): cache/syn_<id>.json

Case list is fixed before any run: 2 HE sources x {q 0.8, 0.5, 0.3, 0.5-absent} x 4 seeds = 32 cases.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from oir.frames import json_safe  # noqa: E402
from oir.pipeline import Side, evaluate, run_method  # noqa: E402
from oir.sample import load_sample  # noqa: E402
from oir.search import SearchConfig  # noqa: E402
from oir.synthetic import make_case, truth_errors  # noqa: E402
from run_real import CACHE, load_feats  # noqa: E402

SOURCES = ("031-6_2", "031-8_6")
CONDS = (("q0.8", 0.8, False), ("q0.5", 0.5, False), ("q0.3", 0.3, False), ("absent_q0.5", 0.5, True))
SEEDS = (0, 1, 2, 3)
CASES = [(src, c, q, ab, sd) for src in SOURCES for (c, q, ab) in CONDS for sd in SEEDS]
METHODS = (("E0", "P0", {}), ("E0x", "P0", {}), ("E1", "P1", {"combine": "zsum"}), ("E2", "P2", {}), ("E3", "P2", {"combine": "zsum"}))
SUCCESS = {"region_median_um": 20.0, "registration_rms_um": 5.0}


def run_case(i):
    src, cond, q, absent, seed = CASES[i]
    S = load_sample(src)
    he_f = load_feats(CACHE / f"feats_{src}_HE_2um.npz")
    he = Side("HE", S.he.nuclei_um, S.he.frame.extent_um(), he_f)
    case = make_case(f"syn{i:03d}", he, seed=1000 + 17 * i, q_shared=q, absent=absent, condition=cond, source=src)
    rec = {"case": i, "source": src, "condition": cond, "q_shared": q, "absent": absent, "seed": seed,
           "truth": {"A": case.A_true, "t": case.t_true}, "n_moving": len(case.moving.points), "n_shared": int(len(case.shared_moving_idx)),
           "moving_tissue_fraction": float(case.moving.feats.maps["tissue_bin"].mean()), "methods": {}}
    for name, mode, over in METHODS:
        fixed = Side("HE", case.fixed.points, case.fixed.extent, case.fixed.feats if mode in ("P1", "P2") else None)
        moving = Side("IF", case.moving.points, case.moving.extent, case.moving.feats if mode == "P2" else None)
        R = run_method(rec["case"], name, fixed, moving, mode, SearchConfig(**over))
        r = {"verdict": R.search.get("verdict"), "margin": R.search.get("margin"), "runtime_s": R.runtime_s, "peak_mem_mb": R.peak_mem_mb,
             "top": R.search.get("top"), "icp_rejected": R.refine.get("icp_rejected")}
        if R.T is not None:
            r["truth_final"] = truth_errors(case, R.T.M, R.T.t)
            r["truth_search"] = truth_errors(case, R.T_search.M, R.T_search.t) if R.T_search is not None else None
            r["params"] = R.T.params()
            ev = evaluate(R, Side("HE", case.fixed.points, case.fixed.extent, None), Side("IF", case.moving.points, case.moving.extent, None),
                          case.fixed.feats, case.moving.feats)
            r["eval"] = {k: ev.get(k) for k in ("final", "null", "inlier_over_null", "geometric_pass", "image")}
            # true identity: shared moving nuclei whose nearest fixed nucleus (<=3 um) is the true partner
            if not absent and len(case.shared_moving_idx):
                w = R.T.apply(case.moving.points[case.shared_moving_idx])
                d, j = cKDTree(case.fixed.points).query(w)
                r["true_id_rate_shared"] = float(np.mean((j == case.shared_fixed_idx) & (d <= 3.0)))
                wa = R.T.apply(case.moving.points); da, ja = cKDTree(case.fixed.points).query(wa)
                prox = da <= 3.0
                is_true = np.zeros(len(wa), bool); is_true[case.shared_moving_idx] = ja[case.shared_moving_idx] == case.shared_fixed_idx
                r["precision_of_le3um_pairs"] = float(np.mean(is_true[prox])) if prox.any() else np.nan
            rf = r["truth_final"]
            r["region_found"] = bool(rf["median_um"] <= SUCCESS["region_median_um"])
            r["registration_success"] = bool(rf["rms_um"] <= SUCCESS["registration_rms_um"])
        else:
            r["region_found"] = False; r["registration_success"] = False
        rec["methods"][name] = r
        print(i, src, cond, name, r["verdict"], r.get("truth_final"), r.get("eval", {}).get("geometric_pass"), "%.0fs" % R.runtime_s, flush=True)
    (CACHE / f"syn_{i:03d}.json").write_text(json.dumps(json_safe(rec), indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--case", type=int, required=True)
    run_case(ap.parse_args().case)
