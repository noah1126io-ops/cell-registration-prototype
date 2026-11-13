"""Synthetic C1 top-k (Experiments B and C): E0x difficulty generator (synth.make_case, unchanged) + the CURRENT engine's
point search and C2 (registration_core.localize / refine_and_validate, unchanged), n_candidates = 6.

  python -B run_c1_synthetic.py --list
  python -B run_c1_synthetic.py --index 3          (Slurm array) -> cache/c1syn_<case>.json

The truth (pose, shared nuclei) is only written next to the candidates for later evaluation; it is never passed to the
search, the matching or the reranking.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis/e0x-difficulty-and-web-integration-20261020"))
import synth  # noqa: E402  (E0x difficulty generator, 2026-10-20, unchanged)

from src.registration_core import SideInput, WorkflowConfig  # noqa: E402
from src.registration_core.frames import Affine2D, json_safe  # noqa: E402
from src.registration_research import candidate_adapter as CA  # noqa: E402

# prespecified 2026-11-13 (experiment_protocol.md §3): conditions where the 10-20 sweep's top-1 failed, plus controls
CONDITIONS = ["base", "nonshared=0.5", "missed=0.4", "deform=5", "deform=10", "ratio=0.25", "overlap=0.5", "overlap=0.3",
              "repeat=0.5", "repeat=0.8", "repeat=1", "grid ratio=0.25 ns=0.85", "absent"]
SEEDS = {"dev": (1001,), "eval": (1002, 1003)}
CORRECT_REGION_UM = 20.0      # E0x definition (region_found: grid error <= 20 um), fixed 2026-10-20


K30_CONDITIONS = ["repeat=0.5", "repeat=0.8", "repeat=1", "overlap=0.5", "overlap=0.3", "grid ratio=0.25 ns=0.85", "absent"]


def tasks(k30=False):
    conds = {cid: p for cid, _, p in synth.conditions()}
    out = []
    if k30:   # Experiment C-K30 (protocol change log 2026-11-13): engine config override n_fine=30, code unchanged
        for split, seeds in SEEDS.items():
            for cid in K30_CONDITIONS:
                for s in seeds:
                    out.append({"case": f"K30_{cid.replace(' ', '_')}__{s}", "cond": cid, "params": conds[cid], "seed": s,
                                "split": split, "n_fine": 30})
        return out
    for split, seeds in SEEDS.items():
        for cid in CONDITIONS:
            for s in seeds:
                out.append({"case": f"{cid.replace(' ', '_')}__{s}", "cond": cid, "params": conds[cid], "seed": s, "split": split})
    return out


def run(task):
    C = synth.make_case(task["params"], task["seed"])
    fixed = SideInput("L", C.fixed, (synth.WF, synth.WF), "syn:L")
    moving = SideInput("S", C.moving, (C.wm, C.wm), "syn:S")
    t0 = time.perf_counter()
    nf = task.get("n_fine")
    cfg = WorkflowConfig(localization="points", n_candidates=nf or 6, search={"n_fine": nf} if nf else {})
    cands, meta, sides = CA.run_c1_topk(fixed, moving, cfg, log=lambda m: print(m, flush=True))
    rt = time.perf_counter() - t0
    recs = []
    for c in cands:
        e_s = synth.grid_error(Affine2D(c.M_search, c.t_search, "S", "L"), C.A, C.t, C.wm)
        e_f = synth.grid_error(Affine2D(c.M_final, c.t_final, "S", "L"), C.A, C.t, C.wm)
        recs.append({**c.to_dict(), "grid_err_search_um": e_s, "grid_err_final_um": e_f,
                     "correct_region": bool((not task["params"]["absent"]) and e_f <= CORRECT_REGION_UM)})
    from src.registration_core.search import pose_matrix
    coarse = []
    for k, c in enumerate(meta.get("points", {}).get("coarse_top", [])):     # pre-refinement distinct peaks (top 10)
        M = pose_matrix(c["chirality"], c["theta"], c["scale"])
        coarse.append({"coarse_rank": k, "grid_err_um": synth.grid_error(Affine2D(M, np.asarray(c["t"], float), "S", "L"), C.A, C.t, C.wm)})
    out = {"task": task, "searches": {k: {kk: vv for kk, vv in v.items() if kk != "coarse_top"} for k, v in meta.items()},
           "coarse_top": coarse, "c1_runtime_s": rt, "candidates": recs, "n_fixed": len(C.fixed), "n_moving": len(C.moving),
           "truth": {"A": C.A, "t": C.t, "wm": C.wm, "overlap_achieved": C.overlap_achieved,
                     "shared_moving": C.shared_moving, "shared_fixed": C.shared_fixed}}
    (HERE / "cache").mkdir(exist_ok=True)
    (HERE / "cache" / f"c1syn_{task['case']}.json").write_text(json.dumps(json_safe(out)))
    print(task["case"], [(r["id"], round(r["grid_err_final_um"])) for r in recs], round(rt), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true"); ap.add_argument("--index", type=int); ap.add_argument("--k30", action="store_true")
    a = ap.parse_args()
    if a.list:
        for i, t in enumerate(tasks(a.k30)):
            print(i, t["case"], t["split"])
    else:
        run(tasks(a.k30)[a.index])
