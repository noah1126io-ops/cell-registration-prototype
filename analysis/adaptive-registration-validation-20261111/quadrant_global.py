"""Spatially more independent alternative to the engine's quadrant check (audit only, not a production rule).

The engine's check re-localises each IF quarter only inside a +-100 um / +-4 deg window around the candidate.  Here each
quarter is localised GLOBALLY over the whole HE field with the unchanged image search (occ+cleft, default SearchConfig).
Agreement of independent global quarter searches with the full-field result is evidence that does not depend on the
candidate itself (no window centred on it).

  python -B quadrant_global.py --index i        (Slurm array; --list prints tasks)
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np

from benchmark_runner import build
from common import RUNS, load_cases
from src.registration_core import WorkflowConfig
from src.registration_core import workflow as W
from src.registration_core.frames import json_safe

CASES = ["P8_6_orig", "P6_2_orig", "N8_6_IFmirror", "N6_2_IFmirror", "Ncross_HE6_2xIF8_6", "Ncross_HE8_6xIF6_2"]


def tasks():
    return [(c, q) for c in CASES for q in range(4)]


def run(case_id, q):
    case = copy.deepcopy(next(c for c in load_cases() if c["case_id"] == case_id))
    fi0, mi0, _ = build(case)
    w, h = (np.asarray(mi0.extent) / 2 // 2 * 2).tolist()
    o = [[0, 0], [w, 0], [0, h], [w, h]][q]
    case["if_ops"] = case["if_ops"] + [["crop", o[0], o[1], w, h]]
    case["case_id"] = f"{case_id}_Q{q}"
    fi, mi, feats = build(case)
    W.extract = lambda mod, res_um=2.0, **kw: feats[mod.frame.name]
    t0 = time.perf_counter()
    searches, st, _, _, _, _ = W.localize(fi, mi, WorkflowConfig(localization="image"), None, print)
    s = searches.get("image")
    out = {"case": case_id, "quadrant": q, "origin_in_parent_moving": o, "extent": [w, h], "verdict": s and s["verdict"], "margin": s and s["margin"],
           "candidates": s and s["candidates"], "image_status": st, "runtime_s": time.perf_counter() - t0}
    p = RUNS / "quadrant_global" / f"{case_id}_Q{q}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(json_safe(out), indent=1, default=str))
    print(case_id, q, out["verdict"], out["runtime_s"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--index", type=int); ap.add_argument("--list", action="store_true"); a = ap.parse_args()
    if a.list:
        print("\n".join(f"{i} {c} Q{q}" for i, (c, q) in enumerate(tasks())))
    else:
        run(*tasks()[a.index])
