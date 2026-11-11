"""Parity: research Workflow C via the Explorer/Slurm Runner (runs/registration-explorer/<id>/artifact.json) vs the
same engine called directly (cache/artifact_auto_<sample>_HEfix.json).  -> cache/parity_runner_vs_direct.json"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
out = {}
for run_dir in sys.argv[1:]:
    a = json.loads((REPO / run_dir / "artifact.json").read_text())
    b = json.loads((HERE / f"cache/artifact_auto_{a['sample']}_{a['input']['fixed']['key']}fix.json").read_text())
    ca = {c["id"]: c for c in a["candidates"]}; cb = {c["id"]: c for c in b["candidates"]}
    dM = max(float(np.abs(np.asarray(ca[k][f]["M"]) - np.asarray(cb[k][f]["M"])).max()) for k in ca for f in ("T_search", "T_final"))
    dt = max(float(np.abs(np.asarray(ca[k][f]["t"]) - np.asarray(cb[k][f]["t"])).max()) for k in ca for f in ("T_search", "T_final"))
    sel = a["decision"]["selected_candidate"]
    fa = (ca[sel]["validation"].get("if_frame") or {}).get("final", {}); fb = (cb[sel]["validation"].get("if_frame") or {}).get("final", {})
    out[a["sample"]] = {"runner_run": a["run_id"], "same_candidate_ids": sorted(ca) == sorted(cb), "max_abs_dM_all_candidates": dM,
                        "max_abs_dt_all_candidates": dt, "decision_runner": a["decision"]["status"], "decision_direct": b["decision"]["status"],
                        "selected_runner": sel, "selected_direct": b["decision"]["selected_candidate"],
                        "nn_median_runner": fa.get("nn_median_um"), "nn_median_direct": fb.get("nn_median_um"),
                        "le_3um_runner": fa.get("le_3um"), "le_3um_direct": fb.get("le_3um"),
                        "input_fingerprint_equal": a["input"] == b["input"], "runtime_runner_s": a["runtime_s"], "runtime_direct_s": b["runtime_s"]}
(HERE / "cache/parity_runner_vs_direct.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
