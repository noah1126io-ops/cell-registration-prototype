"""End-to-end research Workflow C runs with registration_core 1.2 (adjacent-section C2/C3), one task per srun.

  python -B run_engine_checks.py --task auto_031-8_6     (localization=auto, defaults; section relation from file names)
  python -B run_engine_checks.py --task auto_031-6_2
Artifacts -> cache/runs/<task>/artifact.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from src.registration_core import WorkflowConfig, run_workflow, write_artifact  # noqa: E402
from src.registration_explorer.catalog import load_manifest, side_input  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); a = ap.parse_args()
    t0 = time.time()
    sample = a.task.split("_", 1)[1]
    out = HERE / "cache" / "runs" / a.task
    out.mkdir(parents=True, exist_ok=True)
    log = lambda m: (print(m, flush=True), open(out / "progress.log", "a").write(m + "\n"))  # noqa: E731
    S = load_manifest()[sample]
    art = run_workflow(sample, side_input(S, "HE"), side_input(S, "IF"), WorkflowConfig(), None, log=log, run_id=a.task)
    write_artifact(art, out / "artifact.json")
    d = art["decision"]
    print(json.dumps({"task": a.task, "status": d["status"], "accepted": d["accepted"], "tissue_level": d.get("accepted_tissue_level"),
                      "failure_stage": d.get("failure_stage"), "relation": art["section_relation"]["value"]}, ensure_ascii=False))
    print("elapsed_s", round(time.time() - t0, 1))
