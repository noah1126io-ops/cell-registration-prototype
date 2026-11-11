"""End-to-end research Workflow C runs on the shared engine (src/registration_core), one task per srun.

  python -B run_engine_checks.py --task auto_031-8_6_occ+cleft      (localization=auto, image_feature=occ+cleft)
  python -B run_engine_checks.py --task init_031-8_6_ref            (ASSISTED: local search around the manual reference)
  python -B run_engine_checks.py --task feature_parity              (tissue / nuclear / edt maps unchanged by the 10-27 edit)
Artifacts -> cache/runs/<task>/artifact.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from common import CACHE, HERE, store_transform
from src.registration_core import Constraints, WorkflowConfig, run_workflow, write_artifact
from src.registration_core.features import Modality, extract
from src.registration_explorer.catalog import load_manifest, side_input


def sides(sample, fixed="HE"):
    S = load_manifest()[sample]
    return side_input(S, fixed), side_input(S, S.other(fixed))


def run(task):
    out = CACHE / "runs" / task
    out.mkdir(parents=True, exist_ok=True)
    log = lambda m: (print(m, flush=True), open(out / "progress.log", "a").write(m + "\n"))  # noqa: E731
    kind, rest = task.split("_", 1)
    sample, arg = rest[:7], rest[8:]          # sample ids look like 031-8_6
    f, m = sides(sample)
    if kind == "auto":
        cfg = WorkflowConfig(localization="auto", image_feature=arg)
        art = run_workflow(sample, f, m, cfg, None, log=log, run_id=task)
    elif kind == "init":
        M, t, row = store_transform()
        cons = Constraints(init={"M": M.tolist(), "t": t.tolist(), "from": f"manual reference {row['id']}"}, annotation_ids=[row["id"]])
        art = run_workflow(sample, f, m, WorkflowConfig(localization="auto", image_feature="occ+cleft"), cons, log=log, run_id=task)
    else:
        raise ValueError(task)
    write_artifact(art, out / "artifact.json")
    d = art["decision"]
    print(json.dumps({"task": task, "status": d["status"], "accepted": d["accepted"], "failure_stage": d.get("failure_stage")}))


def feature_parity():
    """The 10-27 edit only adds maps: tissue / nuclear / edt / tissue_bin / envelope must equal the pre-edit cache."""
    res = {}
    for sample in ("031-8_6", "031-6_2"):
        for key in ("HE", "IF"):
            si = side_input(load_manifest()[sample], key)
            fm = extract(Modality(si.frame(), np.asarray(si.points, float)), res_um=2.0)
            z = np.load(CACHE / f"feat_{sample}_{key}_2.npz", allow_pickle=True)
            res[f"{sample}:{key}"] = {k: float(np.abs(fm.maps[k].astype(float) - z[f"map_{k}"].astype(float)).max())
                                      for k in ("tissue", "nuclear", "edt", "tissue_bin", "envelope")}
            from variants import variant_maps
            v = variant_maps(sample, key)
            res[f"{sample}:{key}"]["occ_vs_analysis"] = float(np.abs(fm.maps["occ"] - v["occ"]).max())
            res[f"{sample}:{key}"]["cleft_vs_analysis"] = float(np.abs(fm.maps["cleft"] - 1.0 - v["cleft"]).max())
    (CACHE / "feature_parity.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); a = ap.parse_args()
    t0 = time.time()
    feature_parity() if a.task == "feature_parity" else run(a.task)
    print("elapsed_s", round(time.time() - t0, 1))
