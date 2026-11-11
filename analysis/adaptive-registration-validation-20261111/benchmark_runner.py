"""Reproducible benchmark runner: one (case, method) per call, registration_core 1.2 unchanged.

  python -B benchmark_runner.py --case P8_6_orig --method D
  python -B benchmark_runner.py --list                         (prints the task list used by the Slurm array)

Methods (each source of the engine runs independently, so A and C are read from the D run's per-source records):
  A  point search only (E0x nuclei density, union canvas)          = D's "points" source
  B  image search with the pre-10-27 feature (tissue+nuclear)       = separate run, localization="image"
  C  image search with occ+cleft (registration_core 1.1+)           = D's "image" source
  D  current automatic integration (localization="auto", defaults)  = the production decision

For every run the C2/C3 of the engine is evaluated under BOTH relation gates (same-section nuclear rule and the
adjacent-section quadrant rule) from the same C1 searches; the gate the engine would pick for the case is recorded
separately ("production_gate").  The quadrant check is additionally run on every image candidate (audit only).

New input data: add a case to cases.json (he_sample / if_sample = manifest sample ids, ops = [] for an untouched pair,
reference = null when no reference exists) and run prep.py features for the new samples first.
"""
from __future__ import annotations

import argparse
import json
import resource
import time
import tracemalloc
from dataclasses import asdict

import numpy as np

from common import CORRECT_UM, RUNS, DerivedSide, load_cases, pose_params, probe, probe_dist
from src.registration_core import WorkflowConfig
from src.registration_core import workflow as W
from src.registration_core.frames import json_safe
from src.registration_core.pipeline import footprint
from src.registration_core.search import SearchConfig

METHODS = {"D": dict(), "B": dict(localization="image", image_feature="tissue+nuclear")}
# section relation according to provenance (NOT the engine's file-name inference); see dataset_manifest.csv
METADATA_RELATION = {"031-8_6": "DIFFERENT", "031-6_2": "DIFFERENT (user correction 2026-10-20; conflicts with file-name inference SAME)"}


def build(case):
    he, mv = DerivedSide(case["he_sample"], "HE"), DerivedSide(case["if_sample"], "IF")
    for o in case["he_ops"]:
        he.apply(o)
    for o in case["if_ops"]:
        mv.apply(o)
    fi, mi = he.finish(f"{case['case_id']}:HE"), mv.finish(f"{case['case_id']}:IF")
    return fi, mi, {fi.frame_name: he.fm, mi.frame_name: mv.fm}


def production_relation(case, fi, mi):
    """What registration_core 1.2 would infer for this pair if it were opened as an Explorer sample."""
    if case["he_sample"] != case["if_sample"]:
        return {"value": "unknown", "source": "cross-sample pair: no file-name rule applies -> unknown (treated as same)"}
    s = case["he_sample"]
    r = W.infer_section_relation(s, fi, mi, "auto")
    return {k: v for k, v in r.items() if k != "text"}


def ref_eval(T, ref, moving_extent):
    if not ref:
        return None
    pr = probe(moving_extent)
    d = probe_dist(T["M"], T["t"], ref["M"], ref["t"], pr)
    out = {"probe_dist_um": d, "at_reference": bool(d <= CORRECT_UM)}
    if ref.get("landmarks"):
        L = ref["landmarks"]
        r = np.linalg.norm(np.asarray([l["moving"] for l in L]) @ np.asarray(T["M"]).T + T["t"] - np.asarray([l["fixed"] for l in L]), axis=1)
        out.update({"landmark_median_um": float(np.median(r)), "landmark_max_um": float(r.max()), "n_landmarks": int(len(r))})
    return out


def run(case_id, method):
    case = next(c for c in load_cases() if c["case_id"] == case_id)
    out_dir = RUNS / f"{case_id}__{method}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / "progress.log", "w")
    log = lambda m: (print(m, flush=True), logf.write(m + "\n"), logf.flush())  # noqa: E731
    fi, mi, feats = build(case)
    W.extract = lambda mod, res_um=2.0, **kw: feats[mod.frame.name]       # derived maps (engine code untouched)
    cfg = WorkflowConfig(**METHODS[method])
    timing = {}
    tracemalloc.start()
    w0, c0 = time.perf_counter(), time.process_time()
    searches, st, (fixed, moving), _, _, notes = W.localize(fi, mi, cfg, None, log)
    timing["C1_wall_s"], timing["C1_cpu_s"] = time.perf_counter() - w0, time.process_time() - c0
    for k, v in searches.items():
        timing[f"C1_{k}_wall_s"] = v["runtime_s"]
    names = (fi.frame_name, mi.frame_name)
    probe_d = footprint(moving.extent)
    gates = {}
    for gate in ("same", "adjacent"):
        w1, c1 = time.perf_counter(), time.process_time()
        cands = W.refine_and_validate(searches, fixed, moving, cfg, log, names, gate)
        dec = W.decide(searches, cands, probe_d, "automatic", False, gate)
        timing[f"C2C3_{gate}_wall_s"], timing[f"C2C3_{gate}_cpu_s"] = time.perf_counter() - w1, time.process_time() - c1
        for c in cands:
            c["ref_search"] = ref_eval(c["T_search"], case["reference"], moving.extent)
            c["ref_final"] = ref_eval(c["T_final"], case["reference"], moving.extent)
            c["pose_search"] = pose_params(c["T_search"]["M"])
        gates[gate] = {"decision": dec, "candidates": cands}
    # audit: quadrant check on every image candidate (the engine runs it on image#0 only)
    qa = []
    w2 = time.perf_counter()
    if "image" in searches and fixed.feats is not None:
        scfg = SearchConfig(**{**cfg.search, "combine": cfg.image_combine})
        for c in searches["image"]["candidates"][:cfg.n_candidates]:
            q = W.quadrant_consistency(np.asarray(c["M_search"]), np.asarray(c["t"]), fixed, moving, cfg.image_feature, scfg)
            T = {"M": c["M_search"], "t": c["t"]}
            qa.append({"id": f"image#{c['rank']}", "quadrant": q, "ref": ref_eval(T, case["reference"], moving.extent)})
    timing["quadrant_audit_wall_s"] = time.perf_counter() - w2
    timing["total_wall_s"], timing["total_cpu_s"] = time.perf_counter() - w0, time.process_time() - c0
    rel = production_relation(case, fi, mi)
    res = {"case": case, "method": method, "config": asdict(cfg), "software": W.software_version(), "image_status": st, "notes": notes,
           "production_gate": "adjacent" if rel["value"] == "adjacent" else "same", "engine_relation": rel,
           "metadata_relation": "DIFFERENT" if case["he_sample"] != case["if_sample"] else METADATA_RELATION.get(case["he_sample"], "UNKNOWN"),
           "inputs": {"fixed": {**W.fingerprint(fi), "ops": case["he_ops"]}, "moving": {**W.fingerprint(mi), "ops": case["if_ops"]}},
           "searches": {k: {kk: vv for kk, vv in v.items()} for k, v in searches.items()}, "gates": gates, "quadrant_audit": qa,
           "timing": timing, "peak_tracemalloc_mb": tracemalloc.get_traced_memory()[1] / 2 ** 20,
           "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    (out_dir / "result.json").write_text(json.dumps(json_safe(res), indent=1, ensure_ascii=False, default=str))
    d = gates[res["production_gate"]]["decision"]
    log(json.dumps({"case": case_id, "method": method, "production_gate": res["production_gate"], "status": d["status"],
                    "selected": d["selected_candidate"], "wall_s": round(timing["total_wall_s"])}))


def tasks():
    return [f"{c['case_id']} {m}" for c in load_cases() for m in METHODS]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case"); ap.add_argument("--method", choices=sorted(METHODS)); ap.add_argument("--list", action="store_true")
    ap.add_argument("--index", type=int, help="task index (Slurm array)")
    a = ap.parse_args()
    if a.list:
        print("\n".join(f"{i} {t}" for i, t in enumerate(tasks())))
    elif a.index is not None:
        c, m = tasks()[a.index].split()
        run(c, m)
    else:
        run(a.case, a.method)
