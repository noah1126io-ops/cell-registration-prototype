"""Phase 1 (candidate re-evaluation, Experiments B/C) + Phase 2 on the C1 top-1 and the reranked top-1.

  SYN-C1  cache/c1syn_<case>.json (run_c1_synthetic.py)      truth: pose + shared nuclei (evaluation only)
  REAL-D  analysis/adaptive-registration-validation-20261111/runs/<case>__D/result.json   (no cell truth)

  python -B run_phase1.py --list
  python -B run_phase1.py --index 4     -> runs/p1_<case>.json
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
ARV = REPO / "analysis/adaptive-registration-validation-20261111"
sys.path.insert(0, str(REPO))
from src.registration_research import candidate_adapter as CA  # noqa: E402
from src.registration_research import candidate_reranking as RR  # noqa: E402
from src.registration_research import cell_correspondence as CC  # noqa: E402
from src.registration_research import correspondence_anchors as CAn  # noqa: E402
from src.registration_research import deformation_experiment as DX  # noqa: E402
from src.registration_research import evaluation as EV  # noqa: E402
from src.registration_research.legacy import modules  # noqa: E402

E, _, V = modules()


def tasks():
    out = []
    for p in sorted((HERE / "cache").glob("c1syn_*.json")):
        out.append({"dataset": "SYN-C1", "case": p.stem[len("c1syn_"):], "path": str(p)})
    sys.path.insert(0, str(ARV))
    import common as C11  # noqa: E402
    for c in C11.load_cases():
        out.append({"dataset": "REAL-D", "case": c["case_id"], "path": str(ARV / "runs" / f"{c['case_id']}__D" / "result.json")})
    return out


def load_real(task):
    sys.path.insert(0, str(ARV))
    import common as C11  # noqa: E402
    cs = next(c for c in C11.load_cases() if c["case_id"] == task["case"])
    he, mv = C11.DerivedSide(cs["he_sample"], "HE"), C11.DerivedSide(cs["if_sample"], "IF")
    for o in cs["he_ops"]:
        he.apply(o)
    for o in cs["if_ops"]:
        mv.apply(o)
    res = json.loads(Path(task["path"]).read_text())
    cands = CA.from_benchmark_result(Path(task["path"]), "same")
    raw = {c["id"]: c for c in res["gates"]["same"]["candidates"]}
    labels = {}
    for c in cands:
        r = raw[c.id]
        labels[(c.id, "final")] = bool((r.get("ref_final") or {}).get("at_reference", False))
        labels[(c.id, "search")] = bool((r.get("ref_search") or {}).get("at_reference", False))
    poses = ["final", "search"] if cs["he_sample"] == cs["if_sample"] == "031-8_6" else ["final"]
    lm = None
    if cs["case_id"] == "P8_6_orig":
        lm = C11.landmarks_8_6()
    meta = {"expected": cs["expected"], "he_sample": cs["he_sample"], "if_sample": cs["if_sample"], "kind": cs["kind"],
            "production_gate": res["production_gate"], "production_selected": res["gates"][res["production_gate"]]["decision"]["selected_candidate"],
            "production_status": res["gates"][res["production_gate"]]["decision"]["status"]}
    return he.points, tuple(he.extent), mv.points, tuple(mv.extent), cands, labels, poses, None, meta, lm


def load_syn(task):
    d = json.loads(Path(task["path"]).read_text())
    sys.path.insert(0, str(REPO / "analysis/e0x-difficulty-and-web-integration-20261020"))
    import synth  # noqa: E402
    C = synth.make_case(d["task"]["params"], d["task"]["seed"])
    cands, labels = [], {}
    for r in d["candidates"]:
        c = CA.CandidatePose(id=r["id"], source=r["source"], rank=r["rank"], verdict=r["verdict"], margin=r["margin"], score=r["score"],
                             M_search=np.asarray(r["M_search"]), t_search=np.asarray(r["t_search"]), M_final=np.asarray(r["M_final"]),
                             t_final=np.asarray(r["t_final"]), flags=r["flags"], c3=r["c3"])
        cands.append(c)
        labels[(c.id, "final")] = bool(r["correct_region"])
    truth = (np.asarray(C.shared_moving, int), np.asarray(C.shared_fixed, int))
    meta = {"cond": d["task"]["cond"], "seed": d["task"]["seed"], "split": d["task"]["split"], "absent": d["task"]["params"]["absent"],
            "expected": "NEGATIVE" if d["task"]["params"]["absent"] else "POSITIVE",
            "grid_err_final": {r["id"]: r["grid_err_final_um"] for r in d["candidates"]}}
    return C.fixed, (synth.WF, synth.WF), C.moving, (C.wm, C.wm), cands, labels, ["final"], truth, meta, None


def phase2(det, truth_tfi, lm=None, cand=None, which="final"):
    fixed, moving = det["fixed"], det["moving_mapped"]
    gx, gy, bounds = DX.make_grid([fixed, moving], spacing=4.0)
    res, anchors = DX.run_all(fixed, moving, gx, gy, 4.0, b2_frame=det["frames"]["B2"])
    cells = CAn.occupied_cells(moving)
    rows = []
    for m, r in res.items():
        row = {"method": m, "label": DX.LABELS[m], **EV.summary_row(r, 4.0)}
        a = anchors.get(m)
        if a is not None:
            row.update(CAn.quality(a, truth_tfi, cells))
        if truth_tfi is not None:
            row.update(EV.q(EV.nuclei_tre(r, moving, fixed, truth_tfi, bounds, 4.0), "tre_known_um"))
            row.update(EV.q(EV.nuclei_tre(r, moving, fixed, truth_tfi, bounds, 4.0, exclude=None if a is None else a.moving_index),
                            "tre_nonanchor_um"))
        au = r.get("audit") or {}
        for k in ("before", "after"):
            if isinstance(au.get(k), dict):
                row[f"nn_sym_median_{k}"] = au[k].get("symmetric_median_distance")
        if lm is not None and cand is not None:
            M, t = cand.T(which)
            mvl = np.asarray([l["moving"] for l in lm], float) @ np.asarray(M).T + t
            fxl = np.asarray([l["fixed"] for l in lm], float)
            after = mvl + EV.field_at(mvl, r, bounds, 4.0)
            row["landmark_residual_affine_median_um"] = float(np.median(np.linalg.norm(mvl - fxl, axis=1)))
            row["landmark_residual_after_median_um"] = float(np.median(np.linalg.norm(after - fxl, axis=1)))
            row["landmark_n"] = len(lm)
        rows.append(row)
    return rows


def run(task):
    t0 = time.perf_counter()
    loader = load_syn if task["dataset"] == "SYN-C1" else load_real
    fp, fe, mp, me, cands, labels, poses, truth, meta, lm = loader(task)
    if task.get("max_cands"):          # smoke tests only (never used for reported runs)
        cands = cands[:task["max_cands"]]
    out = {"task": task, "meta": meta, "poses": {}}
    for which in poses:
        recs, dets = [], []
        for c in cands:
            rec, det = CC.evaluate_candidate(c, fp, fe, mp, me, which=which, truth=truth, seed=0, keep_frames=True)
            rec["correct_region"] = labels.get((c.id, which), False)
            recs.append(rec)
            dets.append(det)
            print(task["case"], which, c.id, rec.get("status"), rec.get("B2_matched_frac"), rec.get("decoy_B2_matched_frac"),
                  rec["correct_region"], flush=True)
        dec = RR.decide(recs)
        orders = {k: [recs[i]["candidate"] for i in RR.order(recs, k)] for k in ("excess", *RR.ALTERNATIVES)}
        # Phase 2 on the C1 top-1 (first record = the engine's primary source rank 0) and on the reranked top-1
        p2 = {}
        for tag, idx in (("c1_top1", 0), ("rerank_top1", dec.get("top_index"))):
            if idx is None or dets[idx] is None or (tag == "rerank_top1" and idx == 0 and "c1_top1" in p2):
                continue
            tfi = None
            if truth is not None:
                fr = dets[idx]["frames"]["B2"]
                tfi = fr["true_fixed_index"].to_numpy(int)
            p2[tag] = {"candidate": recs[idx]["candidate"], "correct_region": recs[idx]["correct_region"],
                       "rows": phase2(dets[idx], tfi, lm, cands[idx], which)}
        out["poses"][which] = {"records": recs, "decision": dec, "orders": orders, "phase2": p2}
    out["runtime_s"] = time.perf_counter() - t0
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / f"p1_{task['case']}{'_SMOKE' if task.get('max_cands') else ''}.json").write_text(json.dumps(E.json_safe(out), allow_nan=True))
    print("done", task["case"], round(out["runtime_s"]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true"); ap.add_argument("--index", type=int); ap.add_argument("--case")
    ap.add_argument("--smoke", type=int, help="limit candidates (smoke test; output *_SMOKE.json)")
    a = ap.parse_args()
    T = tasks()
    if a.list:
        for i, t in enumerate(T):
            print(i, t["dataset"], t["case"])
    elif a.case:
        run({**next(t for t in T if t["case"] == a.case), "max_cands": a.smoke})
    else:
        run(T[a.index])
