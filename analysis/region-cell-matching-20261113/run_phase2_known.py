"""Experiment A: known global transform.  Local deformation methods A/B/B'/C/D/E + correspondence B0/B1/B3/B2.

  SYN-FIELD  validation_lib.generate_synthetic_sample, 10 scenarios x seeds 1101-1105 (the 2026-09-11 design, unchanged)
  SYN-RL     experiment_lib.generate_real_like_sample, Test split only (seeds 4101-4105; never used to fit/threshold B2)

  python -B run_phase2_known.py --list
  python -B run_phase2_known.py --index 7      -> cache/p2known_<case>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MPath
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from src.registration_research import cell_correspondence as CC  # noqa: E402
from src.registration_research import correspondence_anchors as CAn  # noqa: E402
from src.registration_research import deformation_experiment as DX  # noqa: E402
from src.registration_research import evaluation as EV  # noqa: E402
from src.registration_research.legacy import UOT_STUDY, modules  # noqa: E402

E, _, V = modules()
sys.path.insert(0, str(UOT_STUDY))
import validate_synthetic as VS  # noqa: E402

FIELD_SEEDS = (1101, 1102, 1103, 1104, 1105)


def tasks():
    out = [{"dataset": "SYN-FIELD", "case": f"field_{s['scenario_id']}_{seed}", "scenario": s, "seed": seed}
           for seed in FIELD_SEEDS for s in VS.scenario_table()]
    out += [{"dataset": "SYN-RL", "case": s["scenario_id"], "scenario": s, "seed": s["seed"]}
            for s in E.real_like_scenarios() if s["split"] == "Test"]
    return out


def load(task):
    s = task["scenario"]
    if task["dataset"] == "SYN-FIELD":
        smp = V.generate_synthetic_sample(seed=task["seed"], amplitude_um=s["amplitude_um"], missing_fraction=s["missing_fraction"],
                                          unmatched_fraction=s["unmatched_fraction"], overlap_fraction=s["overlap_fraction"],
                                          noise_um=s["noise_um"])
        tfi = EV.truth_index_by_coordinates(smp.moving_points, smp.fixed_points, smp.matched_moving_points, smp.matched_fixed_targets)
        return smp, smp.fixed_points, smp.moving_points, tfi, smp.grid_x, smp.grid_y, smp.spacing, smp.bounds, smp.true_x, smp.true_y, \
            np.asarray(smp.ground_truth_evaluation_mask, bool)
    smp = E.generate_real_like_sample(seed=task["seed"], strength=s["strength"])
    tfi = np.asarray(smp.true_fixed_index, int)
    hull = smp.moving_points[ConvexHull(smp.moving_points).vertices]
    gx, gy = smp.grid_x, smp.grid_y
    mask = MPath(hull).contains_points(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    return smp, smp.fixed_points, smp.moving_points, tfi, gx, gy, smp.spacing, smp.bounds, smp.true_field_x, smp.true_field_y, mask


def run(task):
    t0 = time.perf_counter()
    smp, fixed, moving, tfi, gx, gy, sp, bounds, tx, ty, mask = load(task)
    fixed, moving = np.asarray(fixed, float), np.asarray(moving, float)
    pipe = CC.run_pipeline(fixed, moving, tfi, task["seed"], task["case"])
    match_rows = [{"dataset": task["dataset"], "case": task["case"], "method": k, **CC.truth_metrics(pipe["frames"][k]),
                   **{"matched_n": int((pipe["frames"][k]["status"] == "matched").sum())}} for k in CC.METHODS]
    mnn = CAn.mnn(moving, fixed)
    match_rows.append({"dataset": task["dataset"], "case": task["case"], "method": "MNN<=10um", "matched_n": len(mnn),
                       "precision": float(np.mean(tfi[mnn.moving_index] == mnn.fixed_index)) if len(mnn) else np.nan,
                       "recall": float(np.sum(tfi[mnn.moving_index] == mnn.fixed_index) / max((tfi >= 0).sum(), 1)),
                       "common_n": int((tfi >= 0).sum())})
    res, anchors = DX.run_all(fixed, moving, gx, gy, sp, b2_frame=pipe["frames"]["B2"])
    cells = CAn.occupied_cells(moving)
    rows = []
    for m, r in res.items():
        row = {"dataset": task["dataset"], "case": task["case"], "seed": task["seed"], "method": m, "label": DX.LABELS[m],
               "scenario": task["scenario"].get("scenario_id"), "difficulty": task["scenario"].get("difficulty") or task["scenario"].get("strength"),
               **EV.summary_row(r, sp)}
        row.update(EV.q(EV.epe_on_grid(r, tx, ty, mask), "epe_um"))
        row.update(EV.q(EV.epe_on_grid(r, tx, ty, mask, key="attempted_field"), "attempted_epe_um"))
        a = anchors.get(m)
        if a is not None:
            row.update(CAn.quality(a, tfi, cells))
        row.update(EV.q(EV.nuclei_tre(r, moving, fixed, tfi, bounds, sp), "tre_known_um"))
        row.update(EV.q(EV.nuclei_tre(r, moving, fixed, tfi, bounds, sp, exclude=None if a is None else a.moving_index), "tre_nonanchor_um"))
        um = np.flatnonzero(tfi < 0)
        if len(um):
            row.update(EV.q(np.linalg.norm(EV.field_at(moving[um], r, bounds, sp), axis=1), "unmatched_move_um"))
        if task["dataset"] == "SYN-FIELD":
            ev = V.evaluate_synthetic(smp, {**r, "method": m})
            row.update({k: ev[k] for k in ("independent_landmark_tre_um_median", "independent_landmark_tre_um_p95", "field_error_um_median",
                                           "field_error_um_p95", "jacobian_abs_error_median") if k in ev})
        rows.append(row)
    out = {"task": {k: v for k, v in task.items()}, "deformation": rows, "matching": match_rows, "runtime_s": time.perf_counter() - t0}
    (HERE / "cache" / f"p2known_{task['case']}.json").write_text(json.dumps(E.json_safe(out), allow_nan=True))
    print(task["case"], {r["method"]: (r["applied"], round(r["epe_um_median"], 2)) for r in rows}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true"); ap.add_argument("--index", type=int); ap.add_argument("--chunks", type=int)
    a = ap.parse_args()
    T = tasks()
    if a.list:
        for i, t in enumerate(T):
            print(i, t["dataset"], t["case"])
    elif a.chunks:
        for t in T[a.index::a.chunks]:
            if not (HERE / "cache" / f"p2known_{t['case']}.json").exists():
                run(t)
    else:
        run(T[a.index])
