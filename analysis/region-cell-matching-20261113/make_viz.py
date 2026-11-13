"""Point-level display data for the Explorer research tab (viz/<case>__<pose>.json, coordinates rounded to 0.1 um).

Recomputes (same functions, same inputs) the C1 top-1, the reranked top-1 and the best reference candidate of a case
already evaluated by run_phase1.py, plus the local-deformation preview of the reranked top-1.

  python -B make_viz.py P6_2_orig P8_6_orig
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_phase1 as P1  # noqa: E402
from src.registration_research import cell_correspondence as CC  # noqa: E402
from src.registration_research import deformation_experiment as DX  # noqa: E402
from src.registration_research import evaluation as EV  # noqa: E402


def r1(a):
    return np.round(np.asarray(a, float), 1).tolist()


def main(case):
    T = {t["case"]: t for t in P1.tasks()}
    task = T[case]
    loader = P1.load_syn if task["dataset"] == "SYN-C1" else P1.load_real
    fp, fe, mp, me, cands, labels, poses, truth, meta, lm = loader(task)
    saved = json.loads((HERE / "runs" / f"p1_{case}.json").read_text())
    (HERE / "viz").mkdir(exist_ok=True)
    for which in poses:
        pose = saved["poses"][which]
        recs = pose["records"]
        want = [recs[0]["candidate"], pose["decision"]["top"]]
        want += [r["candidate"] for r in recs if r.get("correct_region")][:1]
        want = list(dict.fromkeys([w for w in want if w]))
        out = {"case": case, "pose": which, "candidates": []}
        dets = {}
        for c in cands:
            if c.id not in want:
                continue
            rec, det = CC.evaluate_candidate(c, fp, fe, mp, me, which=which, truth=truth, with_decoy=False, keep_frames=True)
            if det is None:
                continue
            dets[c.id] = (c, det)
            f = det["frames"]["B2"].sort_values("moving_index")
            mi, fi = CC.matched_pairs(f)
            out["candidates"].append({"id": c.id, "fixed": r1(det["fixed"]), "moving": r1(det["moving_mapped"]),
                                      "status": f["status"].tolist(), "pairs": np.column_stack([mi, fi]).astype(int).tolist()})
        top = pose["decision"]["top"]
        if top in dets:
            c, det = dets[top]
            fixed, moving = det["fixed"], det["moving_mapped"]
            gx, gy, bounds = DX.make_grid([fixed, moving], spacing=4.0)
            res, _ = DX.run_all(fixed, moving, gx, gy, 4.0, b2_frame=det["frames"]["B2"])
            out["deformation_candidate"] = top
            out["deformation_fixed"] = r1(fixed)
            out["deformation"] = {}
            sub = (slice(None, None, 10), slice(None, None, 10))
            for m, r in res.items():
                q = np.column_stack([gx[sub].ravel(), gy[sub].ravel(), r["field_x"][sub].ravel(), r["field_y"][sub].ravel()])
                q = q[np.hypot(q[:, 2], q[:, 3]) > 0.05]
                out["deformation"][m] = {"applied": bool(r["applied"]), "points_after": r1(r["points"]), "quiver": r1(q),
                                         "rejection_reason": r.get("rejection_reason")}
        (HERE / "viz" / f"{case}__{which}.json").write_text(json.dumps(out))
        print(case, which, [c["id"] for c in out["candidates"]], flush=True)


if __name__ == "__main__":
    for c in sys.argv[1:]:
        main(c)
