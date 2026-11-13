"""REAL-S: serial_section_pair_20260904 (points only, saved A2 R2 affine, UNVERIFIED).  No cell-level truth.

Phase 2 methods on the saved post-affine points (the same input as the 09-11 / 10-14 studies) -> runs/real_s.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from src.registration_research import cell_correspondence as CC  # noqa: E402
from src.registration_research import correspondence_anchors as CAn  # noqa: E402
from src.registration_research import deformation_experiment as DX  # noqa: E402
from src.registration_research import evaluation as EV  # noqa: E402
from src.registration_research.legacy import modules  # noqa: E402

E, _, V = modules()


def main():
    fixed = np.load(E.REAL_INPUT / "success_metric_fixed_points.npy")
    moving = np.load(E.REAL_INPUT / "success_metric_moving_points.npy")
    pipe = CC.run_pipeline(fixed, moving, None, 20260904, "serial_section_pair_20260904")
    summ = CC.summarize(pipe, fixed, moving)
    dec = CC.run_pipeline(fixed, CC.mirror_decoy(moving), None, 20260904, "decoy")
    summ.update({f"decoy_{k}": v for k, v in CC.summarize(dec, fixed, CC.mirror_decoy(moving)).items() if k.endswith("matched_frac")})
    gx, gy, bounds = DX.make_grid([fixed, moving], spacing=4.0)
    res, anchors = DX.run_all(fixed, moving, gx, gy, 4.0, b2_frame=pipe["frames"]["B2"])
    cells = CAn.occupied_cells(moving)
    rows = []
    for m, r in res.items():
        row = {"method": m, "label": DX.LABELS[m], **EV.summary_row(r, 4.0)}
        if m in anchors:
            row.update(CAn.quality(anchors[m], None, cells))
        au = r.get("audit") or {}
        for k in ("before", "after"):
            if isinstance(au.get(k), dict):
                row[f"nn_sym_median_{k}"] = au[k].get("symmetric_median_distance")
        rows.append(row)
    out = {"dataset": "REAL-S", "case": "serial_section_pair_20260904", "correspondence": summ, "deformation": rows}
    (HERE / "runs" / "real_s.json").write_text(json.dumps(E.json_safe(out), allow_nan=True))
    print({r["method"]: (r["applied"], r["rejection_reason"], r.get("anchors_n")) for r in rows})


if __name__ == "__main__":
    main()
