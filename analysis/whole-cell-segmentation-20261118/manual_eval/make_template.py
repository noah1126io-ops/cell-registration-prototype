"""Manual evaluation set (Phase 2): a fixed random sample of cells from the EVALUATION ROIs only (tuning ROIs excluded).

Fill `judgement` per row (one of JUDGEMENTS) looking at the image; `boundary_uncertain` = 1 when the true boundary itself is not
visible (such rows are reported separately, never counted as correct or wrong).  Nothing here is ground truth until a person fills it.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
JUDGEMENTS = ["correct", "over_segmented", "merged", "partial", "not_a_cell", "nucleus_only"]
METHODS = {"IF": ["S1_cpsam_if_mem", "S3_watershed"], "HE": ["S1_cpsam_he_rgb", "S3_watershed"]}
PER_ROI = 8

rows = []
rng = np.random.default_rng(20261118)
rois = json.loads((STUDY / "rois.json").read_text())
for sample, sv in rois["samples"].items():
    for sec, ms in METHODS.items():
        for m in ms:
            p = STUDY / "results" / m / sample / f"{sec}.geojson"
            if not p.is_file():
                continue
            feats = json.loads(p.read_text())["features"]
            for roi, r in sv["rois"].items():
                if r is None or r["role"] != "evaluation":
                    continue
                fs = [f for f in feats if f["properties"]["roi"] == roi and "touches_roi_border" not in f["properties"]["quality_flags"]]
                for f in rng.permutation(len(fs))[:PER_ROI]:
                    pr = fs[int(f)]["properties"]
                    rows.append({"sample_id": sample, "section": sec, "roi": roi, "method": m, "cell_segment_id": pr["cell_segment_id"],
                                 "centroid_x": pr["centroid_x"], "centroid_y": pr["centroid_y"], "area_um2": pr["area_um2"],
                                 "judgement": "", "boundary_uncertain": "", "reviewer": "", "comment": ""})
pd.DataFrame(rows).to_csv(HERE / "template.csv", index=False)
print(len(rows), "rows; judgements:", JUDGEMENTS)
