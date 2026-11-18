"""Whole IF section with the only method that passed the ROI comparison (S1_cpsam_if_mem), for display in Cell Review.

Decided AFTER the ROI experiment (REPORT §4): IF + Cellpose-SAM on the G+R membrane proxy gave plausible cardiomyocyte
outlines on 031-8_6 and partial ones on 031-6_2; every H&E method failed, so H&E is not run on the full section.
Cellpose tiles internally (overlapping tiles, flows stitched before mask creation), so there is no separate tile seam.

Steps (each in its own env):
  crop    (cell-rg)     cache/crops/<sample>/full/IF.png at 0.5 um/px + crop_meta.json  (tile reads, no full-res load)
  seg     (cellseg-cp)  cache/seg/S1_cpsam_if_mem_full/<sample>/full/IF.npz
  export  (cell-rg)     results/S1_cpsam_if_mem_full/<sample>/IF.geojson + IF_nucleus_cell.csv + full_section_quality.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SAMPLES = ("031-6_2", "031-8_6")
METHOD = "S1_cpsam_if_mem_full"


def crop():
    sys.path.insert(0, str(HERE))
    import roi_select as R
    from PIL import Image
    for s in SAMPLES:
        S = R.catalog.load_manifest()[s]
        m = S.modality("IF")
        w, h = m.extent()
        rgb, valid, info = R.crop_world(m, 0.0, 0.0, w, h, R.WORK_UM)
        d = HERE / "cache/crops" / s / "full"
        d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(d / "IF.png")
        (d / "crop_meta.json").write_text(json.dumps({"IF": {"origin_um": [0.0, 0.0], "um_per_px": R.WORK_UM, "shape": list(rgb.shape[:2]),
                                                             "valid_fraction": float(valid.mean()), "source": m.image_path, **info}}, indent=1))
        print(s, rgb.shape)


def seg():
    sys.path.insert(0, str(HERE))
    import torch
    from cellpose import models
    from PIL import Image
    import run_cellpose as RC
    model = models.CellposeModel(gpu=torch.cuda.is_available(), pretrained_model="cpsam")
    log = {"method": METHOD, "variant": "if_mem", "cellprob_threshold": 0.0, "flow_threshold": 0.4, "runs": []}
    for s in SAMPLES:
        d = HERE / "cache/crops" / s / "full"
        meta = json.loads((d / "crop_meta.json").read_text())["IF"]
        x = RC.prep(np.asarray(Image.open(d / "IF.png").convert("RGB")), "if_mem")
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        masks, _, _ = model.eval(x, channel_axis=-1, cellprob_threshold=0.0, flow_threshold=0.4, normalize=True)
        rec = {"sample": s, "shape": list(x.shape), "seconds": time.time() - t, "gpu_peak_mb": torch.cuda.max_memory_allocated() / 2**20,
               "n_labels": int(np.asarray(masks).max())}
        out = HERE / "cache/seg" / METHOD / s / "full"
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "IF.npz", labels=np.asarray(masks).astype(np.int32), um_per_px=meta["um_per_px"], origin_um=np.asarray(meta["origin_um"]))
        log["runs"].append(rec)
        print(rec, flush=True)
    (HERE / "runs_S1_full.json").write_text(json.dumps(log, indent=1))


def export():
    sys.path.insert(0, str(HERE))
    import evaluate as E
    import pandas as pd
    E.DESC[METHOD] = "Cellpose-SAM (cpsam), IF G+R membrane proxy, whole IF section (0.5 um/px)"
    qual = {}
    for s in SAMPLES:
        S = E.catalog.load_manifest()[s]
        c = E.Crop(s, "full", "IF", S)
        lab = np.load(HERE / "cache/seg" / METHOD / s / "full" / "IF.npz")["labels"]
        t = time.time()
        cells, nucs = E.associate(c, lab, METHOD)
        feats, rows = [], []
        for x in cells:
            rc = E.polygon_of(x["image"])
            if rc is None:
                continue
            poly = c.to_world(rc + np.array([x["slice"][0].start, x["slice"][1].start]))
            feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [np.round(np.vstack([poly, poly[:1]]), 2).tolist()]},
                          "properties": {"cell_segment_id": f"full:{x['label']}", "segmentation_method": METHOD, "segmentation_version": E.VERSION,
                                         "roi": "full", "roi_role": "full_section", "area_um2": round(x["area_um2"], 3), "perimeter_um": round(x["perimeter_um"], 3),
                                         "centroid_x": round(x["centroid"][0], 3), "centroid_y": round(x["centroid"][1], 3), "solidity": round(x["solidity"], 4),
                                         "eccentricity": round(x["eccentricity"], 4), "major_um": round(x["major_um"], 3), "minor_um": round(x["minor_um"], 3),
                                         "nucleus_ids": [E.NC._key(c.nuc_ids[n]).strip("'") for n in x["nuclei"]], "nucleus_indices": x["nuclei"],
                                         "cell_relation": x["relation"], "quality_flags": x["flags"], "nucleus_area_um2": round(x["nucleus_area_um2"], 3)}})
        bylab = {x["label"]: x for x in cells}
        for n in nucs:
            x = bylab.get(n["cell_label"]) if n["cell_label"] is not None else None
            rows.append({"nucleus_id": n["nucleus_id"], "nucleus_index": n["nucleus_index"], "cell_segment_id": f"full:{n['cell_label']}" if x else None,
                         "segmentation_method": METHOD, "segmentation_version": E.VERSION, "roi": "full", "roi_role": "full_section",
                         "nucleus_centroid_x": c.nuc_pts[n["nucleus_index"], 0], "nucleus_centroid_y": c.nuc_pts[n["nucleus_index"], 1],
                         "cell_area_um2": x["area_um2"] if x else None, "cell_perimeter_um": x["perimeter_um"] if x else None,
                         "cell_centroid_x": x["centroid"][0] if x else None, "cell_centroid_y": x["centroid"][1] if x else None,
                         "nucleus_inside_cell": n["nucleus_inside_cell"], "relation": n["relation"], "cell_relation": x["relation"] if x else None,
                         "competing_frac": round(n["competing_frac"], 3), "segmentation_quality_flags": ";".join(x["flags"]) if x else "no_cell"})
        d = HERE / "results" / METHOD / s
        d.mkdir(parents=True, exist_ok=True)
        (d / "IF.geojson").write_text(json.dumps({"type": "FeatureCollection", "properties": {
            "segmentation_method": METHOD, "segmentation_version": E.VERSION, "description": E.DESC[METHOD], "sample_id": s, "section": "IF",
            "frame": "IF section frame, um (same as the nuclei)", "rois": {"full": {"roi_if": [0, 0, *S.modality("IF").extent()], "role": "full_section"}},
            "is_whole_cell_detection": True, "note": "no ground truth; partial success only (see REPORT); not pathological cell boundaries"},
            "features": feats}))
        nr = pd.DataFrame(rows)
        nr.to_csv(d / "IF_nucleus_cell.csv", index=False)
        inner = [x for x in cells if "touches_roi_border" not in x["flags"]]
        qual[s] = {"n_cells": len(cells), "n_nuclei": len(nucs), "area_um2_median": float(np.median([x["area_um2"] for x in inner])),
                   "relations_nuclei": nr["relation"].value_counts().to_dict(),
                   "relations_cells": pd.Series([x["relation"] for x in cells]).value_counts().to_dict(),
                   "frac_nucleus_sized": float(np.mean(["nucleus_sized" in x["flags"] for x in cells])),
                   "tissue_covered_by_cells": float(((lab > 0) & c.tissue).sum() / max(c.tissue.sum(), 1)),
                   "boundary_contrast": E.boundary_contrast(c, lab), "export_seconds": time.time() - t}
        print(s, qual[s], flush=True)
    (HERE / "full_section_quality.json").write_text(json.dumps(qual, indent=1))


if __name__ == "__main__":
    {"crop": crop, "seg": seg, "export": export}[sys.argv[1]]()
