"""Phase 5: whole-cell features that could later help correspondence (env: cell-rg).  Graph / UOT / B2 are NOT changed.

Per IF cell of S1_cpsam_if_mem_full (the only method that gave plausible whole-cell outlines, IF only):
  area, perimeter, aspect ratio (major/minor), solidity, eccentricity, nucleus/cell area ratio, nucleus-centre offset
  (|nucleus centroid - cell centroid| / equivalent radius), nearest-cell-centroid distance, number of touching cells (contact).
Then, for each H&E nucleus of the latest Cell Review auto result, the IF candidates' cells are compared:
  algorithm top-1 candidate vs the other candidates (a PROXY for "confirmed vs rejected": no human reviews exist yet).
H&E has no usable whole-cell outlines (REPORT §4), so cross-section cell-to-cell features cannot be computed.
Outputs: features_if_cells.csv, features_candidates.csv, features_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.measure import regionprops

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from src.registration_explorer import catalog  # noqa: E402
from src.registration_research import cell_review as CR  # noqa: E402
from src.registration_research import cell_segmentation as CS  # noqa: E402

METHOD = "S1_cpsam_if_mem_full"


def contacts(lab):
    """{label: number of distinct touching labels} (4-neighbourhood, background ignored)."""
    pairs = set()
    for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1, :], lab[1:, :])):
        m = (a != b) & (a > 0) & (b > 0)
        for x, y in zip(a[m], b[m]):
            pairs.add((min(x, y), max(x, y)))
    n = {}
    for x, y in pairs:
        n[x] = n.get(x, 0) + 1
        n[y] = n.get(y, 0) + 1
    return n


def main():
    man = catalog.load_manifest()
    rows, crows, summ = [], [], {}
    for s in ("031-6_2", "031-8_6"):
        S = man[s]
        pts, ids = catalog.load_nuclei(S.modality("IF"))
        seg = CS.load(s, "IF", METHOD, ids)
        lab = np.load(HERE / "cache/seg" / METHOD / s / "full" / "IF.npz")
        L, um = lab["labels"], float(lab["um_per_px"])
        con = contacts(L)
        rp = {int(r.label): r for r in regionprops(L)}
        cen = np.array([seg.centroid(j) for j in range(seg.n)])
        tree = cKDTree(cen)
        dnn = tree.query(cen, k=2)[0][:, 1]
        for j in range(seg.n):
            p = seg.props[j]
            lbl = int(str(seg.ids[j]).split(":")[-1])
            r = rp[lbl]
            nuc = seg.nuclei_of_cell[j]
            req = np.sqrt(p["area_um2"] / np.pi)
            off = float(np.min(np.linalg.norm(pts[nuc] - cen[j], axis=1)) / req) if len(nuc) else np.nan
            rows.append({"sample_id": s, "cell_segment_id": seg.ids[j], "area_um2": p["area_um2"], "perimeter_um": p["perimeter_um"],
                         "aspect_ratio": r.axis_major_length / max(r.axis_minor_length, 1e-6), "solidity": r.solidity, "eccentricity": r.eccentricity,
                         "n_nuclei": len(nuc), "nucleus_cell_area_ratio": p["nucleus_area_um2"] / p["area_um2"] if len(nuc) else np.nan,
                         "nucleus_offset_rel": off, "nn_cell_distance_um": float(dnn[j]), "n_contacts": con.get(lbl, 0),
                         "cell_relation": p["cell_relation"], "flags": ";".join(p["quality_flags"])})
        autos = CR.list_autos(s)
        if not autos:
            summ[s] = {"auto": None}
            continue
        auto = CR.load_auto(autos[-1])
        fr = pd.DataFrame([r for r in rows if r["sample_id"] == s]).set_index("cell_segment_id")
        cand = auto.candidates
        for r in cand.itertuples(index=False):
            cells = seg.cells_for(int(r.if_index))
            f = fr.loc[seg.ids[cells[0]]] if len(cells) == 1 else None
            crows.append({"sample_id": s, "auto_id": auto.auto_id, "he_index": int(r.he_index), "if_index": int(r.if_index), "rank": int(r.rank),
                          "is_top1": int(r.rank) == 1, "n_cells_of_if_nucleus": len(cells),
                          **({k: f[k] for k in ("area_um2", "aspect_ratio", "solidity", "nucleus_cell_area_ratio", "nucleus_offset_rel",
                                                 "nn_cell_distance_um", "n_contacts", "n_nuclei")} if f is not None else {})})
        summ[s] = {"auto": auto.auto_id, "n_cells": seg.n}
    fc = pd.DataFrame(rows)
    fc.to_csv(HERE / "features_if_cells.csv", index=False)
    cc = pd.DataFrame(crows)
    cc.to_csv(HERE / "features_candidates.csv", index=False)
    out = {"note": "top-1 vs other candidates is a proxy (no human-confirmed pairs yet); differences do not show usefulness for matching",
           "samples": summ, "cells": {}, "candidates": {}}
    feats = ["area_um2", "aspect_ratio", "solidity", "nucleus_cell_area_ratio", "nucleus_offset_rel", "nn_cell_distance_um", "n_contacts"]
    for s, g in fc.groupby("sample_id"):
        inner = g[~g["flags"].str.contains("touches_roi_border", na=False)]
        out["cells"][s] = {k: {"median": float(inner[k].median()), "p10": float(inner[k].quantile(0.1)), "p90": float(inner[k].quantile(0.9))} for k in feats}
        out["cells"][s]["n"] = int(len(inner))
    if len(cc):
        for s, g in cc.groupby("sample_id"):
            d = {"n_candidates": int(len(g)), "share_with_one_cell": float((g["n_cells_of_if_nucleus"] == 1).mean()),
                 "share_with_no_cell": float((g["n_cells_of_if_nucleus"] == 0).mean())}
            for k in feats:
                if k in g:
                    a, b = g[g.is_top1][k].dropna(), g[~g.is_top1][k].dropna()
                    d[k] = {"top1_median": float(a.median()) if len(a) else None, "others_median": float(b.median()) if len(b) else None,
                            "n_top1": int(len(a)), "n_others": int(len(b))}
            out["candidates"][s] = d
    (HERE / "features_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:4000])


if __name__ == "__main__":
    main()
