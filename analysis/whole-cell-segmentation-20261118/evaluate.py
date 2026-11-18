"""S3 (nucleus-seeded watershed / Voronoi baseline), nucleus-cell association, quality metrics, exports and figures (env: cell-rg).

Methods compared (each in its own section frame; crops at 0.5 um/px from roi_select.py):
  S0_nuclei            existing nuclear contours (StarDist H&E, provider IF) -- reference, NOT whole-cell
  S1_cpsam_he_rgb      Cellpose-SAM, H&E RGB as is                      (default input)
  S1_cpsam_he_eosin    Cellpose-SAM, H&E eosin OD only                  (tuned on tune_* ROIs)
  S1_cpsam_if_rgb      Cellpose-SAM, IF display RGB as is               (default input)
  S1_cpsam_if_mem      Cellpose-SAM, IF G+R membrane proxy only         (tuned on tune_* ROIs)
  S2_mesmer            Mesmer IF (only if it ran; see runs_S2_all.json)
  S3_watershed         existing nuclei as seeds, watershed on a boundary-signal image inside the tissue mask, <= 25 um from seed
  S3_voronoi           existing nuclei as seeds, distance only (expand_labels) inside the tissue mask, <= 25 um -- BASELINE,
                       not based on any membrane; never counted as a detected cell boundary

No ground-truth whole-cell outlines exist.  All metrics here are structural / image-consistency proxies, NOT accuracy:
  boundary_contrast = mean boundary-signal on predicted cell borders / mean inside cells
                      (IF: G+R; H&E: 1 - eosin OD, i.e. pale lines between eosinophilic cytoplasm)
  nucleus_sized     = share of cells whose area <= 1.5 x the area of their (single) nucleus  -> nucleus-only segmentation
Outputs: results/<method>/<sample>/<key>.geojson, results/<method>/<sample>/<key>_nucleus_cell.csv,
         quality_by_roi.csv, quality_summary.csv, figures/*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.color import rgb2hed
from skimage.filters import gaussian
from skimage.measure import find_contours, regionprops
from skimage.segmentation import expand_labels, find_boundaries, watershed

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from src.registration_explorer import catalog  # noqa: E402
from src.registration_research import nuclear_contours as NC  # noqa: E402

VERSION = "wcs-20261118.1"
MAX_GROW_UM = 25.0
COMPETE_FRAC = 0.25
NUC_SIZED = 1.5
SIMPLIFY_PX = 0.4
S1 = ["S1_cpsam_he_rgb", "S1_cpsam_he_eosin", "S1_cpsam_if_rgb", "S1_cpsam_if_mem"]
METHODS = ["S0_nuclei"] + S1 + ["S2_mesmer", "S3_watershed", "S3_voronoi"]
DESC = {"S0_nuclei": "existing nuclear contours (not whole-cell)", "S1_cpsam_he_rgb": "Cellpose-SAM (cpsam), H&E RGB",
        "S1_cpsam_he_eosin": "Cellpose-SAM (cpsam), H&E eosin OD", "S1_cpsam_if_rgb": "Cellpose-SAM (cpsam), IF display RGB",
        "S1_cpsam_if_mem": "Cellpose-SAM (cpsam), IF G+R membrane proxy", "S2_mesmer": "Mesmer whole-cell (IF proxies)",
        "S3_watershed": "nucleus-seeded watershed on boundary signal", "S3_voronoi": "nucleus-seeded Voronoi (baseline, image-agnostic)"}


def applies(method, key):
    if method in ("S0_nuclei", "S3_watershed", "S3_voronoi"):
        return True
    if method == "S2_mesmer":
        return key == "IF"
    return ("_he_" in method) == (key == "HE")


# ----------------------------------------------------------------------------- crop context
class Crop:
    def __init__(self, sample, roi, key, S):
        self.sample, self.roi, self.key = sample, roi, key
        m = json.loads((HERE / "cache/crops" / sample / roi / "crop_meta.json").read_text())[key]
        self.um = float(m["um_per_px"]); self.origin = np.asarray(m["origin_um"], float)
        self.rgb = np.asarray(Image.open(HERE / "cache/crops" / sample / roi / f"{key}.png").convert("RGB"))
        self.h, self.w = self.rgb.shape[:2]
        mod = S.modality(key)
        pts, ids = catalog.load_nuclei(mod)
        self.cont = NC.load_contours(mod, pts, ids)
        x0, y0 = self.origin; x1, y1 = x0 + self.w * self.um, y0 + self.h * self.um
        inside = (pts[:, 0] >= x0) & (pts[:, 0] < x1) & (pts[:, 1] >= y0) & (pts[:, 1] < y1)
        self.nuc_idx = np.flatnonzero(inside)
        self.nuc_pts, self.nuc_ids = pts, ids
        # nucleus label image (value = position in nuc_idx + 1); polygons rasterised on the crop grid
        im = Image.new("I", (self.w, self.h), 0)
        dr = ImageDraw.Draw(im)
        near = self.cont.query([x0 - 20, y0 - 20, x1 + 20, y1 + 20]) if self.cont is not None else []
        self.nuc_near = [int(i) for i in near]
        self.nuc_lab = np.zeros((self.h, self.w), np.int32)
        for i in self.nuc_near:
            p = self.cont.polygon(i)
            if p is None:
                continue
            q = (p - self.origin) / self.um
            dr.polygon([tuple(v) for v in q], fill=int(i) + 1)
        self.nuc_lab = np.asarray(im, np.int32)            # value = global nucleus index + 1
        rgbf = self.rgb.astype(np.float32)
        if key == "IF":
            self.signal = gaussian((rgbf[..., 1] + rgbf[..., 0]) / 510.0, 1.0)
            self.tissue = gaussian((rgbf.max(2) > 20).astype(float), 8) > 0.08
        else:
            e = rgb2hed(self.rgb)[..., 1]
            e = (e - np.percentile(e, 1)) / max(np.percentile(e, 99) - np.percentile(e, 1), 1e-6)
            self.signal = gaussian(1 - np.clip(e, 0, 1), 1.0)
            s = (rgbf.max(2) - rgbf.min(2)) / np.maximum(rgbf.max(2), 1)
            self.tissue = gaussian(((s > 0.12) & (rgbf.mean(2) < 235)).astype(float), 4) > 0.3

    def to_world(self, rc):
        """find_contours (row, col) on the pixel grid -> section frame um (pixel centre at (col + 0.5) * um)."""
        return np.column_stack([self.origin[0] + (rc[:, 1] + 0.5) * self.um, self.origin[1] + (rc[:, 0] + 0.5) * self.um])


def run_s3(c: Crop, kind: str) -> np.ndarray:
    seeds = c.nuc_lab.copy()
    seeds[~c.tissue] = 0
    grow = int(round(MAX_GROW_UM / c.um))
    dist_ok = ndi.distance_transform_edt(seeds == 0) <= grow
    mask = c.tissue & dist_ok
    if kind == "voronoi":
        lab = expand_labels(seeds, distance=grow)
    else:
        lab = watershed(c.signal, markers=seeds, mask=mask)
    lab[~mask] = 0
    return lab.astype(np.int32)


def load_labels(method, c: Crop):
    if method == "S0_nuclei":
        return c.nuc_lab
    if method.startswith("S3_"):
        return run_s3(c, method.split("_", 1)[1])
    p = HERE / "cache/seg" / method / c.sample / c.roi / f"{c.key}.npz"
    if not p.is_file():
        return None
    return np.load(p)["labels"].astype(np.int32)


def polygon_of(mask: np.ndarray):
    cs = find_contours(np.pad(mask.astype(float), 1), 0.5)
    if not cs:
        return None
    rc = max(cs, key=len) - 1
    from skimage.measure import approximate_polygon
    return approximate_polygon(rc, SIMPLIFY_PX)           # <= 0.4 px (0.2 um) deviation; keeps exports small


def associate(c: Crop, lab: np.ndarray, method: str):
    """Nucleus <-> cell relations (never forced to 1:1). Returns (cells list, nucleus rows)."""
    nuc_rows = []
    cell_nuclei = {}
    sl = ndi.find_objects(c.nuc_lab)                     # bounding slices per nucleus label (fast at full-section size)
    nuc_area = {}
    for i in c.nuc_idx:
        s_ = sl[i] if i < len(sl) else None
        if s_ is None:
            m, sub = np.zeros((1, 1), bool), np.zeros((1, 1), np.int32)
        else:
            m, sub = c.nuc_lab[s_] == (i + 1), lab[s_]
        area_n = int(m.sum())
        nuc_area[int(i)] = area_n
        cx, cy = ((c.nuc_pts[i] - c.origin) / c.um).astype(int)
        cx, cy = min(max(cx, 0), c.w - 1), min(max(cy, 0), c.h - 1)
        centre_cell = int(lab[cy, cx])
        vals, cnt = np.unique(sub[m], return_counts=True) if area_n else (np.array([centre_cell]), np.array([1]))
        ov = {int(v): int(k) for v, k in zip(vals, cnt) if v != 0}
        tot = max(area_n, 1)
        frac = sorted((k / tot for k in ov.values()), reverse=True)
        if not ov and centre_cell == 0:
            rel = "nucleus_without_cell"
            cell = None
        elif centre_cell == 0:
            rel = "nucleus_outside_cell"
            cell = max(ov, key=ov.get)
        elif len(frac) >= 2 and frac[1] >= COMPETE_FRAC:
            rel = "nucleus_competing_cells"
            cell = centre_cell
        else:
            rel = "assigned"
            cell = centre_cell
        if cell is not None and rel != "nucleus_outside_cell":
            cell_nuclei.setdefault(cell, []).append(int(i))
        nuc_rows.append({"nucleus_index": int(i), "nucleus_id": NC._key(c.nuc_ids[i]).strip("'"), "cell_label": cell, "relation": rel,
                         "nucleus_inside_cell": bool(centre_cell != 0 and centre_cell == cell), "nucleus_area_px": area_n,
                         "overlap_cells": len(ov), "competing_frac": frac[1] if len(frac) >= 2 else 0.0})
    cells = []
    for rp in regionprops(lab):
        L = int(rp.label)
        r0, c0, r1, c1 = rp.bbox
        border = bool(r0 == 0 or c0 == 0 or r1 == lab.shape[0] or c1 == lab.shape[1])
        nucs = cell_nuclei.get(L, [])
        rel = "cell_without_nucleus" if not nucs else ("one_nucleus" if len(nucs) == 1 else "multi_nucleus")
        nuc_area_px = sum(nuc_area.get(n, 0) for n in nucs) if nucs else 0
        flags = []
        if border:
            flags.append("touches_roi_border")
        if len(nucs) == 1 and nuc_area_px and rp.area <= NUC_SIZED * nuc_area_px:
            flags.append("nucleus_sized")
        if rp.solidity < 0.7:
            flags.append("low_solidity")
        cells.append({"label": L, "area_um2": rp.area * c.um ** 2, "perimeter_um": rp.perimeter * c.um,
                      "centroid": [c.origin[0] + (rp.centroid[1] + 0.5) * c.um, c.origin[1] + (rp.centroid[0] + 0.5) * c.um],
                      "solidity": float(rp.solidity), "eccentricity": float(rp.eccentricity),
                      "major_um": rp.axis_major_length * c.um, "minor_um": rp.axis_minor_length * c.um,
                      "nuclei": nucs, "relation": rel, "flags": flags, "nucleus_area_um2": nuc_area_px * c.um ** 2,
                      "slice": rp.slice, "image": rp.image})
    return cells, nuc_rows


def boundary_contrast(c: Crop, lab):
    if lab is None or lab.max() == 0:
        return np.nan
    b = find_boundaries(lab, mode="inner") & (lab > 0)
    inner = (lab > 0) & ~ndi.binary_dilation(b, iterations=2)
    if b.sum() < 10 or inner.sum() < 10:
        return np.nan
    return float(c.signal[b].mean() / max(c.signal[inner].mean(), 1e-6))


def main():
    man = catalog.load_manifest()
    rois = json.loads((HERE / "rois.json").read_text())
    s2 = json.loads((HERE / "runs_S2_all.json").read_text()) if (HERE / "runs_S2_all.json").is_file() else {}
    qrows, exports = [], {}
    figs = HERE / "figures"; figs.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for sample, sv in rois["samples"].items():
        S = man[sample]
        for roi, r in sv["rois"].items():
            if r is None:
                continue
            for key in ("HE", "IF"):
                c = Crop(sample, roi, key, S)
                ms = [m for m in METHODS if applies(m, key)]
                fig, axs = plt.subplots(1, len(ms), figsize=(4.2 * len(ms), 4.6))
                for ax, method in zip(axs, ms):
                    lab = load_labels(method, c)
                    img = c.rgb.copy()
                    if lab is None:
                        reason = s2.get("model_error", "not run") if method == "S2_mesmer" else "not run"
                        qrows.append({"sample": sample, "roi": roi, "role": r["role"], "section": key, "method": method, "available": False,
                                      "reason": str(reason)[:200]})
                        ax.imshow(img); ax.set_title(f"{method}\nunavailable", fontsize=8); ax.axis("off")
                        continue
                    cells, nucs = associate(c, lab, method)
                    nb = find_boundaries(c.nuc_lab, mode="inner")
                    cb = find_boundaries(lab, mode="inner")
                    img[nb] = [0, 255, 255]
                    if method != "S0_nuclei":
                        img[cb] = [255, 235, 0]
                    ax.imshow(img); ax.axis("off")
                    inner = [x for x in cells if "touches_roi_border" not in x["flags"]]
                    nr = pd.DataFrame(nucs)
                    q = {"sample": sample, "roi": roi, "role": r["role"], "section": key, "method": method, "available": True,
                         "n_cells": len(cells), "n_cells_inner": len(inner), "n_nuclei_in_roi": len(nucs),
                         "area_um2_median": float(np.median([x["area_um2"] for x in inner])) if inner else np.nan,
                         "area_um2_p10": float(np.percentile([x["area_um2"] for x in inner], 10)) if inner else np.nan,
                         "area_um2_p90": float(np.percentile([x["area_um2"] for x in inner], 90)) if inner else np.nan,
                         "solidity_median": float(np.median([x["solidity"] for x in inner])) if inner else np.nan,
                         "frac_nucleus_sized": float(np.mean(["nucleus_sized" in x["flags"] for x in cells])) if cells else np.nan,
                         "cells_one_nucleus": sum(x["relation"] == "one_nucleus" for x in cells),
                         "cells_multi_nucleus": sum(x["relation"] == "multi_nucleus" for x in cells),
                         "cells_without_nucleus": sum(x["relation"] == "cell_without_nucleus" for x in cells),
                         "nuclei_assigned": int((nr.relation == "assigned").sum()) if len(nr) else 0,
                         "nuclei_competing": int((nr.relation == "nucleus_competing_cells").sum()) if len(nr) else 0,
                         "nuclei_outside_cell": int((nr.relation == "nucleus_outside_cell").sum()) if len(nr) else 0,
                         "nuclei_without_cell": int((nr.relation == "nucleus_without_cell").sum()) if len(nr) else 0,
                         "tissue_covered_by_cells": float(((lab > 0) & c.tissue).sum() / max(c.tissue.sum(), 1)),
                         "boundary_contrast": boundary_contrast(c, lab)}
                    qrows.append(q)
                    ax.set_title(f"{method}\ncells {len(cells)} nuc-sized {q['frac_nucleus_sized']:.2f} bc {q['boundary_contrast']:.2f}", fontsize=8)
                    if method == "S0_nuclei":
                        continue
                    ex = exports.setdefault((method, sample, key), {"features": [], "rows": []})
                    for x in cells:
                        rc = polygon_of(x["image"])
                        if rc is None:
                            continue
                        rc = rc + np.array([x["slice"][0].start, x["slice"][1].start])
                        poly = c.to_world(rc)
                        cid = f"{roi}:{x['label']}"
                        ex["features"].append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [np.round(np.vstack([poly, poly[:1]]), 2).tolist()]},
                                               "properties": {"cell_segment_id": cid, "segmentation_method": method, "segmentation_version": VERSION,
                                                              "roi": roi, "roi_role": r["role"], "area_um2": round(x["area_um2"], 3),
                                                              "perimeter_um": round(x["perimeter_um"], 3), "centroid_x": round(x["centroid"][0], 3),
                                                              "centroid_y": round(x["centroid"][1], 3), "solidity": round(x["solidity"], 4),
                                                              "eccentricity": round(x["eccentricity"], 4), "major_um": round(x["major_um"], 3),
                                                              "minor_um": round(x["minor_um"], 3),
                                                              "nucleus_ids": [NC._key(c.nuc_ids[n]).strip("'") for n in x["nuclei"]],
                                                              "nucleus_indices": x["nuclei"], "cell_relation": x["relation"], "quality_flags": x["flags"],
                                                              "nucleus_area_um2": round(x["nucleus_area_um2"], 3)}})
                    bylab = {x["label"]: x for x in cells}
                    for n in nucs:
                        x = bylab.get(n["cell_label"]) if n["cell_label"] is not None else None
                        npoly = c.cont.polygon(n["nucleus_index"]) if c.cont is not None else None
                        ex["rows"].append({"nucleus_id": n["nucleus_id"], "nucleus_index": n["nucleus_index"],
                                           "cell_segment_id": f"{roi}:{n['cell_label']}" if x else None, "segmentation_method": method,
                                           "segmentation_version": VERSION, "roi": roi, "roi_role": r["role"],
                                           "nucleus_centroid_x": c.nuc_pts[n["nucleus_index"], 0], "nucleus_centroid_y": c.nuc_pts[n["nucleus_index"], 1],
                                           "nucleus_polygon": json.dumps(np.round(npoly, 2).tolist()) if npoly is not None else None,
                                           "cell_area_um2": x["area_um2"] if x else None, "cell_perimeter_um": x["perimeter_um"] if x else None,
                                           "cell_centroid_x": x["centroid"][0] if x else None, "cell_centroid_y": x["centroid"][1] if x else None,
                                           "nucleus_inside_cell": n["nucleus_inside_cell"], "relation": n["relation"],
                                           "cell_relation": x["relation"] if x else None, "competing_frac": round(n["competing_frac"], 3),
                                           "segmentation_quality_flags": ";".join(x["flags"]) if x else "no_cell"})
                fig.suptitle(f"{sample} {roi} ({r['role']}) {key} — cyan = existing nuclei, yellow = cell outlines (no ground truth)", fontsize=9)
                fig.tight_layout()
                fig.savefig(figs / f"cmp_{sample}_{roi}_{key}.png", dpi=70)
                plt.close(fig)
    for (method, sample, key), ex in exports.items():
        d = HERE / "results" / method / sample
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{key}.geojson").write_text(json.dumps({"type": "FeatureCollection", "properties": {
            "segmentation_method": method, "segmentation_version": VERSION, "description": DESC[method], "sample_id": sample, "section": key,
            "frame": f"{key} section frame, um (same as the nuclei)", "rois": {k: v for k, v in rois["samples"][sample]["rois"].items() if v},
            "is_whole_cell_detection": method not in ("S3_voronoi",), "note": "no ground truth; not pathological cell boundaries"},
            "features": ex["features"]}))
        pd.DataFrame(ex["rows"]).to_csv(d / f"{key}_nucleus_cell.csv", index=False)
    q = pd.DataFrame(qrows)
    q.to_csv(HERE / "quality_by_roi.csv", index=False)
    ev = q[(q.role == "evaluation") & (q.available == True)]  # noqa: E712
    summ = ev.groupby(["sample", "section", "method"]).agg(
        rois=("roi", "count"), n_cells=("n_cells", "sum"), area_um2_median=("area_um2_median", "median"),
        frac_nucleus_sized=("frac_nucleus_sized", "mean"), boundary_contrast=("boundary_contrast", "median"),
        cells_multi_nucleus=("cells_multi_nucleus", "sum"), cells_without_nucleus=("cells_without_nucleus", "sum"),
        nuclei_assigned=("nuclei_assigned", "sum"), nuclei_competing=("nuclei_competing", "sum"),
        nuclei_outside_cell=("nuclei_outside_cell", "sum"), nuclei_without_cell=("nuclei_without_cell", "sum"),
        tissue_covered=("tissue_covered_by_cells", "mean")).reset_index()
    summ.to_csv(HERE / "quality_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
