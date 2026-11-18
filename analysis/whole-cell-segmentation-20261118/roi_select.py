"""Phase 2: reproducible ROI selection + crops for the whole-cell segmentation comparison (env: cell-rg).

ROIs are squares of ROI_UM in the IF frame (fixed); the H&E ROI is the same region mapped back by the Global affine
(the stored WF result, as in Cell Review) -- each section is cropped in its own, undeformed frame.

Per sliding window (stride STRIDE_UM) we compute, on a low-resolution level:
  n_if, n_he         nuclei in the window (IF; H&E mapped)
  tissue_if/he       tissue fraction (IF: any channel above background; H&E: saturation)
  ridge_if           boundary clarity proxy (IF): mean Sato ridge response of G (Dystrophin, display RGB) inside tissue
  ridge_he           boundary clarity proxy (H&E): mean Sato ridge response of the eosin-inverted gray inside tissue
  n_he_suspect       StarDist nuclei with weak hematoxylin contrast (mean H-OD inside / 2 um ring < SUSPECT_RATIO)
Categories (each = argmax/argmin under constraints, no overlap with earlier picks):
  dense, sparse, clear_boundary, unclear_boundary, tissue_edge, stardist_suspect  -> evaluation ROIs
  tune_1, tune_2 (random, seed 0, tissue >= 0.8)                                  -> tuning ROIs (parameters only)
Outputs: rois.json (criteria + coordinates), cache/crops/<sample>/<roi>/{HE,IF}.png + crop_meta.json (0.5 um/px).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from skimage.color import rgb2hed
from skimage.filters import sato
from skimage.transform import resize

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from src.registration_explorer import catalog, imaging, results  # noqa: E402
from src.registration_research import cell_review as CR  # noqa: E402
from src.registration_research import nuclear_contours as NC  # noqa: E402

ROI_UM, STRIDE_UM, WORK_UM, LOW_UM = 150.0, 50.0, 0.5, 2.0
SUSPECT_RATIO = 1.15
TISSUE_IF, TISSUE_HE = 0.7, 0.6           # "tissue-covered" window
CATS = ("dense", "sparse", "clear_boundary", "unclear_boundary", "tissue_edge", "stardist_suspect")


def level_image(mod, target_um):
    info = imaging.pyramid_info(mod.image_path)
    lvl = imaging.pick_level(info, mod.pixel_um, target_um)
    h, w = info.level_shapes[lvl][:2]
    img = imaging.read_region(mod.image_path, lvl, 0, 0, w, h)          # tile reads (no whole-file load)
    return img, mod.pixel_um * info.downsamples[lvl], lvl


def crop_world(mod, x0, y0, x1, y1, out_um):
    """Section-frame crop [x0,x1]x[y0,y1] (um) resampled to out_um/px; pixel (0,0) top-left corner at (x0, y0)."""
    info = imaging.pyramid_info(mod.image_path)
    lvl = imaging.pick_level(info, mod.pixel_um, out_um)
    um = mod.pixel_um * info.downsamples[lvl]
    c = imaging.level_corner_um(mod.pixel_um)
    lo = np.floor((np.array([x0, y0]) - c) / um).astype(int)
    hi = np.ceil((np.array([x1, y1]) - c) / um).astype(int)
    raw = imaging.read_region(mod.image_path, lvl, lo[0], lo[1], hi[0], hi[1])
    raw = raw[..., :3] if raw.ndim == 3 else np.repeat(raw[..., None], 3, 2)
    # exact placement: resample the read block onto the output grid (pixel corners at x0 + k * out_um)
    n = int(round((x1 - x0) / out_um)), int(round((y1 - y0) / out_um))
    gx = (x0 + (np.arange(n[0]) + 0.5) * out_um - (c + lo[0] * um)) / um - 0.5
    gy = (y0 + (np.arange(n[1]) + 0.5) * out_um - (c + lo[1] * um)) / um - 0.5
    from scipy.ndimage import map_coordinates
    yy, xx = np.meshgrid(gy, gx, indexing="ij")
    out = np.stack([map_coordinates(raw[..., k].astype(np.float32), [yy, xx], order=1, mode="nearest") for k in range(3)], -1)
    valid = (xx >= -0.5) & (yy >= -0.5) & (xx <= raw.shape[1] - 0.5) & (yy <= raw.shape[0] - 0.5)
    return np.clip(out, 0, 255).astype(np.uint8), valid, {"level": int(lvl), "level_um": float(um)}


def main():
    out = {"roi_um": ROI_UM, "stride_um": STRIDE_UM, "work_um_per_px": WORK_UM, "suspect_ratio": SUSPECT_RATIO, "tissue_if_min": TISSUE_IF, "tissue_he_min": TISSUE_HE, "samples": {}}
    for sample in ("031-6_2", "031-8_6"):
        S = catalog.load_manifest()[sample]
        he_m, if_m = S.modality("HE"), S.modality("IF")
        recs = [r for r in results.load_records(S) if r.M is not None]
        rec = next((r for r in recs if r.method == "WF"), recs[0])
        M, t, _ = results.oriented(rec.M, rec.t, rec.fixed, "IF")
        Mi, ti = CR.inverse_affine(M, t)
        he, he_ids = catalog.load_nuclei(he_m)
        iff, _ = catalog.load_nuclei(if_m)
        hc = NC.load_contours(he_m, he, he_ids)
        mapped = CR.map_points(M, t, he)
        # low-resolution maps
        ifl, if_um, _ = level_image(if_m, LOW_UM)
        hel, he_um, _ = level_image(he_m, LOW_UM)
        ifl = ifl[..., :3].astype(float); hel = hel[..., :3].astype(float)
        from scipy.ndimage import gaussian_filter      # IF cell interiors are dark (only outlines/nuclei bright): smoothed occupancy
        tis_if = gaussian_filter((ifl.max(2) > 20).astype(float), 4.0) > 0.08
        hsv_s = (hel.max(2) - hel.min(2)) / np.maximum(hel.max(2), 1)
        tis_he = (hsv_s > 0.12) & (hel.mean(2) < 235)
        r_if = sato(ifl[..., 1] / 255.0, sigmas=[1.0, 2.0], black_ridges=False)
        r_he = sato(1 - hel.mean(2) / 255.0, sigmas=[1.0, 2.0], black_ridges=True)
        # StarDist suspects: weak hematoxylin contrast (full-res crops per nucleus would be slow -> level ~0.4 um)
        hl, hl_um, _ = level_image(he_m, 0.4)
        hod = rgb2hed(hl[..., :3])[..., 0]
        cc = imaging.level_corner_um(he_m.pixel_um)
        suspect = np.zeros(len(he), bool); ratio = np.full(len(he), np.nan)
        from matplotlib.path import Path as MPath
        for i in range(len(he)):
            p = hc.polygon(i)
            if p is None:
                continue
            lo = np.floor((p.min(0) - 2.5 - cc) / hl_um).astype(int); hi = np.ceil((p.max(0) + 2.5 - cc) / hl_um).astype(int)
            lo = np.maximum(lo, 0); hi = np.minimum(hi, [hod.shape[1], hod.shape[0]])
            if np.any(hi - lo < 3):
                continue
            yy, xx = np.mgrid[lo[1]:hi[1], lo[0]:hi[0]]
            w = np.column_stack([cc + (xx.ravel() + 0.5) * hl_um, cc + (yy.ravel() + 0.5) * hl_um])
            inside = MPath(p).contains_points(w)
            d = np.min(np.linalg.norm(w[:, None, :] - p[None, :, :], axis=2), axis=1)
            ring = (~inside) & (d <= 2.0)
            v = hod[lo[1]:hi[1], lo[0]:hi[0]].ravel()
            if inside.sum() and ring.sum():
                ratio[i] = (v[inside].mean() + 1e-3) / (v[ring].mean() + 1e-3)
        thr = max(SUSPECT_RATIO, float(np.nanpercentile(ratio, 10)))   # relative rule: weakest 10 % of hematoxylin contrast
        suspect = ratio < thr
        fe = if_m.extent()
        rows = []
        for x0 in np.arange(0, fe[0] - ROI_UM + 1e-6, STRIDE_UM):
            for y0 in np.arange(0, fe[1] - ROI_UM + 1e-6, STRIDE_UM):
                roi = [x0, y0, x0 + ROI_UM, y0 + ROI_UM]
                sl = (slice(int(y0 / if_um), int((y0 + ROI_UM) / if_um)), slice(int(x0 / if_um), int((x0 + ROI_UM) / if_um)))
                ti_ = tis_if[sl]
                # H&E window = bbox of the inverse-mapped IF square
                q = CR.map_points(Mi, ti, np.array([[roi[0], roi[1]], [roi[2], roi[1]], [roi[2], roi[3]], [roi[0], roi[3]]]))
                hx0, hy0 = q.min(0); hx1, hy1 = q.max(0)
                he_ext = he_m.extent()
                if hx0 < 0 or hy0 < 0 or hx1 > he_ext[0] or hy1 > he_ext[1]:
                    continue
                hs = (slice(int(hy0 / he_um), int(hy1 / he_um)), slice(int(hx0 / he_um), int(hx1 / he_um)))
                th = tis_he[hs]
                if ti_.size == 0 or th.size == 0:
                    continue
                inr = CR._inside(mapped, roi)
                rows.append({"roi_if": [float(v) for v in roi], "roi_he_bbox": [float(hx0), float(hy0), float(hx1), float(hy1)],
                             "n_if": int(CR._inside(iff, roi).sum()), "n_he": int(inr.sum()), "n_he_suspect": int((inr & suspect).sum()),
                             "tissue_if": float(ti_.mean()), "tissue_he": float(th.mean()),
                             "ridge_if": float(r_if[sl][ti_].mean()) if ti_.any() else 0.0,
                             "ridge_he": float(r_he[hs][th].mean()) if th.any() else 0.0})
        picks, used = {}, []

        def free(r):
            return all(abs(r["roi_if"][0] - u[0]) >= ROI_UM or abs(r["roi_if"][1] - u[1]) >= ROI_UM for u in used)

        def take(name, cand, key, best_high=True):
            cand = [r for r in cand if free(r)]
            if not cand:
                picks[name] = None
                return
            r = max(cand, key=key) if best_high else min(cand, key=key)
            picks[name] = r; used.append(r["roi_if"])
        full = [r for r in rows if r["tissue_if"] >= TISSUE_IF and r["tissue_he"] >= TISSUE_HE]
        q = lambda k: np.percentile([r[k] for r in rows], [10, 50, 90]).round(3).tolist()  # noqa: E731
        print(sample, "windows", len(rows), "full", len(full), {k: q(k) for k in ("tissue_if", "tissue_he", "n_if", "n_he", "ridge_if", "ridge_he")}, flush=True)
        take("stardist_suspect", [r for r in rows if r["tissue_he"] >= 0.5 and r["tissue_if"] >= 0.5] or rows, lambda r: (r["n_he_suspect"], r["n_he"]))
        take("dense", full, lambda r: r["n_if"] + r["n_he"])
        take("sparse", [r for r in full if r["n_if"] + r["n_he"] >= 4], lambda r: r["n_if"] + r["n_he"], best_high=False)
        take("clear_boundary", full, lambda r: r["ridge_if"])
        take("unclear_boundary", full, lambda r: r["ridge_if"], best_high=False)
        take("tissue_edge", [r for r in rows if 0.2 <= min(r["tissue_if"], r["tissue_he"]) <= 0.8 and r["n_if"] + r["n_he"] >= 4],
             lambda r: -abs(min(r["tissue_if"], r["tissue_he"]) - 0.5))
        rng = np.random.default_rng(0)
        for k in ("tune_1", "tune_2"):
            pool = [r for r in full if free(r)] or sorted([r for r in rows if free(r)], key=lambda r: -min(r["tissue_if"], r["tissue_he"]))[:10]
            r = pool[int(rng.integers(len(pool)))]
            picks[k] = r; used.append(r["roi_if"])
        out["samples"][sample] = {"result_id": rec.rid, "global_M": np.asarray(M).tolist(), "global_t": np.asarray(t).tolist(),
                                  "n_windows": len(rows), "n_he_suspect_total": int(suspect.sum()),
                                  "he_suspect_ratio_median": float(np.nanmedian(ratio)), "he_suspect_threshold": thr,
                                  "rois": {k: ({**v, "role": "tuning" if k.startswith("tune") else "evaluation"} if v else None) for k, v in picks.items()}}
        # crops at WORK_UM in each section's own frame
        for name, r in picks.items():
            if r is None:
                continue
            d = HERE / "cache" / "crops" / sample / name
            d.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            meta = {}
            for key, mod, box in (("IF", if_m, r["roi_if"]), ("HE", he_m, r["roi_he_bbox"])):
                rgb, valid, info = crop_world(mod, *box, WORK_UM)
                Image.fromarray(rgb).save(d / f"{key}.png")
                meta[key] = {"origin_um": [box[0], box[1]], "um_per_px": WORK_UM, "shape": list(rgb.shape[:2]), "valid_fraction": float(valid.mean()),
                             "source": mod.image_path, **info}
            (d / "crop_meta.json").write_text(json.dumps(meta, indent=1))
    (HERE / "rois.json").write_text(json.dumps(out, indent=1))
    for s, v in out["samples"].items():
        print(s, v["n_windows"], "suspects", v["n_he_suspect_total"])
        for k, r in v["rois"].items():
            print("  ", k, None if r is None else {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in r.items() if kk not in ("roi_he_bbox",)})


if __name__ == "__main__":
    main()
