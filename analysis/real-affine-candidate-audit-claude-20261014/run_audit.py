"""Independent Real affine / coordinate / candidate-discovery audit (Claude, 2026-10-14).

Usage:  python -B run_audit.py --stage all|coordinate|correlogram|affine|he|ncc|candidates|domain|spatial
Every stage writes only inside this directory (cache/ for intermediates).
"""
from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

import audit_core as ac

CACHE = ac.OUT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
EDGE_MARGIN_PX = 10.0          # Workflow C UI default (app.py)
NEAR_HE_UM = 18.0 + 25.0       # cluster patch radius + cluster search radius defaults (app.py)
BLOCK_UM = 200.0
CORR_SEARCH_UM = 150.0
CORR_KERNEL_UM = 3.0
PEAK_RATIO_MIN = 5.0           # pre-specified: blocks used to seed the diagnostic affine
PAIR_RADIUS_UM = 4.0
WINDOWS = (18.0, 36.0, 54.0, 72.0)   # 72 is an extra diagnostic beyond the requested 18/36/54
K_VALUES = (10, 30, 50, 100)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =========================================================================== A. coordinates
def stage_coordinate(d: ac.RealData) -> dict:
    log("coordinate audit")
    ori = ac.oriented(d)
    rebuilt = ac.apply_affine(ori, d.affine_matrix, d.translation)
    inv = np.linalg.solve(d.affine_matrix, (rebuilt - d.translation).T).T
    raw_back = ac.apply_axis_flips(inv, image_width=d.width, image_height=d.height, flip_x=d.flip_x, flip_y=d.flip_y)
    flipM = np.diag([-1.0 if d.flip_x else 1.0, -1.0 if d.flip_y else 1.0])
    eff_M = d.affine_matrix @ flipM
    eff_t = d.translation + d.affine_matrix @ np.array([d.width if d.flip_x else 0.0, d.height if d.flip_y else 0.0])
    eff_rebuilt = d.moving_raw_px @ eff_M.T + eff_t

    # independent subset-filter reproduction from the saved warped mask + metadata
    meta, mask = d.warp_meta, d.tissue_mask
    def validity(points, near_he_from=None):
        pix = ac.world_to_warped_pixel(points, meta)
        cols, rows = pix[:, 0], pix[:, 1]
        H, W = mask.shape
        inside = (cols >= 0) & (cols < W) & (rows >= 0) & (rows < H)
        ri = np.clip(np.rint(rows).astype(int), 0, H - 1); ci = np.clip(np.rint(cols).astype(int), 0, W - 1)
        tissue = np.zeros(len(points), bool); tissue[inside] = mask[ri[inside], ci[inside]]
        din = distance_transform_edt(mask)[ri, ci]
        edge_ok = (cols >= EDGE_MARGIN_PX) & (cols < W - EDGE_MARGIN_PX) & (rows >= EDGE_MARGIN_PX) & (rows < H - EDGE_MARGIN_PX)
        near = np.ones(len(points), bool)
        if near_he_from is not None:
            near = cKDTree(near_he_from).query(points)[0] <= NEAR_HE_UM
        reasons = np.where(~inside, "outside_image", np.where(~tissue, "outside_he_footprint",
                  np.where(~edge_ok, "near_image_edge", np.where(din < EDGE_MARGIN_PX, "within_edge_margin",
                  np.where(~near, "no_nearby_he_points", "valid")))))
        return inside & tissue & edge_ok & (din >= EDGE_MARGIN_PX) & near, reasons
    fixed_valid, fixed_reason = validity(d.fixed_all, d.moving_affine_all)
    moving_valid, moving_reason = validity(d.moving_affine_all)
    fixed_match = np.array_equal(np.flatnonzero(fixed_valid), np.sort(d.fixed_orig_index))
    moving_match = np.array_equal(np.flatnonzero(moving_valid), np.sort(d.moving_orig_index))

    # Is the saved "tissue mask" a tissue mask or the HE image footprint?
    he = d.he_image
    nonblack = he.sum(axis=2) > 0
    white = ac.he_white_mask(he, mask)
    # HE-image / HE-point consistency: HE nuclei should sit on dark (hematoxylin) pixels
    gray = he.astype(float).mean(axis=2)
    def gray_at(points):
        pix = np.rint(ac.world_to_warped_pixel(points, meta)).astype(int)
        ok = (pix[:, 1] >= 0) & (pix[:, 1] < gray.shape[0]) & (pix[:, 0] >= 0) & (pix[:, 0] < gray.shape[1])
        vals = gray[pix[ok, 1], pix[ok, 0]]
        return vals[mask[pix[ok, 1], pix[ok, 0]]]
    rng = np.random.default_rng(7)
    rand_pts = np.column_stack([rng.uniform(40, 980, 20000), rng.uniform(40, 980, 20000)])
    res = {
        "inputs_identical_runs_vs_sweep": {
            f: ac.sha256_file(ac.INPUT / f) == ac.sha256_file(ac.SWEEP_INPUT / f)
            for f in sorted(p.name for p in ac.INPUT.iterdir())
        },
        "counts": {"fixed_original": len(d.fixed_all), "fixed_subset": len(d.fixed), "moving_original": len(d.moving_raw_px),
                   "moving_subset": len(d.moving)},
        "raw_moving_range_px": [d.moving_raw_px.min(0).tolist(), d.moving_raw_px.max(0).tolist()],
        "fixed_range_um": [d.fixed_all.min(0).tolist(), d.fixed_all.max(0).tolist()],
        "affine_saved": {"matrix": d.affine_matrix.tolist(), "translation": d.translation.tolist(), "flip_x": d.flip_x,
                         "flip_y": d.flip_y, "image_width": d.width, "image_height": d.height},
        "post_flip_decomposition": ac.decompose_affine(d.affine_matrix),
        "effective_raw_to_world": {"matrix": eff_M.tolist(), "translation": eff_t.tolist(), **ac.decompose_affine(eff_M)},
        "reconstruction_max_abs_error_um": float(np.max(np.abs(rebuilt - d.moving_affine_all))),
        "effective_matrix_reconstruction_max_abs_error_um": float(np.max(np.abs(eff_rebuilt - d.moving_affine_all))),
        "inverse_roundtrip_max_abs_error_px": float(np.max(np.abs(raw_back - d.moving_raw_px))),
        "subset_ids": {
            "fixed_subset_rows_unique_and_exact": True, "moving_subset_rows_unique_and_exact": True,
            "fixed_subset_order_monotonic": bool(np.all(np.diff(d.fixed_orig_index) > 0)),
            "moving_subset_order_monotonic": bool(np.all(np.diff(d.moving_orig_index) > 0)),
        },
        "subset_filter_reproduced": {"fixed": bool(fixed_match), "moving": bool(moving_match),
                                     "fixed_exclusion_reasons": pd.Series(fixed_reason[~fixed_valid]).value_counts().to_dict(),
                                     "moving_exclusion_reasons": pd.Series(moving_reason[~moving_valid]).value_counts().to_dict()},
        "tissue_mask_semantics": {
            "mask_fraction_of_canvas": float(mask.mean()),
            "mask_equals_nonblack_footprint_fraction": float(np.mean(mask == (gray / 255.0 > 0.05))),
            "nonblack_fraction": float(nonblack.mean()),
            "he_white_fraction_inside_mask": float(white[mask].mean()),
            "interpretation": "gray>0.05 threshold on an H&E image keeps bright glass/lumen; the mask is the HE image footprint, not tissue.",
        },
        "he_point_image_consistency": {
            "gray_median_at_he_nuclei": float(np.median(gray_at(d.moving_affine_all))),
            "gray_median_at_random_in_footprint": float(np.median(gray_at(rand_pts))),
            "gray_median_at_fixed_nuclei": float(np.median(gray_at(d.fixed_all))),
        },
        "metadata_available": {k: k in d.warp_meta for k in ("raw_pixel_size_um", "crop_offset", "downsample", "crs")},
    }
    ac.save_json(CACHE / "coordinate.json", res)
    pd.DataFrame({"original_index": np.arange(len(d.fixed_all)), "x": d.fixed_all[:, 0], "y": d.fixed_all[:, 1],
                  "valid": fixed_valid, "reason": fixed_reason}).to_csv(CACHE / "fixed_subset_reasons.csv", index=False)
    pd.DataFrame({"original_index": np.arange(len(d.moving_affine_all)), "x": d.moving_affine_all[:, 0],
                  "y": d.moving_affine_all[:, 1], "valid": moving_valid, "reason": moving_reason}).to_csv(
        CACHE / "moving_subset_reasons.csv", index=False)
    return res


# =========================================================================== B1. correlogram
def block_peaks(fixed, moving, *, density, block=BLOCK_UM, search=CORR_SEARCH_UM, step=2.0, centers=None, half=None):
    rows = []
    if centers is None:
        xs = np.arange(0, 1000, block); ys = np.arange(0, 1000, block)
        centers = [(x + block / 2, y + block / 2) for x in xs for y in ys]
        half = block / 2
    for cx, cy in centers:
        sel = (np.abs(moving[:, 0] - cx) < half) & (np.abs(moving[:, 1] - cy) < half)
        if sel.sum() < 25:
            continue
        r = ac.displacement_peak(fixed, moving[sel], search_um=search, kernel_um=CORR_KERNEL_UM, step_um=step, density=density)
        rows.append({"center_x": cx, "center_y": cy, **r})
    return pd.DataFrame(rows)


def offset_field(table: pd.DataFrame, points: np.ndarray, ratio_min=PEAK_RATIO_MIN) -> np.ndarray:
    """Inverse-distance weighted offset from reliable windows (diagnostic proxy, not a deformation field)."""
    t = table.loc[table["peak_ratio"] >= ratio_min]
    c = t[["center_x", "center_y"]].to_numpy(); o = t[["refined_dx", "refined_dy"]].to_numpy()
    dist, idx = cKDTree(c).query(points, k=min(4, len(c)))
    w = 1.0 / np.maximum(dist, 1.0) ** 2
    return np.einsum("nk,nkd->nd", w, o[idx]) / w.sum(axis=1, keepdims=True)


def stage_correlogram(d: ac.RealData) -> dict:
    log("correlogram")
    density = len(d.fixed_all) / (np.ptp(d.fixed_all[:, 0]) * np.ptp(d.fixed_all[:, 1]))
    blocks = block_peaks(d.fixed_all, d.moving_affine_all, density=density)
    blocks["condition"] = "saved_affine"
    nulls = []
    for name, pts in ac.null_transforms(d.moving_affine_all).items():
        b = block_peaks(d.fixed_all, pts, density=density); b["condition"] = f"null_{name}"; nulls.append(b)
    # sliding windows (100 um spacing, 200 um windows) for the spatial map
    centers = [(x, y) for x in np.arange(50, 1000, 100) for y in np.arange(50, 1000, 100)]
    sliding = block_peaks(d.fixed_all, d.moving_affine_all, density=density, centers=centers, half=100.0)
    sliding["condition"] = "saved_affine_sliding"
    allb = pd.concat([blocks, *nulls, sliding], ignore_index=True)
    allb.to_csv(CACHE / "correlogram_blocks.csv", index=False)
    edges = np.array([0, 2, 4, 6, 8, 10, 14, 20, 30, 40])
    g_saved = ac.radial_cross_correlation(d.fixed, d.moving, edges)
    # linear model of block offsets: offset = B @ [x, y] + c  (diagnostic)
    good = blocks.loc[blocks["peak_ratio"] >= PEAK_RATIO_MIN]
    X = np.column_stack([good["center_x"], good["center_y"], np.ones(len(good))])
    coef, *_ = np.linalg.lstsq(X, good[["refined_dx", "refined_dy"]].to_numpy(), rcond=None)
    J = np.eye(2) + coef[:2].T
    res = {
        "density_used": density,
        "blocks_total": int(len(blocks)),
        "blocks_peak_ratio_ge_5": int((blocks["peak_ratio"] >= PEAK_RATIO_MIN).sum()),
        "block_peak_ratio_median": float(blocks["peak_ratio"].median()),
        "block_peak_ratio_min": float(blocks["peak_ratio"].min()),
        "null_peak_ratio_median": {c: float(t["peak_ratio"].median()) for c, t in pd.concat(nulls).groupby("condition")},
        "null_peak_ratio_max": {c: float(t["peak_ratio"].max()) for c, t in pd.concat(nulls).groupby("condition")},
        "block_offset_median_um": [float(blocks["refined_dx"].median()), float(blocks["refined_dy"].median())],
        "block_offset_norm_range_um": [float(np.hypot(blocks.refined_dx, blocks.refined_dy).min()),
                                        float(np.hypot(blocks.refined_dx, blocks.refined_dy).max())],
        "block_fraction_with_partner_at_peak": float((blocks["peak_count"] / blocks["n_moving"]).median()),
        "linear_offset_model": {"jacobian_of_residual_map": J.tolist(), "offset_at_origin": coef[2].tolist(),
                                "rotation_deg": float(np.degrees(np.arctan2(J[1, 0] - J[0, 1], J[0, 0] + J[1, 1]))),
                                "rms_misfit_um": float(np.sqrt(np.mean(np.sum((X @ coef - good[['refined_dx', 'refined_dy']].to_numpy()) ** 2, 1))))},
        "radial_g_saved_affine": dict(zip([f"{a}-{b}" for a, b in zip(edges[:-1], edges[1:])], g_saved.tolist())),
    }
    ac.save_json(CACHE / "correlogram.json", res)
    return res


# =========================================================================== B2. diagnostic affine
def diagnostic_affine(d: ac.RealData, select_moving: np.ndarray, blocks: pd.DataFrame):
    """Seed pairs from correlogram offsets, then refine with the production affine_icp.

    select_moving: boolean mask over the 1,842 moving nuclei allowed for fitting.
    Diagnostic only; not a proposal for production parameters.
    """
    ori = ac.oriented(d)
    pred = d.moving_affine_all + offset_field(blocks, d.moving_affine_all)
    tree = cKDTree(d.fixed_all)
    dist, j = tree.query(pred, k=1)
    back = cKDTree(pred[select_moving]).query(d.fixed_all, k=1)[1]
    sel_idx = np.flatnonzero(select_moving)
    mutual = np.zeros(len(pred), bool)
    mutual[sel_idx] = sel_idx[back[j[sel_idx]]] == sel_idx
    pairs = select_moving & mutual & (dist <= PAIR_RADIUS_UM)
    M0, t0 = ac.affine_lstsq(ori[pairs], d.fixed_all[j[pairs]])
    M1, t1, err, n_used = ac.affine_icp(ori[select_moving], d.fixed_all, M0, t0, max_iter=80, trim_quantile=0.7)
    return {"seed_pairs": int(pairs.sum()), "seed_matrix": M0, "seed_translation": t0,
            "matrix": M1, "translation": t1, "icp_trimmed_error": err, "icp_pairs": n_used}


def stage_affine(d: ac.RealData) -> dict:
    log("diagnostic affine + pose landscape")
    blocks = pd.read_csv(CACHE / "correlogram_blocks.csv")
    blocks = blocks.loc[blocks["condition"] == "saved_affine"]
    ori = ac.oriented(d)
    full = diagnostic_affine(d, np.ones(len(ori), bool), blocks)
    corr_all = ac.apply_affine(ori, full["matrix"], full["translation"])
    corr_sub = corr_all[d.moving_orig_index]
    saved_obj = ac.icp_trimmed_objective(d.moving_affine_all, d.fixed_all)
    corr_obj = ac.icp_trimmed_objective(corr_all, d.fixed_all)
    # held-out validation: fit on one half, evaluate on the other half
    cv = []
    x, y = d.moving_affine_all[:, 0], d.moving_affine_all[:, 1]
    mx, my = np.median(x), np.median(y)
    splits = (
        ("left->right", x < mx, blocks.center_x < mx),
        ("right->left", x >= mx, blocks.center_x >= mx),
        ("bottom->top", y < my, blocks.center_y < my),
        ("top->bottom", y >= my, blocks.center_y >= my),
    )
    for name, train, block_sel in splits:
        tb = blocks.loc[block_sel]   # correlogram seeds come only from the training half
        fit = diagnostic_affine(d, train, tb)
        test = ~train
        pts = ac.apply_affine(ori[test], fit["matrix"], fit["translation"])
        saved_test = d.moving_affine_all[test]
        null_pts = ac.null_transforms(pts)["mirror_x"]
        cv.append({"split": name, "train_n": int(train.sum()), "test_n": int(test.sum()), "train_blocks": int(len(tb)),
                   "seed_pairs": fit["seed_pairs"],
                   **{f"heldout_corrected_{k}": v for k, v in ac.nn_stats(pts, d.fixed_all).items()},
                   **{f"heldout_saved_{k}": v for k, v in ac.nn_stats(saved_test, d.fixed_all).items()},
                   **{f"heldout_null_{k}": v for k, v in ac.nn_stats(null_pts, d.fixed_all).items()},
                   "matrix": np.asarray(fit["matrix"]).tolist(), "translation": np.asarray(fit["translation"]).tolist()})
    pd.DataFrame(cv).to_csv(CACHE / "affine_holdout.csv", index=False)

    # pose landscape with production functions: 4 flips x 12 rotations
    land = []
    for fx, fy in ((False, False), (False, True), (True, False), (True, True)):
        src = ac.apply_axis_flips(d.moving_raw_px, image_width=d.width, image_height=d.height, flip_x=fx, flip_y=fy)
        for rot in range(0, 360, 30):
            sim, sim_err, _ = ac.best_of_initial_rotations(src, d.fixed_all, rotations_deg=[rot], max_iter=80, trim_quantile=0.8)
            M, t = sim.to_affine()
            M, t, aff_err, _ = ac.affine_icp(src, d.fixed_all, M, t, max_iter=80, trim_quantile=0.7)
            pts = ac.apply_affine(src, M, t)
            land.append({"flip_x": fx, "flip_y": fy, "init_rotation_deg": rot, "similarity_error": sim_err,
                         "affine_trimmed_error": aff_err, "det_post_flip": float(np.linalg.det(M)),
                         "effective_det": float(np.linalg.det(M) * (-1 if fx else 1) * (-1 if fy else 1)),
                         **ac.nn_stats(pts, d.fixed_all)})
    pd.DataFrame(land).to_csv(CACHE / "pose_landscape.csv", index=False)

    # capture range: perturb the saved translation and re-run production affine_icp
    cap = []
    for dx in range(-80, 81, 10):
        for dy in range(-80, 81, 10):
            M, t, err, _ = ac.affine_icp(ori, d.fixed_all, d.affine_matrix, d.translation + np.array([dx, dy]),
                                         max_iter=80, trim_quantile=0.7)
            pts = ac.apply_affine(ori, M, t)
            st = ac.nn_stats(pts, d.fixed_all)
            cap.append({"init_dx": dx, "init_dy": dy, "final_trimmed_error": err,
                        "final_translation_change_x": float(np.mean(pts[:, 0] - d.moving_affine_all[:, 0])),
                        "final_translation_change_y": float(np.mean(pts[:, 1] - d.moving_affine_all[:, 1])),
                        "within_3um": st["within_3um"], "mutual_nn_within_3um_rate": st["mutual_nn_within_3um_rate"]})
    pd.DataFrame(cap).to_csv(CACHE / "icp_capture_range.csv", index=False)

    # residual map of correction (corrected minus saved position)
    delta = corr_all - d.moving_affine_all
    flipM = np.diag([-1.0 if d.flip_x else 1.0, -1.0 if d.flip_y else 1.0])
    rel = full["matrix"] @ np.linalg.inv(d.affine_matrix)   # corrected = rel @ saved + c (in world)
    res = {
        "method": "seed pairs: mutual NN <=4 um after correlogram block offsets (blocks with peak ratio>=5); "
                  "init = affine_lstsq on oriented raw px; refine = production affine_icp(trim 0.7, 80 it) on all 1,842 vs 1,979",
        "seed_pairs": full["seed_pairs"],
        "corrected_matrix_post_flip": np.asarray(full["matrix"]).tolist(),
        "corrected_translation": np.asarray(full["translation"]).tolist(),
        "corrected_decomposition": ac.decompose_affine(full["matrix"]),
        "corrected_effective_decomposition": ac.decompose_affine(full["matrix"] @ flipM),
        "relative_world_correction": {"matrix": rel.tolist(), **ac.decompose_affine(rel)},
        "correction_displacement_um": {"median_dx": float(np.median(delta[:, 0])), "median_dy": float(np.median(delta[:, 1])),
                                       "norm_min": float(np.hypot(*delta.T).min()), "norm_median": float(np.median(np.hypot(*delta.T))),
                                       "norm_max": float(np.hypot(*delta.T).max())},
        "icp_objective_trimmed_q0.7": {"saved": saved_obj, "corrected": corr_obj},
        "nn_saved_subset": ac.nn_stats(d.moving, d.fixed),
        "nn_corrected_subset": ac.nn_stats(corr_sub, d.fixed),
        "nn_null_subset": {k: ac.nn_stats(v, d.fixed) for k, v in ac.null_transforms(d.moving).items()},
        "nn_null_corrected_subset": {k: ac.nn_stats(v, d.fixed) for k, v in ac.null_transforms(corr_sub).items()},
        "poisson_expectation_within_r": {r: float(1 - np.exp(-len(d.fixed) / (np.ptp(d.fixed[:, 0]) * np.ptp(d.fixed[:, 1])) * np.pi * r * r)) for r in (2, 3, 5, 10)},
        "radial_g_corrected": dict(zip(["0-2", "2-4", "4-6", "6-8", "8-10", "10-14", "14-20", "20-30", "30-40"],
                                       ac.radial_cross_correlation(d.fixed, corr_sub, np.array([0, 2, 4, 6, 8, 10, 14, 20, 30, 40])).tolist())),
    }
    np.savez(CACHE / "corrected_points.npz", corrected_all=corr_all, corrected_subset=corr_sub,
             matrix=np.asarray(full["matrix"]), translation=np.asarray(full["translation"]))
    ac.save_json(CACHE / "affine.json", res)
    return res


# =========================================================================== C. HE structure / ROI
def stage_he(d: ac.RealData) -> dict:
    log("HE structure / ROI")
    corr = np.load(CACHE / "corrected_points.npz")
    white = ac.he_white_mask(d.he_image, d.tissue_mask)
    white_loose = ac.he_white_mask(d.he_image, d.tissue_mask, value=180.0, sat=0.30)
    # Fixed points expressed in the saved-affine world frame of the warped HE image:
    # world_saved = A_s * A_c^-1 * (fixed - t_c) + t_s
    Ms, ts = d.affine_matrix, d.translation
    Mc, tc = corr["matrix"], corr["translation"]
    fixed_in_saved = (np.linalg.solve(Mc, (d.fixed_all - tc).T).T) @ Ms.T + ts
    out = {}
    for name, m in (("strict", white), ("loose", white_loose)):
        area = float(m[d.tissue_mask].mean())
        fs, ns = ac.fraction_in_mask(d.fixed_all, m, d.warp_meta)
        fc, nc = ac.fraction_in_mask(fixed_in_saved, m, d.warp_meta)
        fm, _ = ac.fraction_in_mask(d.moving_affine_all, m, d.warp_meta)
        out[name] = {"white_area_fraction": area, "fixed_saved": fs, "fixed_saved_n": ns,
                     "fixed_corrected": fc, "fixed_corrected_n": nc, "he_nuclei": fm,
                     "avoidance_ratio_saved": fs / area, "avoidance_ratio_corrected": fc / area}
    # translation scan (fixed shifted in saved world) -- image-based estimate independent of point matching
    scan = []
    for dx in range(-90, 91, 3):
        for dy in range(-90, 91, 3):
            f, n = ac.fraction_in_mask(d.fixed_all + np.array([dx, dy]), white, d.warp_meta)
            scan.append({"shift_dx": dx, "shift_dy": dy, "fixed_in_white": f, "n_inside": n})
    scan = pd.DataFrame(scan)
    scan.to_csv(CACHE / "he_white_translation_scan.csv", index=False)
    valid = scan.loc[scan["n_inside"] >= 0.8 * len(d.fixed_all)]
    best = valid.loc[valid["fixed_in_white"].idxmin()]
    out["translation_scan_min"] = {"shift_dx": float(best.shift_dx), "shift_dy": float(best.shift_dy),
                                   "fixed_in_white": float(best.fixed_in_white),
                                   "implied_fixed_minus_moving_offset": [-float(best.shift_dx), -float(best.shift_dy)]}
    # ROI geometry under saved vs corrected transforms
    from scipy.spatial import ConvexHull
    from matplotlib.path import Path as MPath
    def hull_path(p):
        h = ConvexHull(p); return MPath(p[h.vertices]), float(h.volume)
    he_corners = np.array([[0, 0], [d.width, 0], [d.width, d.height], [0, d.height]], float)
    for name, (M, t) in (("saved", (Ms, ts)), ("corrected", (Mc, tc))):
        poly = ac.apply_affine(ac.oriented(d, he_corners), M, t)
        path = MPath(poly)
        fixed_in = path.contains_points(d.fixed_all)
        fpath, farea = hull_path(d.fixed_all)
        out[f"roi_{name}"] = {"he_image_footprint_world_polygon": poly.tolist(),
                              "fixed_inside_he_footprint_fraction": float(fixed_in.mean()),
                              "fixed_outside_he_footprint_n": int((~fixed_in).sum()),
                              "fixed_hull_area_um2": farea}
    ac.save_json(CACHE / "he_roi.json", out)
    return out


# =========================================================================== D. regional NCC
def stage_ncc(d: ac.RealData) -> dict:
    log("regional NCC windows")
    import regional_pattern as rp
    corr = np.load(CACHE / "corrected_points.npz")
    blocks = pd.read_csv(CACHE / "correlogram_blocks.csv")
    sliding = blocks.loc[blocks["condition"] == "saved_affine_sliding"]
    t0 = time.time()
    frozen = rp.match_regional_patterns(d.fixed, d.moving)
    frozen_s = time.time() - t0
    rows, anchor_rows, per_cell = [], [], {}
    conditions = [("saved_affine", d.moving, w) for w in WINDOWS]
    conditions += [("corrected_affine", corr["corrected_subset"], 18.0), ("corrected_affine", corr["corrected_subset"], 54.0)]
    conditions += [("null_mirror_x", ac.null_transforms(d.moving)["mirror_x"], 18.0),
                   ("null_mirror_x", ac.null_transforms(d.moving)["mirror_x"], 54.0)]
    validation = {}
    for cond, mov, R in conditions:
        t0 = time.time()
        r = ac.regional_ncc(d.fixed, mov, R)
        el = time.time() - t0
        if cond == "saved_affine" and R == 18.0:
            validation = {
                "anchors_equal": bool(np.array_equal(r.anchors, frozen.anchor_centers)),
                "best_offsets_equal": bool(np.array_equal(r.best_offsets, frozen.best_offsets)),
                "best_ncc_max_abs_diff": float(np.max(np.abs(r.best_ncc - frozen.best_ncc))),
                "second_ncc_max_abs_diff": float(np.max(np.abs(r.second_ncc - frozen.second_ncc))),
                "cell_offsets_equal": bool(np.array_equal(r.cell_offsets, frozen.cell_offsets)),
                "cell_confidence_max_abs_diff": float(np.max(np.abs(r.cell_confidence - frozen.cell_confidence))),
                "frozen_runtime_s": frozen_s, "vectorized_runtime_s": el,
            }
        hit = r.boundary_x | r.boundary_y
        cell_hit = hit[r.cell_anchor_index]
        # agreement with the independent point-correlogram offset at each anchor
        ref = offset_field(sliding, r.anchors) if cond == "saved_affine" else np.full_like(r.anchors, np.nan)
        agree = np.hypot(*(r.best_offsets - ref).T)
        # spatial consistency: neighbouring anchors (<= 60 um) offset difference
        pairs = cKDTree(r.anchors).query_pairs(60.0, output_type="ndarray")
        nb_diff = np.hypot(*(r.best_offsets[pairs[:, 0]] - r.best_offsets[pairs[:, 1]]).T) if len(pairs) else np.array([np.nan])
        rows.append({
            "condition": cond, "search_radius_um": R, "n_anchors": len(r.anchors), "n_nuclei": len(mov),
            "anchor_boundary_hit_rate": float(hit.mean()), "anchor_boundary_hit_n": int(hit.sum()),
            "nucleus_boundary_hit_rate": float(cell_hit.mean()), "nucleus_boundary_hit_n": int(cell_hit.sum()),
            "boundary_x_rate": float(r.boundary_x.mean()), "boundary_y_rate": float(r.boundary_y.mean()),
            "best_ncc_median": float(np.median(r.best_ncc)), "second_ncc_median": float(np.median(r.second_ncc)),
            "peak_margin_median": float(np.median(r.margin)),
            "interior_local_max_rate": float(r.interior_local_max.mean()),
            "offset_dx_median": float(np.median(r.best_offsets[:, 0])), "offset_dy_median": float(np.median(r.best_offsets[:, 1])),
            "offset_norm_median": float(np.median(np.hypot(*r.best_offsets.T))),
            "neighbor_anchor_offset_diff_median_um": float(np.median(nb_diff)),
            "agree_with_correlogram_within_6um_rate": float(np.mean(agree <= 6.0)) if cond == "saved_affine" else np.nan,
            "agree_with_correlogram_median_um": float(np.median(agree)) if cond == "saved_affine" else np.nan,
            "cell_confidence_median": float(np.median(r.cell_confidence)),
            "runtime_s": el,
        })
        for a in range(len(r.anchors)):
            anchor_rows.append({"condition": cond, "search_radius_um": R, "anchor": a, "x": r.anchors[a, 0], "y": r.anchors[a, 1],
                                "best_dx": r.best_offsets[a, 0], "best_dy": r.best_offsets[a, 1], "best_ncc": r.best_ncc[a],
                                "second_ncc": r.second_ncc[a], "margin": r.margin[a], "boundary_hit": bool(hit[a]),
                                "interior_local_max": bool(r.interior_local_max[a]), "peak_curvature": r.peak_curvature[a],
                                "correlogram_dx": ref[a, 0], "correlogram_dy": ref[a, 1], "agree_um": agree[a],
                                "n_cells": int(np.sum(r.cell_anchor_index == a))})
        per_cell[f"{cond}_{int(R)}"] = r
        log(f"  {cond} ±{R:g}: boundary {hit.mean():.3f} (nuclei {cell_hit.mean():.3f}) {el:.1f}s")
    pd.DataFrame(rows).to_csv(ac.OUT / "ncc_window_sensitivity.csv", index=False)
    pd.DataFrame(anchor_rows).to_csv(CACHE / "ncc_anchor_table.csv", index=False)
    np.savez(CACHE / "ncc_cell_offsets.npz", **{k: v.cell_offsets for k, v in per_cell.items()},
             **{f"{k}__conf": v.cell_confidence for k, v in per_cell.items()},
             **{f"{k}__hit": (v.boundary_x | v.boundary_y)[v.cell_anchor_index] for k, v in per_cell.items()})
    ac.save_json(CACHE / "ncc_validation.json", validation)
    return validation


# =========================================================================== E. candidates
def frozen_components(fixed, moving, regional_offsets=None, regional_conf=None):
    """Frozen-equivalent inputs for candidate construction (fingerprint cost + regional result)."""
    from local_fingerprint import compute_fingerprint_pair, pairwise_fingerprint_cost
    import regional_pattern as rp
    fp = pairwise_fingerprint_cost(compute_fingerprint_pair(fixed, moving))
    reg = rp.match_regional_patterns(fixed, moving)
    if regional_offsets is not None:
        reg = SimpleNamespace(**{**reg.__dict__, "cell_offsets": regional_offsets, "cell_confidence": regional_conf})
    return fp, reg


def augmented_pool_candidates(fixed, moving, fp_cost, reg, k_values=K_VALUES, kmax=100):
    """Fixed-source augmented pool; only the final K varies (frozen full base-cost ordering)."""
    from candidate_generation import _top_order
    sd = cdist(moving, fixed)
    rr = cdist(moving + reg.cell_offsets, fixed)
    spatial = _top_order(sd, kmax)
    regional = _top_order(rr, kmax)
    pool80 = _top_order(sd, int(ac.CANDIDATES["fingerprint_spatial_pool"]))
    fpo = np.take_along_axis(pool80, np.argsort(np.take_along_axis(fp_cost, pool80, 1), axis=1), 1)
    w = ac.CANDIDATES["weights_full"]
    ps = float(ac.CANDIDATES["position_scale_um"]); rs = float(ac.CANDIDATES["region_residual_scale_um"])
    base_full = (w["position"] * sd**2 / (2 * ps**2) + w["fingerprint"] * fp_cost
                 + w["region"] * (rr**2 / (2 * rs**2) + 0.25 * (1.0 - reg.cell_confidence[:, None])))
    out = {}
    pools = []
    for i in range(len(moving)):
        pools.append(np.asarray(list(dict.fromkeys(np.concatenate([spatial[i], regional[i], fpo[i]]).tolist())), int))
    for K in k_values:
        idx = np.stack([p[np.argsort(base_full[i, p], kind="stable")[:K]] for i, p in enumerate(pools)])
        out[K] = (idx, np.take_along_axis(base_full, idx, 1))
    return out, spatial, regional, fpo, base_full, sd


def putative_partners(fixed, moving, offsets, radius=3.0):
    """Mutual nearest fixed nucleus within `radius` of moving+offset (a correlogram-derived proxy)."""
    pred = moving + offsets
    dist, j = cKDTree(fixed).query(pred, k=1)
    back = cKDTree(pred).query(fixed, k=1)[1]
    ok = (dist <= radius) & (back[j] == np.arange(len(moving)))
    return np.where(ok, j, -1)


def stage_candidates(d: ac.RealData) -> dict:
    log("candidate source audit + K sensitivity")
    from run_cost_ablation import build_fixed_context
    from candidate_generation import build_candidate_rankings
    stored = pd.read_csv(ac.SEP_SRC / "real_exploratory_per_nucleus.csv", usecols=["method", "moving_index", "candidate_hash", "top1_fixed_index"])
    stored_hash = stored["candidate_hash"].iloc[0]
    corr = np.load(CACHE / "corrected_points.npz")
    blocks = pd.read_csv(CACHE / "correlogram_blocks.csv")
    sliding = blocks.loc[blocks["condition"] == "saved_affine_sliding"]
    saved_offsets = offset_field(sliding, d.moving)
    partner_saved = putative_partners(d.fixed, d.moving, saved_offsets)
    partner_corr = putative_partners(d.fixed, corr["corrected_subset"], np.zeros_like(d.moving))
    null_mov = ac.null_transforms(d.moving)["mirror_x"]
    null_blocks = block_peaks(d.fixed, null_mov, density=len(d.fixed_all) / 1e6,
                              centers=[(x, y) for x in np.arange(50, 1000, 100) for y in np.arange(50, 1000, 100)], half=100.0)
    partner_null = putative_partners(d.fixed, null_mov, offset_field(null_blocks, null_mov, ratio_min=0.0))
    # consistency: correlogram-proxy partner vs corrected-affine partner (independent derivations)
    both = (partner_saved >= 0) & (partner_corr >= 0)

    ctx = build_fixed_context(d.fixed, d.moving)
    ok_hash = ac.candidate_hash(ctx.candidates.indices) == stored_hash
    t0 = time.time()
    gfast = ac.graph_support_fast(d.fixed, d.moving, ctx.candidates.indices, ctx.candidates.base_cost)
    gdiff = float(np.max(np.abs(gfast - ctx.graph_support)))
    fp, reg = frozen_components(d.fixed, d.moving)
    rk = build_candidate_rankings(d.fixed, d.moving, fp, reg)
    final = ctx.candidates.indices
    S = int(ac.CANDIDATES["source_k"])
    pg_cost = ctx.components["position"] + ctx.components["graph"]
    top1_pg = final[np.arange(len(final)), np.argmin(pg_cost, 1)]
    top1_base = final[:, 0]
    sd = rk.spatial_distance_um
    rows = []
    for i in range(len(d.moving)):
        sp, rg, fpl, fin = rk.spatial[i], rk.regional[i], rk.fingerprint[i], final[i]
        union = list(dict.fromkeys(np.concatenate([sp[:S], fpl[:S], rg[:S]]).tolist()))
        p = partner_saved[i]; pc = partner_corr[i]
        def rank(lst, v):
            w = np.flatnonzero(np.asarray(lst) == v)
            return int(w[0]) + 1 if v >= 0 and len(w) else -1
        rows.append({
            "moving_subset_index": i, "moving_original_index": int(d.moving_orig_index[i]),
            "moving_stable_id": f"he_moving_original_{int(d.moving_orig_index[i]):06d}",
            "x": d.moving[i, 0], "y": d.moving[i, 1],
            "spatial_top30": ";".join(map(str, sp)), "regional_top30": ";".join(map(str, rg)),
            "fingerprint_top30": ";".join(map(str, fpl)), "final30": ";".join(map(str, fin)),
            "union_size_before_prune": len(union),
            "final_from_spatial20": int(np.isin(fin, sp[:S]).sum()), "final_from_regional20": int(np.isin(fin, rg[:S]).sum()),
            "final_from_fingerprint20": int(np.isin(fin, fpl[:S]).sum()),
            "final_fill_only": int((~np.isin(fin, union)).sum()),
            "spatial_nn_fixed": int(sp[0]), "spatial_nn_dist_um": float(sd[i, sp[0]]),
            "spatial_nn_in_final": bool(sp[0] in fin), "spatial30_dist_um": float(sd[i, sp[-1]]),
            "final_max_dist_um": float(sd[i, fin].max()), "final_median_dist_um": float(np.median(sd[i, fin])),
            "top1_base_fixed": int(top1_base[i]), "top1_position_graph_fixed": int(top1_pg[i]),
            "top1_position_graph_dist_um": float(sd[i, top1_pg[i]]),
            "top1_position_graph_graph_support": float(ctx.graph_support[i, np.argmin(pg_cost[i])]),
            "max_graph_support": float(ctx.graph_support[i].max()),
            "putative_partner_fixed": int(p), "putative_partner_dist_um": float(sd[i, p]) if p >= 0 else np.nan,
            "putative_partner_rank_spatial": rank(np.argsort(sd[i], kind="stable")[:300], p),
            "putative_partner_in_spatial30": rank(sp, p) > 0, "putative_partner_in_regional30": rank(rg, p) > 0,
            "putative_partner_in_fingerprint30": rank(fpl, p) > 0, "putative_partner_in_union": p in union if p >= 0 else False,
            "putative_partner_in_final30": rank(fin, p) > 0,
            "putative_partner_is_top1_position_graph": bool(p >= 0 and p == top1_pg[i]),
            "corrected_affine_partner_fixed": int(pc), "proxy_partners_agree": bool(both[i] and p == pc),
            "regional_offset_dx": reg.cell_offsets[i, 0], "regional_offset_dy": reg.cell_offsets[i, 1],
        })
    src = pd.DataFrame(rows)
    src.to_csv(ac.OUT / "candidate_source_audit.csv", index=False)

    # K sensitivity under saved, corrected, and saved+wide NCC (±54) offsets
    ncc = np.load(CACHE / "ncc_cell_offsets.npz")
    ksens = []
    conditions = {
        "saved_affine": (d.moving, None, None, partner_saved),
        "saved_affine_ncc54": (d.moving, ncc["saved_affine_54"], ncc["saved_affine_54__conf"], partner_saved),
        "corrected_affine": (corr["corrected_subset"], None, None, partner_corr),
        "null_mirror_x": (null_mov, None, None, partner_null),
    }
    graph_cache = {}
    for name, (mov, off, conf, partner) in conditions.items():
        fpc, regc = frozen_components(d.fixed, mov, off, conf)
        pools, spatial, regional, fpo, base_full, sdc = augmented_pool_candidates(d.fixed, mov, fpc, regc)
        top1_ref = None
        for K in sorted(K_VALUES, reverse=True):
            idx, base = pools[K]
            g = ac.graph_support_fast(d.fixed, mov, idx, base)
            pgc = (ac.CANDIDATES["weights_full"]["position"] * (np.take_along_axis(sdc, idx, 1) ** 2) / (2 * ac.CANDIDATES["position_scale_um"] ** 2)
                   + ac.CANDIDATES["graph_weight"] * (1 - g))
            sl = np.argmin(pgc, 1)
            t1 = idx[np.arange(len(idx)), sl]
            if top1_ref is None:
                top1_ref = t1
            has = partner >= 0
            ksens.append({
                "condition": name, "K": K, "design": "augmented pool = spatial100 ∪ regional100 ∪ fingerprint(pool80); prune by frozen full base cost",
                "n_moving": len(mov), "putative_partner_n": int(has.sum()), "putative_partner_rate": float(has.mean()),
                "partner_coverage_at_K": float(np.mean(np.any(idx[has] == partner[has, None], 1))) if has.any() else np.nan,
                "spatial_topK_partner_coverage": float(np.mean(np.any(spatial[has, :K] == partner[has, None], 1))) if has.any() else np.nan,
                "regional_topK_partner_coverage": float(np.mean(np.any(regional[has, :K] == partner[has, None], 1))) if has.any() else np.nan,
                "kth_spatial_distance_median_um": float(np.median(np.take_along_axis(sdc, spatial[:, K - 1:K], 1))),
                "candidate_distance_median_um": float(np.median(np.take_along_axis(sdc, idx, 1))),
                "top1_pg_distance_median_um": float(np.median(sdc[np.arange(len(t1)), t1])),
                "top1_pg_graph_support_median": float(np.median(g[np.arange(len(sl)), sl])),
                "top1_pg_equals_partner_rate": float(np.mean(t1[has] == partner[has])) if has.any() else np.nan,
                "top1_pg_stability_vs_K100": float(np.mean(t1 == top1_ref)),
            })
            if K == 30:
                graph_cache[name] = (idx, g)
        log(f"  K-sensitivity {name} done")
    ks = pd.DataFrame(ksens).sort_values(["condition", "K"])
    ks.to_csv(ac.OUT / "candidate_k_sensitivity.csv", index=False)
    res = {
        "stored_candidate_hash_reproduced": bool(ok_hash),
        "graph_fast_vs_frozen_max_abs_diff": gdiff,
        "graph_fast_runtime_s": time.time() - t0,
        "union_size_median": float(src["union_size_before_prune"].median()),
        "spatial_nn_in_final_rate": float(src["spatial_nn_in_final"].mean()),
        "putative_partner_rate_saved_proxy": float((partner_saved >= 0).mean()),
        "putative_partner_rate_corrected": float((partner_corr >= 0).mean()),
        "putative_partner_rate_null": float((partner_null >= 0).mean()),
        "proxy_vs_corrected_partner_agreement": float(np.mean(partner_saved[both] == partner_corr[both])) if both.any() else np.nan,
        "partner_saved_proxy_in_spatial30": float(src.loc[src.putative_partner_fixed >= 0, "putative_partner_in_spatial30"].mean()),
        "partner_saved_proxy_in_regional30": float(src.loc[src.putative_partner_fixed >= 0, "putative_partner_in_regional30"].mean()),
        "partner_saved_proxy_in_fingerprint30": float(src.loc[src.putative_partner_fixed >= 0, "putative_partner_in_fingerprint30"].mean()),
        "partner_saved_proxy_in_union": float(src.loc[src.putative_partner_fixed >= 0, "putative_partner_in_union"].mean()),
        "partner_saved_proxy_in_final30": float(src.loc[src.putative_partner_fixed >= 0, "putative_partner_in_final30"].mean()),
        "partner_saved_proxy_is_top1": float(src.loc[src.putative_partner_fixed >= 0, "putative_partner_is_top1_position_graph"].mean()),
        "partner_saved_proxy_distance_median_um": float(src["putative_partner_dist_um"].median()),
        "partner_saved_proxy_spatial_rank_median": float(src.loc[src.putative_partner_rank_spatial > 0, "putative_partner_rank_spatial"].median()),
    }
    ac.save_json(CACHE / "candidates.json", res)
    np.savez(CACHE / "partners.npz", saved=partner_saved, corrected=partner_corr, null=partner_null, saved_offsets=saved_offsets)
    return res


# =========================================================================== Q6. domain comparison
def stage_domain(d: ac.RealData) -> dict:
    log("domain comparison (frozen pipeline on saved / corrected / null / synthetic)")
    import experiment_lib as lib
    corr = np.load(CACHE / "corrected_points.npz")
    per = pd.read_csv(ac.SEP_SRC / "per_nucleus_results.csv", low_memory=False)
    match_model, unmatched_model, thresholds, _ = lib.fit_separated_models(
        per, source_method=lib.METHOD_B1, feature_names=lib.UOT_FEATURES, output_method=lib.METHOD_B2)
    stored_real = pd.read_csv(ac.SEP_SRC / "real_exploratory_per_nucleus.csv")

    def run_case(name, fixed, moving, split="Real"):
        ctx = lib.build_fixed_context(fixed, moving)
        runs = lib.method_runs(ctx)
        sample = SimpleNamespace(fixed_points=fixed, moving_points=moving,
                                 fixed_ids=np.array([f"f{i}" for i in range(len(fixed))], dtype=object),
                                 moving_ids=np.array([f"m{i}" for i in range(len(moving))], dtype=object),
                                 true_fixed_index=np.full(len(moving), -1), common_moving_mask=np.zeros(len(moving), bool),
                                 seed=0, difficulty=name)
        scen = {"scenario_id": name, "split": split, "strength": name, "template_group": name}
        frames = [lib.result_frame(sample, scen, ctx, runs[m], method=m) for m in (lib.METHOD_B0, lib.METHOD_B1, lib.METHOD_B3)]
        b2 = lib.apply_separated_adoption(frames[1], match_model, unmatched_model, thresholds["match"]["threshold"],
                                          thresholds["unmatched"]["threshold"], method=lib.METHOD_B2)
        frames.append(b2)
        reg = ctx.regional
        hit = np.isclose(np.max(np.abs(reg.best_offsets), axis=1), ac.REGIONAL["search_radius_um"])
        knn = cKDTree(moving).query(moving, k=2)[0][:, 1]
        return pd.concat(frames, ignore_index=True), {
            "case": name, "ncc18_nucleus_boundary_rate": float(hit[reg.cell_anchor_index].mean()),
            "ncc18_anchor_boundary_rate": float(hit.mean()), "knn_median_um": float(np.median(knn)),
            "solver_all_converged": all(bool(r.metrics["solver_converged"]) for r in runs.values()),
            "finite_costs": all(np.isfinite(v).all() for v in ctx.components.values()),
        }

    cases = {
        "Real saved affine (baseline)": (d.fixed, d.moving),
        "Real corrected affine (diagnostic)": (d.fixed, corr["corrected_subset"]),
        "Real null: moving mirrored in x": (d.fixed, ac.null_transforms(d.moving)["mirror_x"]),
    }
    summ, all_frames = [], []
    for name, (f, m) in cases.items():
        fr, info = run_case(name, f, m)
        fr["case"] = name
        all_frames.append(fr)
        summ.append(info)
        log(f"  {name} done")
    # synthetic Test seeds: correct affine vs the Real-measured residual injected
    aff = json.loads((CACHE / "affine.json").read_text())
    R = np.asarray(aff["relative_world_correction"]["matrix"])
    for seed, strength in zip(lib.SPLIT_SEEDS["Test"], ("Weak", "Moderate", "Strong", "Weak", "Moderate")):
        s = lib.generate_real_like_sample(seed=seed, strength=strength)
        c = s.moving_points.mean(0)
        shift = np.array([aff["correction_displacement_um"]["median_dx"], aff["correction_displacement_um"]["median_dy"]])
        # moving is misplaced by the inverse of the Real correction (rotation about centre + translation)
        mis = (s.moving_points - c) @ np.linalg.inv(R).T + c - shift
        for label, mov in (("correct affine", s.moving_points), ("Real-measured residual injected", mis)):
            ctx = lib.build_fixed_context(s.fixed_points, mov)
            common = s.true_fixed_index >= 0
            rec = float(np.mean(np.any(ctx.candidates.indices[common] == s.true_fixed_index[common, None], 1)))
            pgc = ctx.components["position"] + ctx.components["graph"]
            sl = np.argmin(pgc, 1)
            t1 = ctx.candidates.indices[np.arange(len(sl)), sl]
            reg = ctx.regional
            hit = np.isclose(np.max(np.abs(reg.best_offsets), axis=1), ac.REGIONAL["search_radius_um"])
            summ.append({"case": f"Synthetic {seed} {strength}: {label}",
                         "ncc18_nucleus_boundary_rate": float(hit[reg.cell_anchor_index].mean()),
                         "ncc18_anchor_boundary_rate": float(hit.mean()),
                         "candidate_recall30_true": rec,
                         "top1_pg_accuracy": float(np.mean(t1[common] == s.true_fixed_index[common])),
                         "top1_pg_graph_support_median": float(np.median(ctx.graph_support[np.arange(len(sl)), sl])),
                         "top1_pg_distance_median_um": float(np.median(np.linalg.norm(s.fixed_points[t1] - mov, axis=1))),
                         "radial_g_0_4": float(np.mean(ac.radial_cross_correlation(s.fixed_points, mov, np.array([0, 2, 4])))),
                         "within_3um": ac.nn_stats(mov, s.fixed_points)["within_3um"]})
        log(f"  synthetic {seed} done")
    frames = pd.concat(all_frames, ignore_index=True)
    frames.to_csv(CACHE / "domain_per_nucleus.csv", index=False)
    # verify B2 refit reproduces the stored Real application exactly
    base_b2 = frames.loc[(frames["case"] == "Real saved affine (baseline)") & (frames["method"] == lib.METHOD_B2)]
    st_b2 = stored_real.loc[stored_real["method"] == lib.METHOD_B2]
    repro = {
        "b2_status_counts_stored": st_b2["status"].value_counts().to_dict(),
        "b2_status_counts_reproduced": base_b2["status"].value_counts().to_dict(),
        "b2_status_identical": bool(np.array_equal(base_b2["status"].to_numpy(), st_b2["status"].to_numpy())),
        "b2_confidence_max_abs_diff": float(np.max(np.abs(base_b2["calibrated_confidence"].to_numpy() - st_b2["calibrated_confidence"].to_numpy()))),
        "b2_thresholds": thresholds,
    }
    metric_rows = []
    for (case, method), t in frames.groupby(["case", "method"], sort=False):
        metric_rows.append({"case": case, "method": method,
                            "matched": int((t.status == "matched").sum()), "ambiguous": int((t.status == "ambiguous").sum()),
                            "unmatched": int((t.status == "unmatched").sum()),
                            "graph_support_median": float(t.top1_graph_support.median()),
                            "candidate_distance_median_um": float(t.spatial_distance_um.median()),
                            "minimum_contextual_cost_median": float(t.absolute_min_cost.median()),
                            "row_mass_ratio_median": float(t.row_mass_ratio.median()),
                            "confidence_median": float(t.calibrated_confidence.median()),
                            "reciprocal_rate": float(t.reciprocal.astype(bool).mean())})
    pd.DataFrame(metric_rows).to_csv(CACHE / "domain_method_metrics.csv", index=False)
    pd.DataFrame(summ).to_csv(CACHE / "domain_case_summary.csv", index=False)
    ac.save_json(CACHE / "domain.json", repro)
    return repro


# =========================================================================== F. spatial map
def stage_spatial(d: ac.RealData) -> dict:
    log("spatial residual map")
    corr = np.load(CACHE / "corrected_points.npz")
    partners = np.load(CACHE / "partners.npz")
    ncc = np.load(CACHE / "ncc_cell_offsets.npz")
    dom = pd.read_csv(CACHE / "domain_per_nucleus.csv", low_memory=False)
    base = dom.loc[dom["case"] == "Real saved affine (baseline)"]
    b0 = base.loc[base.method == "Position + Graph"].set_index("moving_index")
    b2 = base.loc[base.method == "Separated Matching / Adoption"].set_index("moving_index")
    b3 = base.loc[base.method == "Full contextual UOT"].set_index("moving_index")
    cor = dom.loc[(dom["case"] == "Real corrected affine (diagnostic)") & (dom.method == "Position + Graph")].set_index("moving_index")
    tree = cKDTree(d.fixed)
    dist, j = tree.query(d.moving)
    nnv = d.fixed[j] - d.moving
    white = ac.he_white_mask(d.he_image, d.tissue_mask)
    wpix = np.argwhere(white)
    rows = []
    step = 100.0
    for x0 in np.arange(0, 1000, step):
        for y0 in np.arange(0, 1000, step):
            sel = (d.moving[:, 0] >= x0) & (d.moving[:, 0] < x0 + step) & (d.moving[:, 1] >= y0) & (d.moving[:, 1] < y0 + step)
            fsel = (d.fixed[:, 0] >= x0) & (d.fixed[:, 0] < x0 + step) & (d.fixed[:, 1] >= y0) & (d.fixed[:, 1] < y0 + step)
            if sel.sum() < 5:
                continue
            idx = np.flatnonzero(sel)
            po = partners["saved_offsets"][idx]
            rows.append({
                "block_x0": x0, "block_y0": y0, "center_x": x0 + step / 2, "center_y": y0 + step / 2,
                "n_moving": int(sel.sum()), "n_fixed": int(fsel.sum()),
                "moving_density_per_um2": sel.sum() / step**2, "fixed_density_per_um2": fsel.sum() / step**2,
                "nn_distance_median_um": float(np.median(dist[idx])),
                "nn_vector_dx_median": float(np.median(nnv[idx, 0])), "nn_vector_dy_median": float(np.median(nnv[idx, 1])),
                "correlogram_offset_dx": float(np.median(po[:, 0])), "correlogram_offset_dy": float(np.median(po[:, 1])),
                "ncc18_offset_dx_median": float(np.median(ncc["saved_affine_18"][idx, 0])),
                "ncc18_offset_dy_median": float(np.median(ncc["saved_affine_18"][idx, 1])),
                "ncc18_boundary_hit_rate": float(ncc["saved_affine_18__hit"][idx].mean()),
                "ncc54_offset_dx_median": float(np.median(ncc["saved_affine_54"][idx, 0])),
                "ncc54_offset_dy_median": float(np.median(ncc["saved_affine_54"][idx, 1])),
                "ncc54_boundary_hit_rate": float(ncc["saved_affine_54__hit"][idx].mean()),
                "candidate_distance_median_um": float(b0.loc[idx, "spatial_distance_um"].median()),
                "graph_support_median": float(b0.loc[idx, "top1_graph_support"].median()),
                "graph_support_corrected_median": float(cor.loc[idx, "top1_graph_support"].median()),
                "uot_row_mass_full_median": float(b3.loc[idx, "row_mass_ratio"].median()),
                "uot_row_mass_b1_median": float(b2.loc[idx, "row_mass_ratio"].median()),
                "b2_confidence_median": float(b2.loc[idx, "calibrated_confidence"].median()),
                "putative_partner_rate": float((partners["saved"][idx] >= 0).mean()),
                "corrected_partner_rate": float((partners["corrected"][idx] >= 0).mean()),
            })
    sp = pd.DataFrame(rows)
    sp.to_csv(ac.OUT / "spatial_residual_metrics.csv", index=False)
    from scipy.stats import spearmanr
    cors = {}
    for col in ("ncc18_boundary_hit_rate", "nn_distance_median_um", "putative_partner_rate", "moving_density_per_um2",
                "candidate_distance_median_um"):
        r, p = spearmanr(sp["graph_support_median"], sp[col])
        cors[f"graph_support_vs_{col}"] = {"spearman_r": float(r), "p_naive": float(p), "n_blocks": len(sp)}
    ac.save_json(CACHE / "spatial.json", cors)
    return cors


STAGES = {"coordinate": stage_coordinate, "correlogram": stage_correlogram, "affine": stage_affine, "he": stage_he,
          "ncc": stage_ncc, "candidates": stage_candidates, "domain": stage_domain, "spatial": stage_spatial}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    args = ap.parse_args()
    d = ac.load_real()
    names = list(STAGES) if args.stage == "all" else args.stage.split(",")
    t0 = time.time()
    for n in names:
        STAGES[n](d)
    log(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
