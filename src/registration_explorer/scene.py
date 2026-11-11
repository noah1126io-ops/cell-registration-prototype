"""Build viewer scenes and UI-side diagnostics in a single display frame (the chosen fixed modality).

Conventions: transforms map moving world -> fixed world (points, forward).  Images are placed by
pixel -> world affines (corner at -0.5 * level0 spacing); warp previews use the inverse map (imaging.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from . import imaging, viewer
from .catalog import ModalityInfo, SampleInfo, load_nuclei

COLORS = {"HE": "#ff4fa3", "IF": "#40e0ff"}


@dataclass
class ImageOptions:
    target_um: float = 1.6           # overview display resolution
    he_mode: str = "rgb"             # rgb | nuclear
    if_mode: str = "rgb"
    roi: tuple | None = None         # (cx, cy, size) in display-frame world units, high-res detail
    roi_um: float = 0.4


def footprint(extent):
    w, h = extent
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], float)


def hull(points):
    if len(points) < 3:
        return None
    h = ConvexHull(points)
    return points[h.vertices]


def apply(M, t, p):
    return np.asarray(p, float) @ np.asarray(M, float).T + np.asarray(t, float)


# ----------------------------------------------------------------------------- diagnostics
def mirror_null_rate(mapped, fixed_eval, r=3.0):
    """Chance level: mapped moving nuclei mirrored about the vertical axis through their centroid."""
    if len(mapped) < 3 or len(fixed_eval) == 0:
        return None
    cx = mapped[:, 0].mean()
    mir = mapped.copy(); mir[:, 0] = 2 * cx - mir[:, 0]
    d, _ = cKDTree(mir).query(fixed_eval)
    return float(np.mean(d <= r))


def ui_diagnostics(fixed_pts, moving_mapped, fixed_extent, moving_fp_mapped):
    """UI-recomputed geometry (fixed frame): fixed nuclei inside the mapped moving footprint vs mapped moving nuclei.

    Population differs from the saved research metrics (IF frame); shown separately and never used for acceptance.
    """
    from matplotlib.path import Path as MPath
    out = {"population": "fixed nuclei inside the mapped moving footprint (fixed frame), UI-recomputed"}
    sel = MPath(moving_fp_mapped).contains_points(fixed_pts) if moving_fp_mapped is not None else np.ones(len(fixed_pts), bool)
    if fixed_extent is not None:
        sel &= MPath(footprint(fixed_extent)).contains_points(fixed_pts)
    fe = fixed_pts[sel]
    if len(fe) < 5 or len(moving_mapped) < 5:
        out["n_eval"] = int(len(fe)); return out
    d, _ = cKDTree(moving_mapped).query(fe)
    out.update({"n_eval": int(len(fe)), "nn_median": float(np.median(d)), "nn_p95": float(np.percentile(d, 95)),
                "le_1": float(np.mean(d <= 1)), "le_3": float(np.mean(d <= 3)), "le_5": float(np.mean(d <= 5)),
                "null_le_3": mirror_null_rate(moving_mapped, fe)})
    return out


def residual_vectors(moving_mapped, fixed_pts, max_d=10.0, mutual=True, limit=6000):
    """Estimated correspondences: mutual nearest neighbours within max_d (NOT proven same cells)."""
    if len(moving_mapped) == 0 or len(fixed_pts) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    tf, tm = cKDTree(fixed_pts), cKDTree(moving_mapped)
    d, j = tf.query(moving_mapped)
    ok = d <= max_d
    if mutual:
        _, back = tm.query(fixed_pts[j])
        ok &= back == np.arange(len(moving_mapped))
    idx = np.where(ok)[0][:limit]
    return moving_mapped[idx], fixed_pts[j[idx]], d[idx]


# ----------------------------------------------------------------------------- tissue contours
def tissue_contours(mod: ModalityInfo, max_contours=40):
    if not mod.tissue_npz:
        return []
    try:
        from skimage.measure import find_contours
        z = np.load(mod.tissue_npz, allow_pickle=False)
        env = z["map_envelope"].astype(float)
        import json
        res = json.loads(str(z["meta"]))["res_um"]
    except Exception:  # noqa: BLE001 - optional overlay
        return []
    cs = sorted(find_contours(env, 0.5), key=len, reverse=True)[:max_contours]
    return [np.c_[c[::3, 1], c[::3, 0]] * res for c in cs if len(c) > 20]


# ----------------------------------------------------------------------------- images
def overview_level(mod: ModalityInfo, target_um):
    info = imaging.pyramid_info(mod.image_path)
    lvl = imaging.pick_level(info, mod.pixel_um, target_um)
    # never read a whole level above the safety limit: step to a coarser level
    while lvl < len(info.level_shapes) - 1 and np.prod(info.level_shapes[lvl][:2]) > imaging.MAX_FULL_LEVEL_PIXELS:
        lvl += 1
    return lvl, mod.pixel_um * info.downsamples[lvl]


def overview_image(mod: ModalityInfo, target_um, mode):
    lvl, um_k = overview_level(mod, target_um)
    raw = imaging.read_level_cached(mod.image_path, lvl)
    return imaging.to_display_rgb(raw, mod.kind, mode), um_k, lvl


@lru_cache(maxsize=32)
def overview_jpeg(path: str, kind: str, pixel_um: float, level: int, mode: str):
    raw = imaging.read_level_cached(path, level)
    rgb = imaging.to_display_rgb(raw, kind, mode)
    return imaging.encode_jpeg(rgb), rgb.shape[1], rgb.shape[0]


def roi_image(mod: ModalityInfo, M_disp, t_disp, roi, roi_um, mode, max_px=1600):
    """High-res crop of the region of this modality that maps into the display ROI (tile reads only)."""
    cx, cy, size = roi
    corners_disp = np.array([[cx - size / 2, cy - size / 2], [cx + size / 2, cy - size / 2], [cx + size / 2, cy + size / 2], [cx - size / 2, cy + size / 2]])
    Mi = np.linalg.inv(M_disp); ti = -Mi @ t_disp
    src = apply(Mi, ti, corners_disp)
    info = imaging.pyramid_info(mod.image_path)
    lvl = imaging.pick_level(info, mod.pixel_um, roi_um)
    um_k = mod.pixel_um * info.downsamples[lvl]
    corner = imaging.level_corner_um(mod.pixel_um)
    lo = np.floor((src.min(0) - corner) / um_k).astype(int); hi = np.ceil((src.max(0) - corner) / um_k).astype(int)
    while lvl < len(info.level_shapes) - 1 and np.prod(hi - lo) > max_px * max_px:
        lvl += 1; um_k = mod.pixel_um * info.downsamples[lvl]
        lo = np.floor((src.min(0) - corner) / um_k).astype(int); hi = np.ceil((src.max(0) - corner) / um_k).astype(int)
    h, w = info.level_shapes[lvl][:2]
    lo = np.clip(lo, 0, [w, h]); hi = np.clip(hi, 0, [w, h])
    if np.any(hi - lo < 2):
        return None
    raw = imaging.read_region(mod.image_path, lvl, lo[0], lo[1], hi[0], hi[1])
    return imaging.to_display_rgb(raw, mod.kind, mode), um_k, lvl, lo


def build_scene(sample: SampleInfo, avail: dict, fixed_key: str, M_disp, t_disp, *, state_label: str, banners: list,
                opts: ImageOptions, candidate_polys=(), selected_poly=None, show_vectors=True):
    """M_disp, t_disp: moving world -> display (fixed) world, already oriented; identity for 'Before'."""
    moving_key = sample.other(fixed_key)
    fm, mm = sample.modality(fixed_key), sample.modality(moving_key)
    M_disp = np.asarray(M_disp, float); t_disp = np.asarray(t_disp, float)
    images, points, polys, notes = [], [], [], []
    fixed_pts, _ = load_nuclei(fm)
    moving_pts, _ = load_nuclei(mm)
    moving_mapped = apply(M_disp, t_disp, moving_pts)
    for key, mod, role in ((fixed_key, fm, "fixed"), (moving_key, mm, "moving")):
        if not avail[key]["image"]:
            notes.append(f"{mod.label}: image not used ({avail[key]['image_reason']})")
            continue
        try:
            lvl, um_k = overview_level(mod, opts.target_um)
            jpg, w_px, h_px = overview_jpeg(mod.image_path, mod.kind, mod.pixel_um, lvl, opts.he_mode if key == "HE" else opts.if_mode)
        except imaging.ImageUnavailable as exc:
            avail[key]["image"] = False; avail[key]["image_reason"] = f"read failed: {exc}"
            notes.append(f"{mod.label}: IMAGE_UNUSABLE ({exc})")
            continue
        c = imaging.level_corner_um(mod.pixel_um)
        A = um_k * np.eye(2); t0 = np.array([c, c])
        if role == "moving":
            A, t0 = M_disp @ A, M_disp @ t0 + t_disp
        images.append(viewer.image_layer(role, f"{mod.label} ({role}, level {lvl}, {um_k:.2f} µm/px)", jpg, w_px, h_px, A, t0, role))
        if opts.roi is not None:
            Mr, tr = (np.eye(2), np.zeros(2)) if role == "fixed" else (M_disp, t_disp)
            got = roi_image(mod, Mr, tr, opts.roi, opts.roi_um, opts.he_mode if key == "HE" else opts.if_mode)
            if got is not None:
                rgb2, um2, lvl2, lo = got
                A2 = um2 * np.eye(2); t2 = np.array([c, c]) + lo * um2
                if role == "moving":
                    A2, t2 = M_disp @ A2, M_disp @ t2 + t_disp
                images.append(viewer.image_layer(role + "_roi", f"{mod.label} ROI detail (level {lvl2}, {um2:.2f} um/px)", imaging.encode_jpeg(rgb2, 90),
                                                 rgb2.shape[1], rgb2.shape[0], A2, t2, role))
    points.append(viewer.points_layer("fixed_nuclei", f"{fm.label} 核 (fixed)", fixed_pts, COLORS[fixed_key]))
    points.append(viewer.points_layer("moving_nuclei", f"{mm.label} 核 (moving, mapped)", moving_mapped, COLORS[moving_key]))
    fe, me = fm.extent(), mm.extent()
    mfp = apply(M_disp, t_disp, footprint(me)) if me else None
    if fe:
        polys.append(viewer.poly_layer("fov_fixed", f"Image FOV: {fm.label} (fixed)", footprint(fe), "#ffffff", dash=[6, 4]))
    if mfp is not None:
        polys.append(viewer.poly_layer("fov_moving", f"Image FOV: {mm.label} (moving, mapped)", mfp, "#ffd54f", dash=[6, 4]))
    hf, hm = hull(fixed_pts), hull(moving_mapped)
    if hf is not None:
        polys.append(viewer.poly_layer("hull_fixed", f"核検出範囲: {fm.label}", hf, COLORS[fixed_key], dash=[2, 3], visible=False))
    if hm is not None:
        polys.append(viewer.poly_layer("hull_moving", f"核検出範囲: {mm.label} (mapped)", hm, COLORS[moving_key], dash=[2, 3], visible=False))
    for key, mod in ((fixed_key, fm), (moving_key, mm)):
        if not avail[key]["image"]:
            continue
        for i, c in enumerate(tissue_contours(mod)):
            cc = c if key == fixed_key else apply(M_disp, t_disp, c)
            polys.append(viewer.poly_layer(f"tissue_{key}_{i}", f"組織領域: {mod.label}" if i == 0 else "", cc, "#a5d6a7" if key == "HE" else "#80cbc4",
                                           visible=False, width=1.0, closed=False))
    for i, item in enumerate(candidate_polys):       # (label, poly[, color[, pick id[, dash]]])
        lab, poly = item[0], item[1]
        color = item[2] if len(item) > 2 else "#b39ddb"
        pick = item[3] if len(item) > 3 else None
        dash = item[4] if len(item) > 4 else [3, 3]
        polys.append(viewer.poly_layer(f"cand_{i}", lab, poly, color, dash=dash, visible=True, width=1.2, pick=pick))
    if selected_poly is not None:
        polys.append(viewer.poly_layer("selected", "選択中の対応領域", selected_poly, "#ff7043", width=3.0, fill="rgba(255,112,67,0.06)"))
    # merge tissue contour polylines into one toggle per modality (separate segments, no joining lines)
    merged, seen = [], {}
    for p in polys:
        if p["id"].startswith("tissue_"):
            k = p["id"].split("_")[1]
            if k in seen:
                seen[k]["multi"].append(p["pts"])
                continue
            p = {**p, "multi": [p["pts"]], "pts": []}
            seen[k] = p
        merged.append(p)
    vec = None
    if show_vectors:
        a, b, d = residual_vectors(moving_mapped, fixed_pts)
        vec = viewer.vectors_layer("vec", "残留変位ベクトル (mutual NN ≤10)", a, b, d, visible=False)
    allp = [footprint(fe)] if fe else [fixed_pts]
    if mfp is not None:
        allp.append(mfp)
    else:
        allp.append(moving_mapped)
    for item in candidate_polys:
        allp.append(np.asarray(item[1]))
    allp = np.concatenate(allp)
    lo, hi = np.nanmin(allp, 0), np.nanmax(allp, 0)
    pad = 0.03 * float(max(hi - lo))
    units = fm.units
    if fm.units != mm.units:
        notes.append(f"単位が異なる（Fixed {fm.units} / Moving {mm.units}）: Before表示は物理的な配置を表しません")
    scene = {"images": images, "points": points, "polys": merged, "vectors": vec, "bounds": [lo[0] - pad, lo[1] - pad, hi[0] + pad, hi[1] + pad],
             "units": fm.units, "frame_name": f"{sample.sample_id}:{fixed_key} (fixed frame)",
             "frame_note": f"{state_label}. 表示座標 = {fm.label} (Fixed) のframe-local world。原点はUNKNOWN（推測で補完しない）。" + (" " + " / ".join(notes) if notes else ""),
             "banners": banners, "default_mode": "overlay", "checker_um": 100}
    diag = ui_diagnostics(fixed_pts, moving_mapped, fe, mfp)
    return scene, diag, {"n_fixed": len(fixed_pts), "n_moving": len(moving_pts), "units_note": units}
