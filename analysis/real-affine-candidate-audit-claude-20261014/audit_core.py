"""Independent (Claude) audit utilities for the Real affine / coordinate / candidate audit.

Analysis-only. Frozen production and prior-analysis modules are imported read-only;
nothing outside this directory is written.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_opening, distance_transform_edt, gaussian_filter, map_coordinates
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

ROOT = Path("/mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu")
OUT = ROOT / "analysis/real-affine-candidate-audit-claude-20261014"
RUN = ROOT / "runs/20260904-161247-698fe6c9"
INPUT = RUN / "input"
SWEEP_INPUT = ROOT / "sweeps/real-grid27-20260908-v2/input"
CONTEXT_SRC = ROOT / "analysis/contextual-graph-uot-20261008"
COST_SRC = ROOT / "analysis/contextual-uot-cost-ablation-20261008"
SEP_SRC = ROOT / "analysis/graph-matching-adoption-separation-20261014"
CODEX_AUDIT = ROOT / "analysis/real-affine-candidate-audit-20261014"

for _p in (str(ROOT), str(CONTEXT_SRC), str(COST_SRC), str(SEP_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.pointset_registration import (  # noqa: E402
    affine_icp,
    affine_lstsq,
    apply_affine,
    apply_axis_flips,
    best_of_initial_rotations,
)
from config import CANDIDATES, GRAPH, REGIONAL  # noqa: E402
import regional_pattern as frozen_regional  # noqa: E402


# --------------------------------------------------------------------------- data
@dataclass(frozen=True)
class RealData:
    fixed_all: np.ndarray          # 1979 GeoJSON world xy
    fixed: np.ndarray              # 1830 evaluation subset
    moving_raw_px: np.ndarray      # 1842 normalized HE pixel xy
    moving_affine_all: np.ndarray  # 1842 saved affine-aligned
    moving: np.ndarray             # 1754 evaluation subset (saved affine)
    fixed_orig_index: np.ndarray   # subset row -> original fixed row
    moving_orig_index: np.ndarray  # subset row -> original moving row
    affine_matrix: np.ndarray
    translation: np.ndarray
    flip_x: bool
    flip_y: bool
    width: float
    height: float
    he_image: np.ndarray
    tissue_mask: np.ndarray
    warp_meta: dict


def _exact_row_index(subset: np.ndarray, full: np.ndarray) -> np.ndarray:
    tree = cKDTree(full)
    dist, idx = tree.query(subset, k=1)
    if np.max(dist) != 0.0:
        raise AssertionError(f"subset rows not exactly present in full array (max {np.max(dist)})")
    if len(np.unique(idx)) != len(idx):
        raise AssertionError("subset maps to duplicated original rows")
    return idx.astype(int)


def load_real() -> RealData:
    cfg = json.loads((RUN / "config.json").read_text())
    aff = cfg["affine"]
    fixed_all = np.load(INPUT / "original_fixed_points.npy")
    fixed = np.load(INPUT / "success_metric_fixed_points.npy")
    moving_raw = np.load(INPUT / "original_moving_points.npy")
    moving_aff = np.load(INPUT / "moving_points.npy")
    moving = np.load(INPUT / "success_metric_moving_points.npy")
    return RealData(
        fixed_all=fixed_all,
        fixed=fixed,
        moving_raw_px=moving_raw,
        moving_affine_all=moving_aff,
        moving=moving,
        fixed_orig_index=_exact_row_index(fixed, fixed_all),
        moving_orig_index=_exact_row_index(moving, moving_aff),
        affine_matrix=np.asarray(aff["matrix"], float),
        translation=np.asarray(aff["translation"], float),
        flip_x=bool(aff["flip_x"]),
        flip_y=bool(aff["flip_y"]),
        width=float(aff["image_width"]),
        height=float(aff["image_height"]),
        he_image=np.load(INPUT / "he_image.npy"),
        tissue_mask=np.load(INPUT / "tissue_mask.npy"),
        warp_meta=json.loads((INPUT / "metadata.json").read_text()),
    )


def oriented(data: RealData, raw: np.ndarray | None = None) -> np.ndarray:
    raw = data.moving_raw_px if raw is None else raw
    return apply_axis_flips(raw, image_width=data.width, image_height=data.height,
                            flip_x=data.flip_x, flip_y=data.flip_y)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def candidate_hash(indices: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(indices).view(np.uint8)).hexdigest()


# --------------------------------------------------------------------------- affine
def decompose_affine(matrix: np.ndarray) -> dict:
    u, s, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    stretch = vt.T @ np.diag(s) @ vt
    if np.linalg.det(rotation) < 0:  # reflection carried by the polar factor
        reflection = True
    else:
        reflection = False
    angle = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    return {
        "determinant": float(np.linalg.det(matrix)),
        "singular_values": s.tolist(),
        "anisotropy_ratio": float(s[0] / s[1]),
        "polar_rotation_deg": angle,
        "polar_factor_is_reflection": reflection,
        "symmetric_stretch": stretch.tolist(),
        "shear_offdiag": float(stretch[0, 1]),
    }


def world_to_warped_pixel(points: np.ndarray, meta: dict) -> np.ndarray:
    """Independent re-derivation of world -> warped-HE pixel (lower-left origin metadata)."""
    if meta["output_origin"] != "lower-left":
        raise NotImplementedError(meta["output_origin"])
    px = float(meta["output_pixel_size_um"])
    cols = (points[:, 0] - meta["col0_world_x"]) / px
    rows = (meta["row0_world_y"] - points[:, 1]) / px
    return np.column_stack([cols, rows])


def nn_stats(moving: np.ndarray, fixed: np.ndarray) -> dict:
    d_mf, i_mf = cKDTree(fixed).query(moving, k=1)
    d_fm, i_fm = cKDTree(moving).query(fixed, k=1)
    mutual = i_fm[i_mf] == np.arange(len(moving))
    return {
        "n_moving": int(len(moving)),
        "n_fixed": int(len(fixed)),
        "moving_to_fixed_nn_median_um": float(np.median(d_mf)),
        "fixed_to_moving_nn_median_um": float(np.median(d_fm)),
        "within_2um": float(np.mean(d_mf <= 2.0)),
        "within_3um": float(np.mean(d_mf <= 3.0)),
        "within_5um": float(np.mean(d_mf <= 5.0)),
        "within_10um": float(np.mean(d_mf <= 10.0)),
        "mutual_nn_rate": float(np.mean(mutual)),
        "mutual_nn_within_3um_rate": float(np.mean(mutual & (d_mf <= 3.0))),
    }


def icp_trimmed_objective(src_world: np.ndarray, dst: np.ndarray, q: float = 0.7) -> float:
    """The affine-stage ICP objective (trimmed mean NN distance) evaluated at a fixed pose."""
    d, _ = cKDTree(dst).query(src_world, k=1)
    return float(np.mean(d[d <= np.quantile(d, q)]))


# --------------------------------------------------------------------------- nulls
def null_transforms(points: np.ndarray) -> dict[str, np.ndarray]:
    """Pattern-decorrelating nulls that keep the moving density/cardinality."""
    c = np.mean(points, axis=0)
    mx = points.copy(); mx[:, 0] = 2 * c[0] - points[:, 0]
    my = points.copy(); my[:, 1] = 2 * c[1] - points[:, 1]
    r180 = 2 * c - points
    return {"mirror_x": mx, "mirror_y": my, "rotate_180": r180}


# --------------------------------------------------------------------------- correlogram
def displacement_peak(fixed: np.ndarray, moving_sel: np.ndarray, *, search_um: float,
                      kernel_um: float = 3.0, step_um: float = 1.0, density: float) -> dict:
    """Locate the dominant fixed-minus-moving displacement in a window of moving points.

    Counts pairs whose displacement lies within kernel_um of each grid offset. The
    expected count with no shared structure is n_moving * density * pi * kernel^2.
    """
    n = len(moving_sel)
    out = {"n_moving": int(n)}
    if n == 0:
        return out | {"peak_dx": np.nan, "peak_dy": np.nan, "peak_count": 0, "second_count": 0,
                      "background": np.nan, "peak_ratio": np.nan, "refined_dx": np.nan, "refined_dy": np.nan}
    tree = cKDTree(fixed)
    lists = tree.query_ball_point(moving_sel, search_um + kernel_um)
    disp = [fixed[p] - moving_sel[i] for i, p in enumerate(lists) if p]
    disp = np.concatenate(disp) if disp else np.zeros((0, 2))
    axis = np.arange(-search_um, search_um + 0.5 * step_um, step_um)
    grid = np.array([(x, y) for x in axis for y in axis])
    counts = np.array([len(v) for v in cKDTree(disp).query_ball_point(grid, kernel_um)]) if len(disp) else np.zeros(len(grid))
    order = np.argsort(counts, kind="stable")[::-1]
    best = order[0]
    far = order[np.hypot(*(grid[order] - grid[best]).T) > 2.5 * kernel_um]
    second = far[0] if len(far) else best
    near = disp[np.hypot(*(disp - grid[best]).T) <= kernel_um]
    refined = near.mean(axis=0) if len(near) else grid[best]
    background = n * density * np.pi * kernel_um**2
    return out | {
        "peak_dx": float(grid[best, 0]), "peak_dy": float(grid[best, 1]),
        "peak_count": int(counts[best]), "second_count": int(counts[second]),
        "background": float(background), "peak_ratio": float(counts[best] / background),
        "second_ratio": float(counts[second] / background),
        "refined_dx": float(refined[0]), "refined_dy": float(refined[1]),
        "peak_on_search_boundary": bool(np.max(np.abs(grid[best])) >= search_um - 1e-9),
    }


def radial_cross_correlation(fixed: np.ndarray, moving: np.ndarray, edges: np.ndarray,
                             ring=(40.0, 58.0), bin_um: float = 2.0) -> np.ndarray:
    """g_FM(r) normalised by the mean 2-D displacement-histogram level in a distant ring."""
    R = float(edges[-1] if edges[-1] > ring[1] else ring[1])
    lists = cKDTree(fixed).query_ball_point(moving, R)
    d = np.concatenate([fixed[p] - moving[i] for i, p in enumerate(lists) if p])
    nb = int(round(2 * R / bin_um))
    H, xe, ye = np.histogram2d(d[:, 0], d[:, 1], bins=nb, range=[[-R, R], [-R, R]])
    xc = 0.5 * (xe[1:] + xe[:-1])
    rr = np.hypot(*np.meshgrid(xc, xc, indexing="ij"))
    level = H[(rr > ring[0]) & (rr < ring[1])].mean()
    return np.array([H[(rr >= a) & (rr < b)].mean() / level for a, b in zip(edges[:-1], edges[1:])])


# --------------------------------------------------------------------------- HE structure
def he_white_mask(he: np.ndarray, footprint: np.ndarray, *, value: float = 190.0, sat: float = 0.25) -> np.ndarray:
    he = he.astype(float)
    mx, mn = he.max(axis=2), he.min(axis=2)
    saturation = (mx - mn) / np.maximum(mx, 1.0)
    white = (he.mean(axis=2) > value) & (saturation < sat) & footprint
    return binary_opening(white, iterations=2)


def fraction_in_mask(points_world: np.ndarray, mask: np.ndarray, meta: dict) -> tuple[float, int]:
    pix = world_to_warped_pixel(points_world, meta)
    c = np.rint(pix[:, 0]).astype(int); r = np.rint(pix[:, 1]).astype(int)
    ok = (r >= 0) & (r < mask.shape[0]) & (c >= 0) & (c < mask.shape[1])
    return float(mask[r[ok], c[ok]].mean()), int(ok.sum())


# --------------------------------------------------------------------------- regional NCC
@dataclass
class NCCResult:
    radius_um: float
    anchors: np.ndarray
    best_offsets: np.ndarray
    best_ncc: np.ndarray
    second_ncc: np.ndarray
    margin: np.ndarray
    boundary_x: np.ndarray
    boundary_y: np.ndarray
    interior_local_max: np.ndarray
    cell_anchor_index: np.ndarray
    cell_offsets: np.ndarray
    cell_confidence: np.ndarray
    peak_curvature: np.ndarray


def regional_ncc(fixed: np.ndarray, moving: np.ndarray, radius_um: float) -> NCCResult:
    """Vectorised re-implementation of frozen `match_regional_patterns` with a configurable radius.

    The density channels, anchors, patch sampling and NCC definition are identical to the
    frozen code; only the search radius is a parameter. Canvas padding is patch+R+4, i.e.
    exactly what the frozen code computes if its configured radius were R. For R in
    {18,36,54,72} the padding differs by multiples of the 2 um pixel, so impulse rounding
    and therefore the density channels at a given world position are unchanged.
    """
    fixed = np.asarray(fixed, float); moving = np.asarray(moving, float)
    pixel = float(REGIONAL["pixel_size_um"])
    patch_r = float(REGIONAL["patch_radius_um"])
    step = float(REGIONAL["search_step_um"])
    padding = patch_r + radius_um + 4.0
    bounds = frozen_regional._bounds(fixed, moving, padding)
    fch = frozen_regional._channels(fixed, bounds)
    mch = frozen_regional._channels(moving, bounds)
    anchors = frozen_regional._anchors(moving)
    off1 = np.arange(-radius_um, radius_um + 0.5 * step, step)
    offsets = np.asarray([(x, y) for y in off1 for x in off1], float)
    patch_axis = np.arange(-patch_r, patch_r + 0.5 * pixel, pixel)  # 35 samples, 2 um
    # every fixed sample sits at anchor + integer microns -> sample a 1 um grid once
    span = int(round(patch_r + radius_um))
    grid1 = np.arange(-span, span + 1, 1.0)
    pi = np.rint(patch_axis).astype(int) + span                     # patch index into grid1
    oi = np.rint(off1).astype(int)                                   # offset shift in grid1 units
    n_off1 = len(off1)
    best_o, best_s, sec_s, margins, bx, by, interior, curv = [], [], [], [], [], [], [], []
    for a in anchors:
        xx, yy = np.meshgrid(patch_axis, patch_axis)
        mp = np.stack([map_coordinates(c, [(a[1] + yy - bounds[1]) / pixel, (a[0] + xx - bounds[0]) / pixel],
                                       order=1, mode="constant", cval=0.0) for c in mch])
        gx, gy = np.meshgrid(grid1, grid1)
        fg = np.stack([map_coordinates(c, [(a[1] + gy - bounds[1]) / pixel, (a[0] + gx - bounds[0]) / pixel],
                                       order=1, mode="constant", cval=0.0) for c in fch])
        # patches[c, oy, ox, i, j] = fg[c, pi[i] + oi[oy], pi[j] + oi[ox]]
        rows = (pi[None, :] + oi[:, None])            # (n_off1, 35)
        fp = fg[:, rows[:, None, :, None], rows[None, :, None, :]]  # (C, oy, ox, 35, 35)
        mc = mp - mp.mean(axis=(1, 2), keepdims=True)
        mn = np.linalg.norm(mc.reshape(mc.shape[0], -1), axis=1)
        fc = fp - fp.mean(axis=(3, 4), keepdims=True)
        fn = np.sqrt(np.sum(fc * fc, axis=(3, 4)))
        num = np.einsum("cyxij,cij->cyx", fc, mc)
        den = fn * mn[:, None, None]
        per = np.where(den > 1e-12, num / np.where(den > 1e-12, den, 1.0), -1.0)
        scores = per.mean(axis=0).reshape(-1)           # y-major like the frozen code
        order = np.argsort(scores)[::-1]
        b = int(order[0]); bo = offsets[b]
        sep = [int(i) for i in order[1:] if np.max(np.abs(offsets[i] - bo)) > 1.01 * step]
        s2 = sep[0] if sep else int(order[1])
        best_o.append(bo); best_s.append(scores[b]); sec_s.append(scores[s2]); margins.append(scores[b] - scores[s2])
        bx.append(abs(abs(bo[0]) - radius_um) < 1e-9); by.append(abs(abs(bo[1]) - radius_um) < 1e-9)
        surf = scores.reshape(n_off1, n_off1)
        inner = surf[1:-1, 1:-1]
        neigh = np.max(np.stack([surf[:-2, 1:-1], surf[2:, 1:-1], surf[1:-1, :-2], surf[1:-1, 2:],
                                 surf[:-2, :-2], surf[2:, 2:], surf[:-2, 2:], surf[2:, :-2]]), axis=0)
        interior.append(bool(np.any(inner > neigh)))
        iy, ix = divmod(b, n_off1)
        if 0 < iy < n_off1 - 1 and 0 < ix < n_off1 - 1:
            lap = 4 * surf[iy, ix] - surf[iy - 1, ix] - surf[iy + 1, ix] - surf[iy, ix - 1] - surf[iy, ix + 1]
        else:
            lap = np.nan
        curv.append(lap)
    best_o = np.asarray(best_o); best_s = np.asarray(best_s); margins = np.asarray(margins)
    cell_anchor = cKDTree(anchors).query(moving, k=1)[1].astype(int)
    conf = np.clip(0.5 * (best_s + 1.0) + 2.0 * np.maximum(margins, 0.0), 0.0, 1.0)
    return NCCResult(radius_um, anchors, best_o, best_s, np.asarray(sec_s), margins,
                     np.asarray(bx), np.asarray(by), np.asarray(interior), cell_anchor,
                     best_o[cell_anchor], conf[cell_anchor], np.asarray(curv))


# --------------------------------------------------------------------------- graph support
def mutual_knn_lists(points: np.ndarray, k: int) -> list[np.ndarray]:
    k = min(int(k), len(points) - 1)
    nb = cKDTree(points).query(points, k=k + 1)[1][:, 1:]
    sets = [set(map(int, r)) for r in nb]
    return [np.asarray([j for j in r if i in sets[int(j)]], dtype=int) for i, r in enumerate(nb)]


def graph_support_fast(fixed: np.ndarray, moving: np.ndarray, indices: np.ndarray, base_cost: np.ndarray) -> np.ndarray:
    """Vectorised equivalent of frozen `graph_support_scores` (validated against it)."""
    mg = mutual_knn_lists(moving, GRAPH["mutual_knn"])
    fg = mutual_knn_lists(fixed, GRAPH["mutual_knn"])
    nf = len(fixed)
    adj = np.zeros((nf, nf), dtype=bool)
    for i, nb in enumerate(fg):
        adj[i, nb] = True
    has_fixed_nb = adj.any(axis=1)
    lt = float(GRAPH["length_tolerance_um"]); at = float(GRAPH["angle_tolerance_radians"])
    compat_all = np.exp(-0.35 * np.minimum(base_cost, 10.0))
    out = np.zeros(indices.shape, float)
    for i, nbrs in enumerate(mg):
        if len(nbrs) == 0:
            continue
        cand = indices[i]                       # (K,)
        nc = indices[nbrs]                      # (J, K2)
        mv = moving[nbrs] - moving[i]           # (J, 2)
        ml = np.maximum(np.linalg.norm(mv, axis=1), 1e-6)
        fv = fixed[nc][None, :, :, :] - fixed[cand][:, None, None, :]   # (K, J, K2, 2)
        fl = np.maximum(np.linalg.norm(fv, axis=3), 1e-6)
        cos = np.clip(np.einsum("kjtd,jd->kjt", fv, mv) / (ml[None, :, None] * fl), -1.0, 1.0)
        geo = np.exp(-0.5 * ((np.abs(fl - ml[None, :, None])) / lt) ** 2) * np.exp(-0.5 * (np.arccos(cos) / at) ** 2)
        val = geo * compat_all[nbrs][None, :, :]
        member = adj[cand][:, nc]               # (K, J, K2)
        val = np.where(member, val, 0.0)
        best = val.max(axis=2)                  # (K, J)
        J = best.shape[1]
        keep = max(2, int(np.ceil(0.6 * J)))
        srt = np.sort(best, axis=1)[:, -keep:]
        score = 0.5 * np.mean(best >= 0.30, axis=1) + 0.5 * srt.mean(axis=1)
        score[~has_fixed_nb[cand]] = 0.0
        out[i] = score
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------- misc
def json_safe(v):
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, np.ndarray):
        return json_safe(v.tolist())
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    return v


def save_json(path: Path, value) -> None:
    Path(path).write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False) + "\n")
