"""Affine global-initialisation research library (analysis-only, 2026-10-14).

All methods share the production refine: `icp_similarity(trim 0.8, 80 it)` followed by
`affine_icp(trim 0.7, 80 it)` from `src/pointset_registration.py` (imported, unmodified).
Only the initialisation differs:

A0  production `estimate_affine_with_y_flip` (centroid/RMS init, 4 flips x 12 rotations)
A1  displacement correlogram (pairwise fixed-moving displacement histogram, local-background z-score)
A2  point-density masked NCC via FFT (Gaussian rasterisation, overlap-normalised)
A3  coarse-to-fine multi-start (chirality x rotation x translation grid, short ICP scoring, top-K full refine)

Transforms are reported in the *effective* form  world = M @ p_raw + t  (p_raw = HE pixel xy).
No value from the Real diagnostic re-fit is used by any method.
"""
from __future__ import annotations

import json
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MPath
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import ConvexHull, cKDTree

ROOT = Path("/mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu")
OUT = ROOT / "analysis/affine-global-initialization-20261014"
CACHE = OUT / "cache"
REAL_RUN = ROOT / "runs/20260904-161247-698fe6c9"
PRIOR_AUDIT = ROOT / "analysis/real-affine-candidate-audit-claude-20261014"
SEP_SRC = ROOT / "analysis/graph-matching-adoption-separation-20261014"
for _p in (str(ROOT), str(ROOT / "analysis/contextual-graph-uot-20261008"),
           str(ROOT / "analysis/contextual-uot-cost-ablation-20261008"), str(SEP_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.pointset_registration import (  # noqa: E402
    SimilarityTransform,
    affine_icp,
    estimate_affine_with_y_flip,
    icp_similarity,
)

# ----------------------------------------------------------------------------- shared settings
SEARCH_RADIUS_UM = 200.0          # translation search half-width for A1/A2/A3 (same for all)
COARSE_ROT_STEP = 4.0             # deg, A1/A2 coarse
FINE_ROT_HALF, FINE_ROT_STEP = 3.0, 1.0
FINE_SCALE = (0.97, 1.0, 1.03)
N_REFINE = 6                      # hypotheses passed to the shared ICP refine (A1/A2)
A3_ROT_STEP = 15.0
A3_TRANS_STEP = 25.0
A3_SHORT_ITERS = 4
A3_TOPK = 12
CHIRALITY = {"proper": np.eye(2), "reflect": np.diag([1.0, -1.0])}

# Pre-specified failure-detector thresholds (fixed before any Real/control evaluation)
DETECT = {
    "inlier_radius_um": 3.0,
    "pass_min_inlier": 0.25,
    "pass_min_inlier_over_null": 3.0,
    "pass_min_block_consistency": 0.6,
    "pass_min_heldout_inlier": 0.20,
    "ambiguous_min_inlier_over_null": 2.0,
    "ambiguous_basin_margin": 0.10,
}


# ----------------------------------------------------------------------------- utilities
def rot(theta_deg: float) -> np.ndarray:
    a = np.deg2rad(theta_deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def apply_eff(M, t, p):
    return np.asarray(p, float) @ np.asarray(M).T + np.asarray(t)


def nn_median(points):
    return float(np.median(cKDTree(points).query(points, k=2)[0][:, 1]))


def rms_radius(points):
    return float(np.sqrt(np.mean(np.sum((points - points.mean(0)) ** 2, axis=1))))


def scale_candidates(fixed, moving_px):
    """Crop-sensitive (RMS, as production) and density-sensitive (NN spacing) scale guesses."""
    s_rms = rms_radius(fixed) / rms_radius(moving_px)
    s_nn = nn_median(fixed) / nn_median(moving_px)
    out = [s_rms]
    if abs(s_nn / s_rms - 1) > 0.03:
        out.append(s_nn)
    return out


@dataclass
class Hypothesis:
    chirality: str
    theta: float
    scale: float
    translation: np.ndarray
    score: float
    source: str
    extra: dict = field(default_factory=dict)


@dataclass
class Result:
    method: str
    M: np.ndarray
    t: np.ndarray
    objective: float
    n_pairs: int
    runtime_s: float
    peak_mem_mb: float
    n_hypotheses_scored: int
    n_refined: int
    refined: list            # [(objective, M, t, hypothesis-dict)]
    diagnostics: dict


def refine(hyp: Hypothesis, moving_px, fixed):
    """Shared production refine from a similarity initialisation."""
    C = CHIRALITY[hyp.chirality]
    src = moving_px @ C.T
    init = SimilarityTransform(scale=float(hyp.scale), rotation_rad=float(np.deg2rad(hyp.theta)),
                               translation=np.asarray(hyp.translation, float))
    sim, _, _ = icp_similarity(src, fixed, initial=init, max_iter=80, trim_quantile=0.8)
    M0, t0 = sim.to_affine()
    M, t, err, n = affine_icp(src, fixed, M0, t0, max_iter=80, trim_quantile=0.7)
    return M @ C, t, float(err), int(n)


def _distinct(results, tol_um=10.0, probe=None):
    """Collapse refined solutions that land in the same basin (median point difference < tol)."""
    kept = []
    for r in sorted(results, key=lambda x: x[0]):
        if all(np.median(np.linalg.norm(apply_eff(r[1], r[2], probe) - apply_eff(k[1], k[2], probe), axis=1)) > tol_um for k in kept):
            kept.append(r)
    return kept


def _finish(method, hyps, moving_px, fixed, t_start, n_scored, n_refine=N_REFINE, diag=None):
    results = []
    for h in hyps[:n_refine]:
        M, t, err, n = refine(h, moving_px, fixed)
        results.append((err, M, t, {"chirality": h.chirality, "theta": h.theta, "scale": h.scale,
                                     "translation": np.asarray(h.translation).tolist(), "score": h.score, "source": h.source}))
    distinct = _distinct(results, probe=moving_px[:: max(1, len(moving_px) // 300)])
    best = distinct[0]
    diag = dict(diag or {})
    diag["basin_objectives"] = [d[0] for d in distinct]
    diag["basin_margin"] = (distinct[1][0] - best[0]) / best[0] if len(distinct) > 1 else np.inf
    return Result(method, best[1], best[2], best[0], 0, time.perf_counter() - t_start, 0.0, n_scored,
                  len(results), [(d[0], d[1].tolist(), d[2].tolist(), d[3]) for d in distinct], diag)


def _transform_hyp(moving_px, chir, theta, scale):
    return scale * (moving_px @ CHIRALITY[chir].T) @ rot(theta).T


# ----------------------------------------------------------------------------- A0
def method_a0(moving_px, fixed, width, height, flip_candidates=None):
    t0 = time.perf_counter()
    kw = {} if flip_candidates is None else {"flip_candidates": flip_candidates}
    r = estimate_affine_with_y_flip(moving_px, fixed, image_height_px=height, image_width_px=width, **kw)
    F = np.diag([-1.0 if r.flip_x else 1.0, -1.0 if r.flip_y else 1.0])
    M = r.affine_matrix @ F
    t = r.translation + r.affine_matrix @ np.array([width if r.flip_x else 0.0, height if r.flip_y else 0.0])
    return Result("A0 centroid+ICP", M, t, float(r.mean_residual), int(r.n_pairs), time.perf_counter() - t0, 0.0,
                  len(flip_candidates or range(4)) * 12, len(flip_candidates or range(4)) * 12,
                  [(float(r.mean_residual), M.tolist(), t.tolist(), {"flip_x": r.flip_x, "flip_y": r.flip_y})],
                  {"basin_margin": np.nan, "flip_x": bool(r.flip_x), "flip_y": bool(r.flip_y)})


# ----------------------------------------------------------------------------- A1
def correlogram_scores(fixed_tree, fixed, q, R, sigma_um, bg_sigma_um=15.0, bin_um=1.0, npeaks=3):
    """Peaks of the fixed-minus-moving displacement histogram, z-scored against a smooth local background.

    The smooth background (sigma 15 um) absorbs density-overlap bias and partial overlap: a peak must be
    a *sharp* excess over the local expectation, not merely a region where two dense areas overlap.
    """
    lists = fixed_tree.query_ball_point(q, R + 4 * sigma_um)
    lens = np.fromiter((len(p) for p in lists), int, len(lists))
    if lens.sum() == 0:
        return []
    idx = np.concatenate([np.asarray(p, int) for p in lists if p])
    rep = np.repeat(np.arange(len(q)), lens)
    disp = fixed[idx] - q[rep]
    nb = int(round(2 * R / bin_um))
    H, _, _ = np.histogram2d(disp[:, 0], disp[:, 1], bins=nb, range=[[-R, R], [-R, R]])
    s = sigma_um / bin_um
    Hk = gaussian_filter(H, s, mode="constant")
    B = gaussian_filter(H, bg_sigma_um / bin_um, mode="constant")
    var = np.maximum(B, 1e-3) / (4 * np.pi * s * s)
    z = (Hk - B) / np.sqrt(var)
    edge = int(np.ceil(2 * bg_sigma_um / bin_um))
    z[:edge] = z[-edge:] = z[:, :edge] = z[:, -edge:] = -np.inf   # background undefined at histogram edge
    loc = (z == maximum_filter(z, size=int(10 / bin_um))) & np.isfinite(z)
    cand = np.argwhere(loc)
    vals = z[loc]
    order = np.argsort(vals)[::-1][:npeaks]
    centers = -R + (np.arange(nb) + 0.5) * bin_um
    return [(float(vals[o]), np.array([centers[cand[o, 0]], centers[cand[o, 1]]]), float(Hk[tuple(cand[o])] / max(B[tuple(cand[o])], 1e-9)))
            for o in order]


def method_a1(moving_px, fixed, R=SEARCH_RADIUS_UM, chiralities=("proper", "reflect")):
    t0 = time.perf_counter()
    tree = cKDTree(fixed)
    scales = scale_candidates(fixed, moving_px)
    coarse = []
    n = 0
    for chir in chiralities:
        for th in np.arange(0, 360, COARSE_ROT_STEP):
            for s in scales:
                q = _transform_hyp(moving_px, chir, th, s)
                tc = fixed.mean(0) - q.mean(0)
                for z, d, ratio in correlogram_scores(tree, fixed, q + tc, R, sigma_um=4.0, npeaks=2):
                    coarse.append(Hypothesis(chir, float(th), s, tc + d, z, "A1 coarse", {"ratio": ratio}))
                n += 1
    coarse.sort(key=lambda h: -h.score)
    fine = []
    seen = set()
    for h in coarse[:8]:
        key = (h.chirality, round(h.theta), round(h.scale, 3))
        if key in seen:
            continue
        seen.add(key)
        for th in np.arange(h.theta - FINE_ROT_HALF, h.theta + FINE_ROT_HALF + 1e-9, FINE_ROT_STEP):
            for f in FINE_SCALE:
                s = h.scale * f
                q = _transform_hyp(moving_px, h.chirality, th, s)
                tc = fixed.mean(0) - q.mean(0)
                for z, d, ratio in correlogram_scores(tree, fixed, q + tc, R, sigma_um=2.0, npeaks=2):
                    fine.append(Hypothesis(h.chirality, float(th), s, tc + d, z, "A1 fine", {"ratio": ratio}))
                n += 1
    fine.sort(key=lambda h: -h.score)
    top = fine[0] if fine else None
    second = next((h for h in fine if np.linalg.norm(h.translation - top.translation) > 15 or h.chirality != top.chirality), None)
    diag = {"peak_z": top.score if top else np.nan, "peak_ratio": top.extra["ratio"] if top else np.nan,
            "second_peak_z": second.score if second else np.nan,
            "coarse_top": [(h.chirality, h.theta, h.scale, h.translation.tolist(), h.score) for h in coarse[:10]]}
    return _finish("A1 correlogram+ICP", fine, moving_px, fixed, t0, n, diag=diag)


# ----------------------------------------------------------------------------- A2
def _raster(points, origin, shape, px, sigma_um):
    img = np.zeros(shape)
    ij = np.rint((points - origin) / px).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[0])
    np.add.at(img, (ij[ok, 1], ij[ok, 0]), 1.0)
    return gaussian_filter(img, sigma_um / px)


def _hull_mask(points, origin, shape, px):
    h = ConvexHull(points)
    path = MPath(points[h.vertices])
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    xy = np.column_stack([xx.ravel() * px + origin[0], yy.ravel() * px + origin[1]])
    return path.contains_points(xy).reshape(shape).astype(float)


def masked_ncc(f, mf, m, mm, min_overlap):
    """Padfield (2012) masked normalised cross-correlation of f (fixed) against m (moving), all shifts.

    Returns NCC[dy, dx] for moving shifted by (dx, dy) pixels (fftshifted, centre = zero shift) and the
    overlap-pixel count; shifts with overlap < min_overlap are set to -inf (empty-region artefact guard).
    """
    from numpy.fft import fft2, ifft2, fftshift
    shp = [2 * s for s in f.shape]
    F = lambda a: fft2(a, shp)  # noqa: E731
    Fm, Ffm, Ff2m = F(mf), F(f * mf), F(f * f * mf)
    # correlation (moving shifted onto fixed) = conv(fixed, flipped moving)
    mr, mmr = m[::-1, ::-1], mm[::-1, ::-1]
    Gm, Gmm, Gm2m = F(mmr), F(mr * mmr), F(mr * mr * mmr)
    ov = np.real(ifft2(Fm * Gm))
    sf = np.real(ifft2(Ffm * Gm)); sm = np.real(ifft2(Fm * Gmm))
    sff = np.real(ifft2(Ff2m * Gm)); smm = np.real(ifft2(Fm * Gm2m)); sfm = np.real(ifft2(Ffm * Gmm))
    ov = np.maximum(np.rint(ov), 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        num = sfm - sf * sm / ov
        den = np.sqrt(np.maximum(sff - sf**2 / ov, 0) * np.maximum(smm - sm**2 / ov, 0))
        ncc = np.where((ov >= min_overlap) & (den > 1e-9), num / den, -np.inf)
    # index k of the full correlation corresponds to shift k - (m.shape - 1)
    return ncc, ov


def method_a2(moving_px, fixed, R=SEARCH_RADIUS_UM, px=4.0, sigma_um=8.0, chiralities=("proper", "reflect")):
    t0 = time.perf_counter()
    scales = scale_candidates(fixed, moving_px)
    lo = fixed.min(0) - 40; hi = fixed.max(0) + 40
    shape = (int(np.ceil((hi[1] - lo[1]) / px)), int(np.ceil((hi[0] - lo[0]) / px)))
    f = _raster(fixed, lo, shape, px, sigma_um); mf = _hull_mask(fixed, lo, shape, px)
    f = f / max(f[mf > 0].mean(), 1e-12)
    rmax = int(R / px)
    n = 0

    def score(chir, th, s, npeaks):
        q = _transform_hyp(moving_px, chir, th, s)
        tc = fixed.mean(0) - q.mean(0)
        qq = q + tc
        m = _raster(qq, lo, shape, px, sigma_um); mm = _hull_mask(qq, lo, shape, px)
        if mm.sum() < 10:
            return []
        m = m / max(m[mm > 0].mean(), 1e-12)
        ncc, ov = masked_ncc(f, mf, m, mm, min_overlap=0.3 * min(mf.sum(), mm.sum()))
        cy, cx = shape[0] - 1, shape[1] - 1
        win = ncc[cy - rmax: cy + rmax + 1, cx - rmax: cx + rmax + 1]
        loc = (win == maximum_filter(win, size=5)) & np.isfinite(win)
        cand = np.argwhere(loc); vals = win[loc]
        finite = win[np.isfinite(win)]
        med, mad = np.median(finite), np.median(np.abs(finite - np.median(finite))) + 1e-9
        out = []
        for o in np.argsort(vals)[::-1][:npeaks]:
            dy, dx = (cand[o] - rmax) * px
            out.append((float(vals[o]), tc + np.array([dx, dy]), float((vals[o] - med) / (1.4826 * mad))))
        return out

    coarse = []
    for chir in chiralities:
        for th in np.arange(0, 360, COARSE_ROT_STEP):
            for s in scales:
                for v, t, z in score(chir, th, s, 2):
                    coarse.append(Hypothesis(chir, float(th), s, t, v, "A2 coarse", {"robust_z": z}))
                n += 1
    coarse.sort(key=lambda h: -h.score)
    fine, seen = [], set()
    for h in coarse[:8]:
        key = (h.chirality, round(h.theta), round(h.scale, 3))
        if key in seen:
            continue
        seen.add(key)
        for th in np.arange(h.theta - FINE_ROT_HALF, h.theta + FINE_ROT_HALF + 1e-9, FINE_ROT_STEP):
            for fct in FINE_SCALE:
                for v, t, z in score(h.chirality, th, h.scale * fct, 2):
                    fine.append(Hypothesis(h.chirality, float(th), h.scale * fct, t, v, "A2 fine", {"robust_z": z}))
                n += 1
    fine.sort(key=lambda h: -h.score)
    top = fine[0]
    second = next((h for h in fine if np.linalg.norm(h.translation - top.translation) > 15 or h.chirality != top.chirality), None)
    diag = {"peak_ncc": top.score, "peak_robust_z": top.extra["robust_z"],
            "second_peak_ncc": second.score if second else np.nan,
            "coarse_top": [(h.chirality, h.theta, h.scale, h.translation.tolist(), h.score) for h in coarse[:10]]}
    return _finish("A2 density-NCC+ICP", fine, moving_px, fixed, t0, n, diag=diag)


# ----------------------------------------------------------------------------- A3
def _short_icp_error(src, fixed, tree, init, iters):
    tr = init
    err = np.inf
    for _ in range(iters):
        moved = tr.apply(src)
        d, j = tree.query(moved, k=1)
        keep = d <= np.quantile(d, 0.8)
        from src.pointset_registration import umeyama_similarity
        tr = umeyama_similarity(src[keep], fixed[j[keep]])
        err = float(np.mean(d[keep]))
    return err, tr


def method_a3(moving_px, fixed, R=SEARCH_RADIUS_UM, chiralities=("proper", "reflect")):
    t0 = time.perf_counter()
    tree = cKDTree(fixed)
    scales = scale_candidates(fixed, moving_px)
    sub = moving_px[:: max(1, len(moving_px) // 600)]          # scoring subsample (refine uses all points)
    offs = np.arange(-R, R + 1e-9, A3_TRANS_STEP)
    starts = []
    n = 0
    for chir in chiralities:
        C = CHIRALITY[chir]
        src = sub @ C.T
        for th in np.arange(0, 360, A3_ROT_STEP):
            for s in scales:
                q = s * src @ rot(th).T
                tc = fixed.mean(0) - q.mean(0)
                for dx in offs:
                    for dy in offs:
                        init = SimilarityTransform(scale=s, rotation_rad=float(np.deg2rad(th)), translation=tc + np.array([dx, dy]))
                        err, tr = _short_icp_error(src, fixed, tree, init, A3_SHORT_ITERS)
                        starts.append(Hypothesis(chir, float(np.rad2deg(tr.rotation_rad)), float(tr.scale), tr.translation, -err,
                                                 "A3 multistart", {"start": (th, s, dx, dy)}))
                        n += 1
    starts.sort(key=lambda h: -h.score)
    # keep distinct starts (after short ICP) before full refine
    kept = []
    for h in starts:
        if all(abs(h.theta - k.theta) > 2 or np.linalg.norm(h.translation - k.translation) > 10 or h.chirality != k.chirality for k in kept):
            kept.append(h)
        if len(kept) >= A3_TOPK:
            break
    diag = {"best_short_error": -kept[0].score, "second_short_error": -kept[1].score if len(kept) > 1 else np.nan,
            "short_error_quantiles": np.quantile([-h.score for h in starts], [0.0, 0.01, 0.1, 0.5]).tolist()}
    return _finish("A3 multistart+ICP", kept, moving_px, fixed, t0, n, n_refine=A3_TOPK, diag=diag)


METHODS = {"A0": method_a0, "A1": method_a1, "A2": method_a2, "A3": method_a3}


def run_method(key, moving_px, fixed, width, height, restrict_chirality=None):
    """Run one method with runtime and peak traced memory. restrict_chirality: None|'proper'|'reflect'."""
    tracemalloc.start()
    t0 = time.perf_counter()
    if key == "A0":
        flips = None
        if restrict_chirality == "proper":
            flips = ((False, False), (True, True))
        elif restrict_chirality == "reflect":
            flips = ((True, False), (False, True))
        res = method_a0(moving_px, fixed, width, height, flip_candidates=flips)
    else:
        chir = ("proper", "reflect") if restrict_chirality is None else (restrict_chirality,)
        res = METHODS[key](moving_px, fixed, chiralities=chir)
    res.runtime_s = time.perf_counter() - t0
    res.peak_mem_mb = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()
    return res


# ----------------------------------------------------------------------------- evaluation
def transform_errors(M, t, Mt, tt, width, height):
    gx, gy = np.meshgrid(np.linspace(0, width, 21), np.linspace(0, height, 21))
    g = np.column_stack([gx.ravel(), gy.ravel()])
    e = np.linalg.norm(apply_eff(M, t, g) - apply_eff(Mt, tt, g), axis=1)
    rel = M @ np.linalg.inv(Mt)
    u, s, vt = np.linalg.svd(rel)
    Rp = u @ vt
    c = np.array([width / 2, height / 2])
    return {
        "transform_rms_error_um": float(np.sqrt(np.mean(e**2))), "transform_max_error_um": float(e.max()),
        "translation_error_center_um": float(np.linalg.norm(apply_eff(M, t, c[None]) - apply_eff(Mt, tt, c[None]))),
        "rotation_error_deg": float(np.degrees(np.arctan2(Rp[1, 0], Rp[0, 0]))) if np.linalg.det(Rp) > 0 else 180.0,
        "chirality_correct": bool(np.sign(np.linalg.det(M)) == np.sign(np.linalg.det(Mt))),
        "scale_error_pct": float(100 * (np.sqrt(abs(np.linalg.det(M)) / abs(np.linalg.det(Mt))) - 1)),
    }


def overlap_moving(world_moving, fixed, erode_um=15.0):
    h = ConvexHull(fixed)
    path = MPath(fixed[h.vertices])
    inside = path.contains_points(world_moving)
    # erode by checking distance to hull edges
    verts = fixed[h.vertices]
    edges = np.stack([verts, np.roll(verts, -1, 0)], 1)
    a, b = edges[:, 0][None], edges[:, 1][None]
    p = world_moving[:, None]
    ab = b - a
    tt = np.clip(np.sum((p - a) * ab, -1) / np.sum(ab * ab, -1), 0, 1)
    dist = np.min(np.linalg.norm(p - (a + tt[..., None] * ab), axis=-1), axis=1)
    return inside & (dist >= erode_um)


def mirror_nulls(points):
    c = points.mean(0)
    mx = points.copy(); mx[:, 0] = 2 * c[0] - points[:, 0]
    return {"mirror_x": mx, "rotate_180": 2 * c - points}


def inlier_rate(world_moving, fixed, r=3.0):
    if len(world_moving) == 0:
        return np.nan
    return float(np.mean(cKDTree(fixed).query(world_moving)[0] <= r))


def block_consistency(world_moving, fixed, block=200.0, search=30.0):
    """Fraction of 200-um blocks whose local displacement-histogram peak is sharp (ratio>=5) and at ~0."""
    lo = world_moving.min(0)
    tree = cKDTree(fixed)
    rho = len(fixed) / ConvexHull(fixed).volume
    ok, total = 0, 0
    for bx in np.arange(lo[0], world_moving[:, 0].max(), block):
        for by in np.arange(lo[1], world_moving[:, 1].max(), block):
            sel = (world_moving[:, 0] >= bx) & (world_moving[:, 0] < bx + block) & (world_moving[:, 1] >= by) & (world_moving[:, 1] < by + block)
            if sel.sum() < 25:
                continue
            total += 1
            pts = world_moving[sel]
            lists = tree.query_ball_point(pts, search + 3)
            disp = np.concatenate([fixed[p] - pts[i] for i, p in enumerate(lists) if p]) if any(lists) else np.zeros((0, 2))
            if not len(disp):
                continue
            g = np.arange(-search, search + 0.1, 1.0)
            G = np.array([(x, y) for x in g for y in g])
            cnt = np.array([len(v) for v in cKDTree(disp).query_ball_point(G, 3.0)])
            k = int(np.argmax(cnt))
            ratio = cnt[k] / (sel.sum() * rho * np.pi * 9)
            if ratio >= 5 and np.linalg.norm(G[k]) <= 6:
                ok += 1
    return ok / total if total else np.nan, total


def detector(M, t, moving_px, fixed, *, basin_margin=np.nan, heldout_inlier=np.nan):
    """Pre-specified PASS / AMBIGUOUS / FAIL decision for a final affine (no truth used)."""
    w = apply_eff(M, t, moving_px)
    ov = overlap_moving(w, fixed)
    wo = w[ov]
    inl = inlier_rate(wo, fixed, DETECT["inlier_radius_um"])
    nulls = [inlier_rate(v[overlap_moving(v, fixed)], fixed) for v in mirror_nulls(wo).values()] if len(wo) > 10 else [np.nan]
    null = float(np.nanmean(nulls)) if np.any(np.isfinite(nulls)) else np.nan
    cons, nblocks = block_consistency(wo, fixed) if len(wo) > 25 else (np.nan, 0)
    ratio = inl / null if null and np.isfinite(null) and null > 0 else np.nan
    d = DETECT
    if (np.isfinite(inl) and inl >= d["pass_min_inlier"] and ratio >= d["pass_min_inlier_over_null"]
            and cons >= d["pass_min_block_consistency"]
            and (not np.isfinite(heldout_inlier) or heldout_inlier >= d["pass_min_heldout_inlier"])
            and (not np.isfinite(basin_margin) or basin_margin >= d["ambiguous_basin_margin"])):
        status = "PASS"
    elif np.isfinite(ratio) and ratio >= d["ambiguous_min_inlier_over_null"]:
        status = "AMBIGUOUS"
    else:
        status = "FAIL"
    return {"detector_status": status, "overlap_moving_n": int(ov.sum()), "overlap_moving_fraction": float(ov.mean()),
            "inlier3": inl, "inlier3_null": null, "inlier_over_null": ratio, "block_consistency": cons,
            "blocks": nblocks, "basin_margin": basin_margin, "heldout_inlier3": heldout_inlier}


def checker_heldout(key, moving_px, fixed, width, height, block_px=100.0):
    """Spatial held-out over the full field: fit on one checkerboard colour (100-px blocks), evaluate the other.

    Field extent/centroid are unchanged, so this isolates generalisation to unseen nuclei from crop effects.
    Fixed points are shared targets (no fixed-side split), as in the prior audit.
    """
    colour = (np.floor(moving_px[:, 0] / block_px) + np.floor(moving_px[:, 1] / block_px)) % 2 == 0
    out = []
    for c in (True, False):
        tr = colour == c
        res = run_method(key, moving_px[tr], fixed, width, height)
        te = apply_eff(res.M, res.t, moving_px[~tr])
        te = te[overlap_moving(te, fixed)]
        out.append({"train": f"checker_{int(c)}", "heldout_inlier3": inlier_rate(te, fixed),
                    "heldout_null": float(np.mean([inlier_rate(v, fixed) for v in mirror_nulls(te).values()])),
                    "heldout_n": int(len(te)), "M": res.M.tolist(), "t": res.t.tolist(), "objective": res.objective})
    return out


def split_heldout(key, moving_px, fixed, width, height, axis):
    """Fit on one half of the moving image (raw px), evaluate inlier rate on the other half (all fixed as targets)."""
    out = []
    mid = (width if axis == 0 else height) / 2
    for train_low in (True, False):
        tr = moving_px[:, axis] < mid if train_low else moving_px[:, axis] >= mid
        res = run_method(key, moving_px[tr], fixed, width, height)
        te = apply_eff(res.M, res.t, moving_px[~tr])
        te = te[overlap_moving(te, fixed)]
        nulls = [inlier_rate(v, fixed) for v in mirror_nulls(te).values()]
        out.append({"train": f"{'xy'[axis]}{'<' if train_low else '>='}{mid:g}px", "heldout_inlier3": inlier_rate(te, fixed),
                    "heldout_null": float(np.mean(nulls)), "heldout_n": int(len(te)),
                    "M": res.M.tolist(), "t": res.t.tolist(), "objective": res.objective})
    return out


# ----------------------------------------------------------------------------- synthetic generator
@dataclass
class Pair:
    name: str
    fixed: np.ndarray
    moving_px: np.ndarray
    width: float
    height: float
    M_true: np.ndarray | None
    t_true: np.ndarray | None
    true_fixed_index: np.ndarray        # per moving point, -1 if no shared nucleus
    moving_world_true: np.ndarray | None
    params: dict


def _intensity(rng, L, nonuniform, gaps):
    n = 256
    ax = np.linspace(-L / 2, L / 2, n)
    X, Y = np.meshgrid(ax, ax)
    I = np.ones_like(X)
    for _ in range(10):
        c = rng.uniform(-L / 2, L / 2, 2); s = rng.uniform(60, 220); a = rng.uniform(-0.6, 1.2)
        I += nonuniform * a * np.exp(-0.5 * ((X - c[0]) ** 2 + (Y - c[1]) ** 2) / s**2)
    I = np.clip(I, 0.15, None)
    for _ in range(gaps):
        c = rng.uniform(-L / 2, L / 2, 2); ang = rng.uniform(0, np.pi); ln = rng.uniform(150, 450); wd = rng.uniform(12, 45)
        u = np.array([np.cos(ang), np.sin(ang)])
        rel = np.stack([X - c[0], Y - c[1]], -1)
        along = rel @ u; across = rel @ np.array([-u[1], u[0]])
        I[(np.abs(along) < ln / 2) & (np.abs(across) < wd / 2)] = 0.0
    return ax, I


def _sample(rng, ax, I, rho_mean, hard_core=4.5):
    from scipy.interpolate import RegularGridInterpolator
    L = ax[-1] - ax[0]
    lam = I / I.mean() * rho_mean
    n = rng.poisson(lam.max() * L * L)
    p = rng.uniform(ax[0], ax[-1], (n, 2))
    f = RegularGridInterpolator((ax, ax), lam.T)(p)
    p = p[rng.random(n) < f / lam.max()]
    # greedy hard-core thinning (nuclei do not overlap)
    tree = cKDTree(p)
    keep = np.ones(len(p), bool)
    for i, j in tree.query_pairs(hard_core):
        if keep[i] and keep[j]:
            keep[j] = False
    return p[keep]


def make_pair(seed, *, offset_um=50.0, fov_rot_deg=0.6, reflect=True, scale=1.1, mov_w_px=901, mov_h_px=901,
              fixed_fov=(1010.0, 1010.0), q_shared=0.6, det_fixed=0.95, det_moving=0.9, nonuniform=1.0, gaps=6,
              deform_amp=3.0, noise=0.8, density=0.0019, name="synthetic", unrelated_moving=False):
    rng = np.random.default_rng(seed)
    L = 2400.0
    ax, I = _intensity(rng, L, nonuniform, gaps)
    per_section = (q_shared + (1 - q_shared) / 2)
    latent = _sample(rng, ax, I, density / (per_section * 0.5 * (det_fixed + det_moving)))
    latent += np.array([fixed_fov[0] / 2, fixed_fov[1] / 2])
    u = rng.random(len(latent))
    in_a = u < q_shared + (1 - q_shared) / 2
    in_b = (u < q_shared) | (u >= q_shared + (1 - q_shared) / 2)
    det_a = rng.random(len(latent)) < det_fixed
    det_b = rng.random(len(latent)) < det_moving
    fov = (latent[:, 0] >= 0) & (latent[:, 0] < fixed_fov[0]) & (latent[:, 1] >= 0) & (latent[:, 1] < fixed_fov[1])
    fsel = in_a & det_a & fov
    fixed = latent[fsel]
    fixed_index = -np.ones(len(latent), int); fixed_index[fsel] = np.arange(fsel.sum())
    # section B geometry: smooth deformation + centroid shift (different section height)
    if unrelated_moving:
        ax2, I2 = _intensity(np.random.default_rng(seed + 9999), L, nonuniform, gaps)
        latent_b = _sample(np.random.default_rng(seed + 9998), ax2, I2, density / 0.9) + np.array([fixed_fov[0] / 2, fixed_fov[1] / 2])
        in_b2 = np.ones(len(latent_b), bool); det_b2 = rng.random(len(latent_b)) < 0.9
        world_b = latent_b; bmask = in_b2 & det_b2; b_fixed = -np.ones(len(latent_b), int)
    else:
        ph = rng.uniform(0, 2 * np.pi, 4)
        dxy = deform_amp * np.column_stack([np.sin(2 * np.pi * latent[:, 1] / 700 + ph[0]) * np.cos(2 * np.pi * latent[:, 0] / 900 + ph[1]),
                                            np.sin(2 * np.pi * latent[:, 0] / 650 + ph[2]) * np.cos(2 * np.pi * latent[:, 1] / 850 + ph[3])])
        world_b = latent + dxy + rng.normal(0, noise, latent.shape)
        bmask = in_b & det_b; b_fixed = fixed_index
    C = np.diag([1.0, -1.0]) if reflect else np.eye(2)
    M = scale * rot(fov_rot_deg) @ C
    ang = rng.uniform(0, 2 * np.pi)
    center_world = np.array([fixed_fov[0] / 2, fixed_fov[1] / 2]) + offset_um * np.array([np.cos(ang), np.sin(ang)])
    t = center_world - M @ np.array([mov_w_px / 2, mov_h_px / 2])
    px = np.linalg.solve(M, (world_b - t).T).T
    crop = bmask & (px[:, 0] >= 0) & (px[:, 0] < mov_w_px) & (px[:, 1] >= 0) & (px[:, 1] < mov_h_px)
    return Pair(name, fixed, px[crop], float(mov_w_px), float(mov_h_px), M, t, b_fixed[crop], world_b[crop],
                {"seed": seed, "offset_um": offset_um, "offset_angle_deg": float(np.degrees(ang)), "fov_rot_deg": fov_rot_deg,
                 "reflect": reflect, "scale": scale, "mov_w_px": mov_w_px, "mov_h_px": mov_h_px, "q_shared": q_shared,
                 "det_fixed": det_fixed, "det_moving": det_moving, "nonuniform": nonuniform, "gaps": gaps,
                 "deform_amp": deform_amp, "noise": noise, "n_fixed": int(len(fixed)), "n_moving": int(crop.sum()),
                 "shared_fraction_of_moving": float(np.mean(b_fixed[crop] >= 0)), "unrelated_moving": unrelated_moving})


SYNTHETIC_CONDITIONS = {
    "offset000": dict(offset_um=0.0),
    "offset025": dict(offset_um=25.0),
    "offset050": dict(offset_um=50.0),
    "offset075": dict(offset_um=75.0),
    "offset100": dict(offset_um=100.0),
    "offset150_large": dict(offset_um=150.0),
    "rot3_offset50": dict(offset_um=50.0, fov_rot_deg=3.0),
    "rot6_offset50": dict(offset_um=50.0, fov_rot_deg=6.0),
    "rot90_proper_offset50": dict(offset_um=50.0, fov_rot_deg=90.0, reflect=False),
    "asym_crop_offset50": dict(offset_um=50.0, mov_w_px=901, mov_h_px=640),
    "small_fov_offset80": dict(offset_um=80.0, mov_w_px=640, mov_h_px=640),
    "partial_overlap_350": dict(offset_um=350.0),
    "missing40_offset50": dict(offset_um=50.0, det_moving=0.55),
    "strong_nonuniform_offset50": dict(offset_um=50.0, nonuniform=2.0, gaps=14),
    "low_shared_offset50": dict(offset_um=50.0, q_shared=0.3),
}
SYNTHETIC_SEEDS = (11, 12, 13)


def synthetic_cases():
    return [(c, s) for c in SYNTHETIC_CONDITIONS for s in SYNTHETIC_SEEDS]


# ----------------------------------------------------------------------------- Real
def load_real():
    cfg = json.loads((REAL_RUN / "config.json").read_text())["affine"]
    I = REAL_RUN / "input"
    return {
        "fixed_all": np.load(I / "original_fixed_points.npy"),
        "moving_px": np.load(I / "original_moving_points.npy"),
        "moving_saved_all": np.load(I / "moving_points.npy"),
        "fixed_subset": np.load(I / "success_metric_fixed_points.npy"),
        "moving_subset": np.load(I / "success_metric_moving_points.npy"),
        "width": float(cfg["image_width"]), "height": float(cfg["image_height"]),
        "saved_post_flip": (np.asarray(cfg["matrix"]), np.asarray(cfg["translation"]), cfg["flip_x"], cfg["flip_y"]),
        "he_image": np.load(I / "he_image.npy"), "tissue_mask": np.load(I / "tissue_mask.npy"),
        "warp_meta": json.loads((I / "metadata.json").read_text()),
    }


def saved_effective(real):
    M, t, fx, fy = real["saved_post_flip"]
    F = np.diag([-1.0 if fx else 1.0, -1.0 if fy else 1.0])
    return M @ F, t + M @ np.array([real["width"] if fx else 0.0, real["height"] if fy else 0.0])


def subset_rule(M, t, moving_px, fixed, width, height, edge_um=10.0, near_he_um=43.0):
    """Production-equivalent subset rule with the HE footprint as an exact polygon (no raster)."""
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], float)
    poly = apply_eff(M, t, corners)
    path = MPath(poly)

    def edge_dist(p):
        a = poly[None]; b = np.roll(poly, -1, 0)[None]; q = p[:, None]
        ab = b - a
        tt = np.clip(np.sum((q - a) * ab, -1) / np.sum(ab * ab, -1), 0, 1)
        return np.min(np.linalg.norm(q - (a + tt[..., None] * ab), axis=-1), axis=1)

    w = apply_eff(M, t, moving_px)
    mv = path.contains_points(w) & (edge_dist(w) >= edge_um)
    near = cKDTree(w).query(fixed)[0] <= near_he_um
    fv = path.contains_points(fixed) & (edge_dist(fixed) >= edge_um) & near
    return mv, fv


def he_white_fraction(real, M, t):
    """Fraction of fixed nuclei that land in HE white (no-tissue) pixels under transform (M, t)."""
    from scipy.ndimage import binary_opening
    he = real["he_image"].astype(float); mask = real["tissue_mask"]; meta = real["warp_meta"]
    mx, mn = he.max(2), he.min(2)
    white = binary_opening((he.mean(2) > 190) & ((mx - mn) / np.maximum(mx, 1) < 0.25) & mask, iterations=2)
    Ms, ts = saved_effective(real)
    f_saved = apply_eff(Ms, ts, np.linalg.solve(M, (real["fixed_all"] - t).T).T)
    c = np.rint(f_saved[:, 0] - meta["col0_world_x"]).astype(int); r = np.rint(meta["row0_world_y"] - f_saved[:, 1]).astype(int)
    ok = (r >= 0) & (r < mask.shape[0]) & (c >= 0) & (c < mask.shape[1])
    return float(white[r[ok], c[ok]].mean()), float(white[mask].mean()), int(ok.sum())


def json_safe(v):
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, np.ndarray):
        return json_safe(v.tolist())
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else float(v)
    return v


def save_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False) + "\n")
