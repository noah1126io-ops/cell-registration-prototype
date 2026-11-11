"""A2 (point-density masked NCC + production ICP refine) and its pre-specified detector.

Vendored VERBATIM (function bodies unchanged) from analysis/affine-global-initialization-20261014/init_lib.py
(2026-10-14) so that production code can import it without sys.path manipulation.  Only the module header,
imports and path constants differ.  Parity with the analysis copy is tested in tests/test_registration_core.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from matplotlib.path import Path as MPath
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import ConvexHull, cKDTree

from ..pointset_registration import SimilarityTransform, affine_icp, icp_similarity


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


