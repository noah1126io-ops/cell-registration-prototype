"""Semi-synthetic partial-overlap cases with known transform, region and cell identity.

Fixed  = a real HE (features + StarDist nuclei), optionally with the true region removed ("absent").
Moving = a pseudo-fluorescence window (~1 mm) cut from that HE under a known similarity transform:
  * tissue map: HE tissue sampled into the window, then perturbed by a smooth random field (boundary
    shapes change by a few um, as between serial sections / modalities);
  * nuclear map: re-rendered from the *moving* nuclei (not from hematoxylin), random intensities + noise;
  * nuclei: a fraction q of the true HE nuclei (identity known) + 0.8 um jitter, plus (1-q) foreign nuclei
    drawn inside tissue (non-shared cells).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_fill_holes, binary_closing, distance_transform_edt, gaussian_filter, map_coordinates

from .features import FeatureMaps
from .pipeline import Side, footprint
from .search import pose_matrix


@dataclass
class SynCase:
    case_id: str
    source: str
    condition: str
    q_shared: float
    absent: bool
    A_true: np.ndarray          # fixed_world = A_true @ moving_world + t_true
    t_true: np.ndarray
    fixed: Side
    moving: Side
    shared_moving_idx: np.ndarray
    shared_fixed_idx: np.ndarray


def _sample(img, res, pts):
    return map_coordinates(img, [pts[:, 1] / res, pts[:, 0] / res], order=1, cval=0.0)


def make_case(case_id, he_side: Side, *, seed, q_shared=0.6, window_um=1000.0, res=2.0, absent=False, condition="", source=""):
    rng = np.random.default_rng(seed)
    fm = he_side.feats
    H, W = fm.shape
    ext = np.array(he_side.extent)
    theta = float(rng.uniform(0, 360)); s = float(rng.uniform(0.97, 1.03))
    A = pose_matrix("proper", theta, s)
    # window centre: inside the HE so the rotated window fits, with >= 50 % tissue
    half = window_um / 2 * s * np.sqrt(2)
    for _ in range(500):
        c = rng.uniform(half, ext - half)
        g = np.stack(np.meshgrid(np.linspace(-window_um / 2, window_um / 2, 25), np.linspace(-window_um / 2, window_um / 2, 25)), -1).reshape(-1, 2)
        q = g @ A.T + c
        if _sample(fm.maps["tissue_bin"].astype(np.float32), res, q).mean() > 0.5:
            break
    mc = np.array([window_um, window_um]) / 2
    t = c - A @ mc
    n = int(round(window_um / res))
    yy, xx = np.mgrid[0:n, 0:n]
    pm = np.c_[xx.ravel(), yy.ravel()] * res                       # moving world (grid)
    qf = pm @ A.T + t                                               # fixed world
    tis = _sample(fm.maps["tissue_bin"].astype(np.float32), res, qf).reshape(n, n)
    noise = gaussian_filter(rng.normal(size=(n, n)), 10.0 / res)
    noise /= noise.std() + 1e-9
    tb = (gaussian_filter(tis, 1.5) + 0.12 * noise) > 0.5
    env = binary_fill_holes(binary_closing(tb, iterations=int(round(30 / res)), border_value=0))
    ei = distance_transform_edt(np.pad(env, 1, mode="edge"))[1:-1, 1:-1]; eo = distance_transform_edt(np.pad(~env, 1, mode="edge"))[1:-1, 1:-1]
    edt = np.clip((ei - eo) * res, -60, 60) / 60
    # nuclei
    Ainv = np.linalg.inv(A)
    he_pts = he_side.points
    mov_all = (he_pts - t) @ Ainv.T
    inwin = np.all((mov_all >= 0) & (mov_all < window_um), 1)
    idx_in = np.where(inwin)[0]
    keep = rng.random(len(idx_in)) < q_shared
    shared_f = idx_in[keep]
    shared_m = mov_all[shared_f] + rng.normal(0, 0.8, (len(shared_f), 2))
    n_foreign = int(round(len(idx_in) * (1 - q_shared)))
    tis_idx = np.argwhere(tb)
    pick = tis_idx[rng.choice(len(tis_idx), n_foreign, replace=len(tis_idx) < n_foreign)] if len(tis_idx) else np.zeros((0, 2), int)
    foreign = pick[:, ::-1] * res + rng.uniform(0, res, (len(pick), 2))
    mov_pts = np.vstack([shared_m, foreign])
    perm = rng.permutation(len(mov_pts))
    mov_pts = mov_pts[perm]
    shared_moving_idx = np.where(perm < len(shared_m))[0]
    order = perm[shared_moving_idx]
    # nuclear map rendered from moving nuclei
    img = np.zeros((n, n), np.float32)
    ij = np.clip(np.rint(mov_pts / res).astype(int), 0, n - 1)
    np.add.at(img, (ij[:, 1], ij[:, 0]), rng.lognormal(0, 0.4, len(ij)))
    nuc = gaussian_filter(img, 2.5 / res) + 0.02 * np.abs(rng.normal(size=(n, n)))
    nuc = np.clip(nuc / np.percentile(nuc[tb] if tb.any() else nuc, 99.5), 0, 1)
    mov_feats = FeatureMaps(f"{case_id}:pseudoIF", "IF", res,
                            {"tissue": gaussian_filter(tb.astype(np.float32), 1.0), "nuclear": nuc.astype(np.float32), "edt": edt.astype(np.float32),
                             "tissue_bin": tb, "envelope": env}, np.ones((n, n), np.float32), {"usable": True})
    moving = Side("IF", mov_pts, (window_um, window_um), mov_feats)
    fixed = he_side
    if absent:   # remove the true region (maps, FOV and nuclei) from the fixed side
        from .metrics import inside_poly
        poly = footprint((window_um, window_um)) @ A.T + t
        gy, gx = np.mgrid[0:H, 0:W]
        cell = inside_poly(np.c_[gx.ravel(), gy.ravel()] * res, poly, -20.0).reshape(H, W)
        maps = {k: np.where(cell, 0, v).astype(v.dtype) for k, v in fm.maps.items()}
        fov = np.where(cell, 0, fm.fov).astype(np.float32)
        keep_f = ~inside_poly(he_pts, poly, -20.0)
        fixed = Side("HE", he_pts[keep_f], he_side.extent, FeatureMaps(fm.frame + ":absent", "HE", res, maps, fov, fm.quality))
    return SynCase(case_id, source, condition, q_shared, absent, A, t, fixed, moving, shared_moving_idx, shared_f[order - 0] if len(shared_f) else shared_f)


def truth_errors(case: SynCase, M, t):
    g = np.stack(np.meshgrid(np.linspace(0, case.moving.extent[0], 21), np.linspace(0, case.moving.extent[1], 21)), -1).reshape(-1, 2)
    d = np.linalg.norm(g @ np.asarray(M).T + t - (g @ case.A_true.T + case.t_true), axis=1)
    return {"median_um": float(np.median(d)), "rms_um": float(np.sqrt(np.mean(d ** 2))), "max_um": float(d.max())}
