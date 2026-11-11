"""Geometric and image-consistency diagnostics. None of these measure true cell identity."""
from __future__ import annotations

import numpy as np
from matplotlib.path import Path as MPath
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree


def auc(pos, neg, n=200000, seed=0):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    a = rng.choice(pos, min(n, len(pos))); b = rng.choice(neg, min(n, len(neg)))
    r = np.argsort(np.argsort(np.r_[a, b])) + 1
    return float((r[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def edge_distance(points, poly):
    a = np.asarray(poly, float); b = np.roll(a, -1, 0)
    q = np.asarray(points, float)[:, None]
    ab = (b - a)[None]
    tt = np.clip(np.sum((q - a[None]) * ab, -1) / np.maximum(np.sum(ab * ab, -1), 1e-12), 0, 1)
    return np.min(np.linalg.norm(q - (a[None] + tt[..., None] * ab), axis=-1), axis=1)


def inside_poly(points, poly, inset_um=0.0):
    """Inside the polygon and at least inset_um from its edge (inset<0 = dilate)."""
    points = np.asarray(points, float)
    if len(points) == 0:
        return np.zeros(0, bool)
    ins = MPath(poly).contains_points(points)
    d = edge_distance(points, poly)
    return (ins & (d >= inset_um)) if inset_um >= 0 else (ins | (d <= -inset_um))


def proximity(a_w, b, b_mask=None, radii=(1, 2, 3, 5)):
    """For each b point (optionally masked) distance to nearest a_w point."""
    bb = b if b_mask is None else b[b_mask]
    if len(bb) == 0 or len(a_w) == 0:
        return {**{f"le_{r}um": np.nan for r in radii}, "nn_median_um": np.nan, "nn_p95_um": np.nan, "n_eval": 0}
    d = cKDTree(a_w).query(bb)[0]
    out = {f"le_{r}um": float(np.mean(d <= r)) for r in radii}
    out.update(nn_median_um=float(np.median(d)), nn_p95_um=float(np.percentile(d, 95)), n_eval=int(len(bb)))
    return out


def mirror_null(a_w, b, b_mask, axis=0):
    """Reflect the transformed set about the centroid of the evaluated b points (density kept, identity destroyed)."""
    bb = b[b_mask] if b_mask is not None else b
    c = bb.mean(0)
    m = a_w.copy(); m[:, axis] = 2 * c[axis] - m[:, axis]
    return proximity(m, b, b_mask)


def checker_colour(points, block_um=100.0):
    return ((np.floor(points[:, 0] / block_um) + np.floor(points[:, 1] / block_um)) % 2).astype(int)


def sample_map(img, res_um, pts_um, order=1):
    rc = np.asarray(pts_um, float)[:, ::-1] / res_um
    return map_coordinates(img, rc.T, order=order, cval=np.nan)


def warp_map_into(src_img, src_res, T, dst_shape, dst_res, order=1, cval=np.nan):
    """Resample a source-frame map into the destination grid with the INVERSE transform (dst world -> src world)."""
    Ti = T.inverse()
    yy, xx = np.mgrid[0:dst_shape[0], 0:dst_shape[1]]
    w = np.c_[xx.ravel(), yy.ravel()] * dst_res
    s = Ti.apply(w) / src_res
    out = map_coordinates(src_img, [s[:, 1], s[:, 0]], order=order, cval=cval)
    return out.reshape(dst_shape)


def masked_corr(a, b, mask):
    m = mask & np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50:
        return float("nan")
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum() + 1e-12))


def block_corrs(a, b, mask, block_px, min_frac=0.3):
    out = []
    H, W = a.shape
    for r in range(0, H - block_px + 1, block_px):
        for c in range(0, W - block_px + 1, block_px):
            m = mask[r:r + block_px, c:c + block_px]
            if m.mean() >= min_frac:
                out.append(masked_corr(a[r:r + block_px, c:c + block_px], b[r:r + block_px, c:c + block_px], m))
    return np.array([v for v in out if np.isfinite(v)])


def dice(a, b, mask):
    m = mask & np.isfinite(a) & np.isfinite(b)
    A, B = a[m] > 0.5, b[m] > 0.5
    return float(2 * (A & B).sum() / max(A.sum() + B.sum(), 1))


def boundary_chamfer(env_a, env_b, mask, res_um):
    """Median distance (um) from envelope edge pixels of a to the envelope edge of b, inside mask."""
    from scipy.ndimage import binary_erosion, distance_transform_edt
    ea = env_a & ~binary_erosion(env_a) & mask
    eb = env_b & ~binary_erosion(env_b)
    if ea.sum() < 20 or eb.sum() < 20:
        return float("nan"), int(ea.sum())
    d = distance_transform_edt(~eb) * res_um
    return float(np.median(d[ea])), int(ea.sum())
