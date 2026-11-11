"""Stage-comparison metrics (analysis only). Geometric agreement only: none of these is a cell-identity accuracy."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MPath
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter, map_coordinates
from scipy.spatial import ConvexHull, cKDTree

ROOT = Path("/mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu")
sys.path.insert(0, str(ROOT / "analysis/affine-global-initialization-20261014"))
sys.path.insert(0, str(ROOT / "analysis/real-affine-candidate-audit-claude-20261014"))
import init_lib as L  # noqa: E402  (frozen)
import audit_core as ac  # noqa: E402  (frozen audit helpers: correlogram peak, regional NCC, HE white mask)

THRESH = (1.0, 2.0, 3.0, 5.0, 10.0)


# ----------------------------------------------------------------------------- transforms
def apply(M, t, p):
    return np.asarray(p, float) @ np.asarray(M, float).T + np.asarray(t, float)


def decompose(M, t, width=None, height=None) -> dict:
    M = np.asarray(M, float)
    d = ac.decompose_affine(M)
    s = d["singular_values"]
    out = {"m00": M[0, 0], "m01": M[0, 1], "m10": M[1, 0], "m11": M[1, 1], "tx": float(t[0]), "ty": float(t[1]),
           "determinant": d["determinant"], "scale_sqrt_absdet": float(np.sqrt(abs(d["determinant"]))),
           "sv_max": s[0], "sv_min": s[1], "anisotropy_ratio": d["anisotropy_ratio"],
           "polar_rotation_deg": d["polar_rotation_deg"], "reflection": d["polar_factor_is_reflection"],
           "shear_offdiag": d["shear_offdiag"]}
    # rotation of the reflection-free part (M C with C = diag(1,-1) when reflected)
    C = np.diag([1.0, -1.0]) if d["determinant"] < 0 else np.eye(2)
    A = M @ C
    out["rotation_deg_after_chirality"] = float(np.degrees(np.arctan2(A[1, 0] - A[0, 1], A[0, 0] + A[1, 1])))
    if width is not None:
        c = apply(M, t, np.array([[width / 2, height / 2]]))[0]
        out["image_center_world_x"], out["image_center_world_y"] = float(c[0]), float(c[1])
    return out


def displacement_between(Ma, ta, Mb, tb, pts) -> dict:
    """|A_b(p) - A_a(p)| over given points: a displacement (how far the transform moves points), not an error."""
    e = np.linalg.norm(apply(Mb, tb, pts) - apply(Ma, ta, pts), axis=1)
    v = apply(Mb, tb, pts) - apply(Ma, ta, pts)
    return {"disp_median_um": float(np.median(e)), "disp_p95_um": float(np.quantile(e, 0.95)), "disp_max_um": float(e.max()),
            "disp_mean_dx_um": float(v[:, 0].mean()), "disp_mean_dy_um": float(v[:, 1].mean())}


# ----------------------------------------------------------------------------- regions
def eroded_hull_contains(hull_pts, q, erode_um):
    h = ConvexHull(hull_pts)
    verts = hull_pts[h.vertices]
    inside = MPath(verts).contains_points(q)
    return inside & (poly_edge_distance(verts, q) >= erode_um)


def poly_edge_distance(poly, q):
    a = poly[None]; b = np.roll(poly, -1, 0)[None]; p = np.asarray(q)[:, None]
    ab = b - a
    tt = np.clip(np.sum((p - a) * ab, -1) / np.sum(ab * ab, -1), 0, 1)
    return np.min(np.linalg.norm(p - (a + tt[..., None] * ab), axis=-1), axis=1)


def he_footprint_world(M, t, width, height):
    return apply(M, t, np.array([[0, 0], [width, 0], [width, height], [0, height]], float))


def in_footprint(M, t, width, height, q, erode_um):
    poly = he_footprint_world(M, t, width, height)
    return MPath(poly).contains_points(q) & (poly_edge_distance(poly, q) >= erode_um)


# ----------------------------------------------------------------------------- point-cloud agreement
def nn_block(mov_w, fixed, prefix="") -> dict:
    tf, tm = cKDTree(fixed), cKDTree(mov_w)
    d_mf, i_mf = tf.query(mov_w)
    d_fm, i_fm = tm.query(fixed)
    sym = np.concatenate([d_mf, d_fm])
    mutual = i_fm[i_mf] == np.arange(len(mov_w))
    out = {"n_moving": len(mov_w), "n_fixed": len(fixed),
           "sym_nn_median_um": np.median(sym), "sym_nn_p90_um": np.quantile(sym, 0.9), "sym_nn_p95_um": np.quantile(sym, 0.95),
           "m2f_nn_median_um": np.median(d_mf), "m2f_nn_p90_um": np.quantile(d_mf, 0.9), "m2f_nn_p95_um": np.quantile(d_mf, 0.95),
           "f2m_nn_median_um": np.median(d_fm), "f2m_nn_p90_um": np.quantile(d_fm, 0.9), "f2m_nn_p95_um": np.quantile(d_fm, 0.95),
           "mutual_nn_rate": mutual.mean(), "mutual_nn_within3_rate": (mutual & (d_mf <= 3)).mean(),
           "icp_trimmed_objective_q07": float(np.mean(d_mf[d_mf <= np.quantile(d_mf, 0.7)]))}
    for r in THRESH:
        out[f"m2f_within_{r:g}um"] = float(np.mean(d_mf <= r))
        out[f"f2m_within_{r:g}um"] = float(np.mean(d_fm <= r))
    return {prefix + k: float(v) for k, v in out.items()}


def null_within3(mov_w, fixed) -> dict:
    tf = cKDTree(fixed)
    return {f"null_{k}_within3": float(np.mean(tf.query(v)[0] <= 3)) for k, v in ac.null_transforms(mov_w).items()}


def density_correlation(mov_w, fixed, mask_fn, sigmas=(5.0, 10.0, 20.0), px=2.0) -> dict:
    """Pearson correlation of Gaussian density maps inside mask_fn(grid_points) (same mask for every stage)."""
    lo = np.minimum(mov_w.min(0), fixed.min(0)) - 40; hi = np.maximum(mov_w.max(0), fixed.max(0)) + 40
    shape = (int(np.ceil((hi[1] - lo[1]) / px)), int(np.ceil((hi[0] - lo[0]) / px)))
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    g = np.column_stack([xx.ravel() * px + lo[0], yy.ravel() * px + lo[1]])
    m = mask_fn(g).reshape(shape)
    out = {}
    for s in sigmas:
        a = L._raster(mov_w, lo, shape, px, s)[m]; b = L._raster(fixed, lo, shape, px, s)[m]
        out[f"density_corr_sigma{s:g}"] = float(np.corrcoef(a, b)[0, 1])
    return out


def block_correlogram(mov_w, fixed, *, block, search, kernel, step=1.0, min_n=25, origin=(0.0, 0.0), extent=1000.0):
    """Local displacement-histogram peak per block of moving points (prior audit method, via frozen audit helper)."""
    density = len(fixed) / ConvexHull(fixed).volume
    rows = []
    for x0 in np.arange(origin[0], origin[0] + extent, block):
        for y0 in np.arange(origin[1], origin[1] + extent, block):
            sel = (mov_w[:, 0] >= x0) & (mov_w[:, 0] < x0 + block) & (mov_w[:, 1] >= y0) & (mov_w[:, 1] < y0 + block)
            if sel.sum() < min_n:
                continue
            r = ac.displacement_peak(fixed, mov_w[sel], search_um=search, kernel_um=kernel, step_um=step, density=density)
            r.update(center_x=x0 + block / 2, center_y=y0 + block / 2, block_um=block)
            r["residual_mag_um"] = float(np.hypot(r["refined_dx"], r["refined_dy"]))
            rows.append(r)
    return rows


def correlogram_summary(rows, ratio_min=5.0, prefix="") -> dict:
    ok = [r for r in rows if r["peak_ratio"] >= ratio_min]
    mags = np.array([r["residual_mag_um"] for r in ok])
    return {prefix + "blocks": len(rows), prefix + "blocks_reliable": len(ok),
            prefix + "peak_ratio_median": float(np.median([r["peak_ratio"] for r in rows])) if rows else np.nan,
            prefix + "local_residual_median_um": float(np.median(mags)) if len(mags) else np.nan,
            prefix + "local_residual_p90_um": float(np.quantile(mags, 0.9)) if len(mags) else np.nan,
            prefix + "blocks_peak_within3um": float(np.mean(mags <= 3)) if len(mags) else np.nan}


def regional_ncc_summary(fixed_sub, moving_sub, radius=18.0) -> dict:
    r = ac.regional_ncc(fixed_sub, moving_sub, radius)
    bnd = r.boundary_x | r.boundary_y
    return {"regional_ncc_best_median": float(np.median(r.best_ncc)), "regional_ncc_margin_median": float(np.median(r.margin)),
            "regional_ncc_anchor_boundary_rate": float(bnd.mean()),
            "regional_ncc_offset_mag_median_um": float(np.median(np.hypot(*r.best_offsets.T))),
            "regional_ncc_nucleus_boundary_rate": float(bnd[r.cell_anchor_index].mean()), "regional_ncc_anchors": int(len(r.anchors))}


# ----------------------------------------------------------------------------- HE image (independent of moving points)
class HE:
    """HE image warped by the SAVED affine (1 um/px). A stage (M, t) is evaluated by pulling fixed nuclei to HE pixels
    with its inverse and pushing them through the saved warp: pixel = warp_saved(A_stage^{-1}(fixed))."""

    def __init__(self, real):
        from skimage.color import rgb2hed
        self.meta = real["warp_meta"]
        self.footprint = real["tissue_mask"]                 # = HE image footprint (not tissue; prior audit)
        self.white = ac.he_white_mask(real["he_image"], self.footprint)
        self.Ms, self.ts = L.saved_effective(real)
        hed = rgb2hed(real["he_image"])
        h = gaussian_filter(hed[..., 0], 2.0)
        fp = binary_erosion(self.footprint, iterations=12)
        self.hema = (h - h[fp].mean()) / h[fp].std()
        self.fp_inner = fp
        # signed distance to the white region (negative inside white), in um (1 um/px)
        self.sdist = distance_transform_edt(~self.white) - distance_transform_edt(self.white)

    def pix(self, M, t, fixed):
        px = np.linalg.solve(np.asarray(M, float), (np.asarray(fixed) - np.asarray(t)).T).T
        return ac.world_to_warped_pixel(apply(self.Ms, self.ts, px), self.meta)

    def _sample(self, img, cr, order=1):
        return map_coordinates(img.astype(float), [cr[:, 1], cr[:, 0]], order=order, mode="constant", cval=np.nan)

    def metrics(self, M, t, fixed) -> dict:
        cr = self.pix(M, t, fixed)
        inside = self._sample(self.fp_inner, cr, order=0) > 0.5
        c = cr[inside]
        white = self._sample(self.white, c, order=0) > 0.5
        hv = self._sample(self.hema, c)
        sd = self._sample(self.sdist, c)
        area_white = float(self.white[self.fp_inner].mean())
        dens = lambda lo, hi: np.sum((sd >= lo) & (sd < hi)) / max(np.sum((self.sdist[self.fp_inner] >= lo) & (self.sdist[self.fp_inner] < hi)), 1)  # noqa: E731
        tissue = dens(5, 30)
        out = {"he_n_fixed_inside": int(inside.sum()), "he_white_fixed_fraction": float(white.mean()),
               "he_white_area_fraction": area_white, "he_white_ratio_to_area": float(white.mean() / area_white),
               "he_hema_mean_at_fixed_z": float(np.nanmean(hv)),
               "he_density_inside_white_deep_rel": float(dens(-1e9, -5) / tissue),     # >5 um inside white
               "he_density_white_edge_rel": float(dens(-5, 0) / tissue),              # 0-5 um inside the white edge
               "he_density_tissue_edge_rel": float(dens(0, 5) / tissue)}              # 0-5 um outside the white edge
        # fixed density map vs hematoxylin map (sigma 10 um) inside the inner footprint
        img = np.zeros(self.hema.shape)
        ij = np.rint(c).astype(int)
        np.add.at(img, (ij[:, 1], ij[:, 0]), 1.0)
        dmap = gaussian_filter(img, 10.0)
        hmap = gaussian_filter(self.hema, 8.0)          # total 10 um-ish
        out["he_hema_vs_fixed_density_corr"] = float(np.corrcoef(dmap[self.fp_inner], hmap[self.fp_inner])[0, 1])
        return out

    def translation_scan(self, M, t, fixed, half=30.0, step=1.0, region=None):
        """World-space offset delta (added to fixed nuclei in the HE-warped frame) minimising the white fraction and
        maximising mean hematoxylin at fixed nuclei. Independent of moving points."""
        cr = self.pix(M, t, fixed)
        if region is not None:
            cr = cr[region]
        offs = np.arange(-half, half + 1e-9, step)
        W = np.full((len(offs), len(offs)), np.nan); H = np.full_like(W, np.nan)
        for i, dy in enumerate(offs):
            for j, dx in enumerate(offs):
                c = cr + np.array([dx, -dy])                 # world +y = row -1
                ins = self._sample(self.fp_inner, c, order=0) > 0.5
                W[i, j] = np.mean(self._sample(self.white, c[ins], order=0) > 0.5)
                H[i, j] = np.nanmean(self._sample(self.hema, c[ins]))
        iw = np.unravel_index(np.nanargmin(W), W.shape); ih = np.unravel_index(np.nanargmax(H), H.shape)
        # min-white offset: centroid of near-minimal cells (plateau-robust)
        thr = np.nanmin(W) + 0.1 * (np.nanmedian(W) - np.nanmin(W))
        yy, xx = np.meshgrid(offs, offs, indexing="ij")
        plateau = W <= thr
        return {"white_min_dx": float(xx[iw]), "white_min_dy": float(yy[iw]), "white_min": float(W[iw]),
                "white_at_zero": float(W[len(offs) // 2, len(offs) // 2]),
                "white_plateau_centroid_dx": float(xx[plateau].mean()), "white_plateau_centroid_dy": float(yy[plateau].mean()),
                "hema_max_dx": float(xx[ih]), "hema_max_dy": float(yy[ih]), "hema_max": float(H[ih]),
                "hema_at_zero": float(H[len(offs) // 2, len(offs) // 2]),
                "hema_offset_mag_um": float(np.hypot(xx[ih], yy[ih])), "n": int(len(cr))}, (offs, W, H)
