"""Partial-overlap pose search: chirality x rotation x scale grid, translation by masked NCC (FFT).

A *layer* pairs one fixed-frame representation with one moving-frame representation:
  points  : A2 Gaussian nuclear density + point convex hull mask (needs no image; P0)
  image   : structural feature map + imaged field-of-view mask (needs images; P2)
  cross   : tissue map of the side that has an image vs nuclear density of the side that has none (P1)
Each layer gives an NCC map over all translations. Layers are never summed raw: every map is converted
to a robust z-score (median / MAD over valid shifts) and candidates are combined by an explicit rule.

Transform convention:  fixed_world = A @ moving_world + t,  A = s * R(theta) @ C(chirality).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter, maximum_filter
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MPath

import sys
from pathlib import Path
_A2 = Path(__file__).resolve().parents[2] / "affine-global-initialization-20261014"
if str(_A2) not in sys.path:
    sys.path.insert(0, str(_A2))
from init_lib import masked_ncc  # noqa: E402  (existing A2 masked NCC, imported unmodified)

CHIR = {"proper": np.eye(2), "reflect": np.diag([1.0, -1.0])}


def rot(theta_deg):
    a = np.deg2rad(theta_deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def pose_matrix(chir, theta, s):
    return s * rot(theta) @ CHIR[chir]


@dataclass
class Layer:
    name: str
    kind: str                      # points | image | cross
    fixed_map: np.ndarray | None = None      # on fixed grid (res_um), for image/cross-with-fixed-image
    fixed_mask: np.ndarray | None = None
    moving_map: np.ndarray | None = None
    moving_mask: np.ndarray | None = None
    fixed_points: np.ndarray | None = None   # world um, for points / cross-with-fixed-points
    moving_points: np.ndarray | None = None
    res_um: float = 4.0
    sigma_um: float = 8.0


@dataclass
class Candidate:
    chirality: str
    theta: float
    scale: float
    t: np.ndarray
    score: float                   # combined score under the run's rule
    layer_ncc: dict
    layer_z: dict
    stage: str = "coarse"

    def A(self):
        return pose_matrix(self.chirality, self.theta, self.scale)

    def as_dict(self):
        return {"chirality": self.chirality, "theta": self.theta, "scale": self.scale, "t": self.t.tolist(),
                "score": self.score, "layer_ncc": self.layer_ncc, "layer_z": self.layer_z, "stage": self.stage}


# ----------------------------------------------------------------------------- rendering on a common canvas
class Canvas:
    def __init__(self, fixed_extent_um, moving_extent_um, fixed_center, res_um, max_scale=1.1):
        L = max(np.hypot(*fixed_extent_um), np.hypot(*moving_extent_um) * max_scale) + 8 * res_um
        self.n = int(np.ceil(L / res_um))
        self.res = res_um
        self.center = np.asarray(fixed_center, float)
        self.lo = self.center - self.n * res_um / 2

    @property
    def shape(self):
        return (self.n, self.n)

    def _rc(self, B, offset_world, src_res):
        """affine_transform params for output canvas idx (r,c) -> source map idx, given source_world = B @ (canvas_world) + offset."""
        P = np.array([[0, 1], [1, 0]], float)
        Brc = P @ B @ P
        matrix = Brc * (self.res / src_res)
        off = (Brc @ (self.lo[::-1]) + offset_world[::-1]) / src_res
        return matrix, off

    def render_map(self, img, src_res, A=None, t0=None, order=1):
        """Render a frame map into the canvas.  canvas_world = A @ src_world + t0  (A=None: identity)."""
        if A is None:
            A, t0 = np.eye(2), np.zeros(2)
        B = np.linalg.inv(A)
        matrix, off = self._rc(B, -B @ t0, src_res)
        return affine_transform(img.astype(np.float32), matrix, offset=off, output_shape=self.shape, order=order, cval=0.0)

    def raster_points(self, pts_world, sigma_um):
        img = np.zeros(self.shape, np.float32)
        ij = np.rint((pts_world - self.lo) / self.res).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < self.n) & (ij[:, 1] >= 0) & (ij[:, 1] < self.n)
        np.add.at(img, (ij[ok, 1], ij[ok, 0]), 1.0)
        return gaussian_filter(img, sigma_um / self.res)

    def hull_mask(self, pts_world):
        h = ConvexHull(pts_world)
        path = MPath(pts_world[h.vertices])
        yy, xx = np.mgrid[0:self.n, 0:self.n]
        xy = np.column_stack([xx.ravel(), yy.ravel()]) * self.res + self.lo
        return path.contains_points(xy).reshape(self.shape).astype(np.float32)


def _normalise(m, mask):
    v = m[mask > 0.5]
    return m / max(float(v.mean()) if v.size else 1.0, 1e-9)


# ----------------------------------------------------------------------------- search
@dataclass
class SearchConfig:
    res_coarse: float = 8.0
    res_fine: float = 4.0
    rot_step: float = 4.0
    rot_range: tuple = (0.0, 360.0)
    scales: tuple = (0.95, 1.0, 1.05)
    fine_rot_half: float = 4.0
    fine_rot_step: float = 1.0
    fine_scales: tuple = (0.97, 0.985, 1.0, 1.015, 1.03)
    fine_window_um: float = 40.0
    chiralities: tuple = ("proper",)
    min_overlap_frac: float = 0.5          # of the smaller field
    peaks_per_pose: int = 3
    n_fine: int = 6
    combine: str = "zsum"                  # zsum | zmin | first | rerank (rank by layers[0], re-rank top-K by sum)
    distinct_um: float = 40.0
    distinct_deg: float = 8.0
    ambiguous_margin: float = 0.10         # prespecified (same value as A2 detector 'ambiguous_basin_margin')
    min_peak_z: float = 6.0


class PoseSearch:
    def __init__(self, layers: list[Layer], fixed_extent_um, moving_extent_um, fixed_center, moving_center, cfg: SearchConfig):
        self.layers, self.cfg = layers, cfg
        self.fe, self.me = fixed_extent_um, moving_extent_um
        self.fc, self.mc = np.asarray(fixed_center, float), np.asarray(moving_center, float)
        self.n_poses = 0

    # one pose -> NCC maps of all layers on the canvas at resolution res
    def _maps(self, canvas, chir, theta, s, cache):
        A = pose_matrix(chir, theta, s)
        t0 = canvas.center - A @ self.mc
        out = {}
        for L in self.layers:
            key = (L.name, canvas.res)
            if key not in cache:
                if L.fixed_points is not None:
                    f = canvas.raster_points(L.fixed_points, L.sigma_um); mf = canvas.hull_mask(L.fixed_points)
                else:
                    f = canvas.render_map(L.fixed_map, L.res_um); mf = (canvas.render_map(L.fixed_mask, L.res_um, order=0) > 0.5).astype(np.float32)
                cache[key] = (_normalise(f, mf), mf)
            f, mf = cache[key]
            if L.moving_points is not None:
                q = L.moving_points @ A.T + t0
                m = canvas.raster_points(q, L.sigma_um); mm = canvas.hull_mask(q)
            else:
                m = canvas.render_map(L.moving_map, L.res_um, A, t0); mm = (canvas.render_map(L.moving_mask, L.res_um, A, t0, order=0) > 0.5).astype(np.float32)
            if mm.sum() < 10:
                return None, t0
            m = _normalise(m, mm)
            min_ov = self.cfg.min_overlap_frac * min(mf.sum(), mm.sum())
            ncc, _ = masked_ncc(f, mf, m, mm, min_overlap=min_ov)
            fin = ncc[np.isfinite(ncc)]
            med = np.median(fin); mad = np.median(np.abs(fin - med)) + 1e-9
            out[L.name] = (ncc, (ncc - med) / (1.4826 * mad))
        self.n_poses += 1
        return out, t0

    def _combine(self, maps):
        names = [L.name for L in self.layers]
        zs = [maps[n][1] for n in names]
        rule = self.cfg.combine
        if len(zs) == 1 or rule in ("first", "rerank"):
            return zs[0]
        if rule == "zsum":
            return np.sum(zs, 0)
        if rule == "zmin":
            return np.min(zs, 0)
        raise ValueError(rule)

    def _peaks(self, canvas, maps, t0, chir, theta, s, npk, window=None, stage="coarse"):
        comb = self._combine(maps)
        comb = np.where(np.isfinite(comb), comb, -np.inf)
        n = canvas.n
        if window is not None:          # restrict to translations near a given t (um)
            cen = (np.asarray(window[0]) - t0) / canvas.res + (n - 1)
            r = window[1] / canvas.res
            yy, xx = np.ogrid[0:comb.shape[0], 0:comb.shape[1]]
            comb = np.where((xx - cen[0]) ** 2 + (yy - cen[1]) ** 2 <= r * r, comb, -np.inf)
        loc = (comb == maximum_filter(comb, size=max(3, int(round(16 / canvas.res)) | 1))) & np.isfinite(comb)
        idx = np.argwhere(loc)
        if not len(idx):
            return []
        vals = comb[loc]
        out = []
        for o in np.argsort(vals)[::-1][:npk]:
            i, j = idx[o]
            t = t0 + (np.array([j, i]) - (n - 1)) * canvas.res
            out.append(Candidate(chir, float(theta), float(s), t, float(vals[o]),
                                 {k: float(v[0][i, j]) for k, v in maps.items()}, {k: float(v[1][i, j]) for k, v in maps.items()}, stage))
        return out

    def distinct(self, cands):
        kept = []
        probe = self.mc + np.array([[0, 0], [200, 0], [0, 200], [-200, 0], [0, -200]], float)
        for c in sorted(cands, key=lambda c: -c.score):
            pc = probe @ c.A().T + c.t
            same = False
            for k in kept:
                pk = probe @ k.A().T + k.t
                if k.chirality == c.chirality and np.median(np.linalg.norm(pc - pk, axis=1)) < self.cfg.distinct_um:
                    same = True
                    break
            if not same:
                kept.append(c)
        return kept

    def run(self):
        t_start = time.perf_counter()
        cfg = self.cfg
        cache = {}
        cv = Canvas(self.fe, self.me, self.fc, cfg.res_coarse, max(cfg.scales) * 1.05)
        coarse = []
        for chir in cfg.chiralities:
            for th in np.arange(cfg.rot_range[0], cfg.rot_range[1], cfg.rot_step):
                for s in cfg.scales:
                    maps, t0 = self._maps(cv, chir, th, s, cache)
                    if maps is not None:
                        coarse += self._peaks(cv, maps, t0, chir, th, s, cfg.peaks_per_pose)
        coarse_d = self.distinct(coarse)
        cvf = Canvas(self.fe, self.me, self.fc, cfg.res_fine, max(cfg.scales) * 1.1)
        fine = []
        for c in coarse_d[: cfg.n_fine]:
            best = None
            for th in np.arange(c.theta - cfg.fine_rot_half, c.theta + cfg.fine_rot_half + 1e-9, cfg.fine_rot_step):
                for f in cfg.fine_scales:
                    maps, t0 = self._maps(cvf, c.chirality, th, c.scale * f, cache)
                    if maps is None:
                        continue
                    pk = self._peaks(cvf, maps, t0, c.chirality, th, c.scale * f, 1, window=(c.t, cfg.fine_window_um), stage="fine")
                    if pk and (best is None or pk[0].score > best.score):
                        best = pk[0]
            if best is not None:
                fine.append(best)
        if cfg.combine == "rerank" and len(self.layers) > 1:
            for c in fine:
                c.score = float(sum(c.layer_z.values()))
        fine_d = self.distinct(fine)
        top = fine_d[0] if fine_d else None
        second = fine_d[1] if len(fine_d) > 1 else None
        margin = (top.score - second.score) / abs(top.score) if (top and second and top.score != 0) else np.inf
        verdict = "NO_CANDIDATE" if top is None else (
            "AMBIGUOUS" if margin < cfg.ambiguous_margin else ("LOW_PEAK" if top.score < cfg.min_peak_z * (len(self.layers) if cfg.combine in ("zsum", "rerank") else 1) else "UNIQUE"))
        return {"top": top, "second": second, "margin": float(margin), "verdict": verdict,
                "fine": fine_d, "coarse_top": coarse_d[:10], "n_poses": self.n_poses, "runtime_s": time.perf_counter() - t_start}
