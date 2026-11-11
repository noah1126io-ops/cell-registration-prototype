"""Independent synthetic generator for external validation of the frozen A2 affine initialisation (2026-10-15).

Written from scratch: no code, latent template, coordinate formula or seed of `init_lib.make_pair` (2026-10-14) is used.
Model = 2D partial-sharing serial-section model (no 3D nucleus ellipsoids):

  family template  --sample-->  latent nuclei (persistent ids)
  latent nucleus   --membership-->  {both sections, fixed only, moving only}        (q_shared)
                   --detection-->   independent Bernoulli per section              (det_fixed, det_moving)
                   --local loss-->  disc patches dropped in one section            (loss_patches)
                   --centroid noise per section                                    (centroid_noise_um)
  moving section   --non-affine smooth deformation (affine part projected out)     (deform_rms_um)
                   --global affine to HE-pixel frame, cropped to the moving image  (offset/rotation/reflect/pixel size)

`q_shared` (fraction of latent nuclei present in both sections) and detection loss are separate processes.
Truth (persistent ids, global affine) is stored on the pair and is never passed to the algorithm.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

FAMILIES = {"G1": "Clustered", "G2": "Inhomogeneous", "G3": "Structured Tissue", "G4": "Repeated / Ambiguous"}
SEED_ROOT = 20261015  # distinct from all earlier seeds (11-13, 102-105, 301, 401, 2101-4105)
HARD_CORE_UM = 5.0


def rng_for(*keys) -> np.random.Generator:
    """Deterministic, collision-free stream per (purpose, family, condition, replicate) key."""
    digest = hashlib.sha256("|".join(map(str, keys)).encode()).digest()
    ints = [SEED_ROOT] + [int.from_bytes(digest[i:i + 4], "little") for i in range(0, 32, 4)]
    return np.random.default_rng(np.random.SeedSequence(ints))


def rotation(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


# ----------------------------------------------------------------------------- geometry helpers
def rect(x0, y0, w, h):
    return np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]], float)


def polygon_area(p):
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _ccw(p):
    x, y = p[:, 0], p[:, 1]
    return p if np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)) > 0 else p[::-1]


def clip_convex(subject, clip):
    """Sutherland-Hodgman intersection of two convex polygons."""
    out = _ccw(np.asarray(subject, float)).tolist()
    clip = _ccw(np.asarray(clip, float))
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        inp, out = out, []
        if not inp:
            break
        side = lambda p: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])  # noqa: E731
        for j in range(len(inp)):
            p, q = np.array(inp[j]), np.array(inp[(j + 1) % len(inp)])
            sp, sq = side(p), side(q)
            if sp >= 0:
                out.append(p.tolist())
            if (sp >= 0) != (sq >= 0):
                out.append((p + (q - p) * sp / (sp - sq)).tolist())
    return np.array(out) if len(out) >= 3 else np.zeros((0, 2))


def in_polygon(points, poly):
    from matplotlib.path import Path as MPath
    return MPath(poly).contains_points(points)


def hard_core_thin(points, rng, r=HARD_CORE_UM):
    """Random-order sequential inhibition (nuclei do not overlap)."""
    if len(points) == 0:
        return points
    order = rng.permutation(len(points))
    p = points[order]
    keep = np.ones(len(p), bool)
    for i, j in sorted(cKDTree(p).query_pairs(r)):
        if keep[i] and keep[j]:
            keep[j] = False
    return p[keep]


def to_target(points, rng, n_target):
    """Uniform random thinning to the target count (never adds points)."""
    if len(points) <= n_target:
        return points
    return points[np.sort(rng.choice(len(points), n_target, replace=False))]


def smooth_field(rng, box, px, corr_um, n=1):
    """Standardised Gaussian random field(s) on a grid covering box=(x0,y0,x1,y1)."""
    x = np.arange(box[0], box[2] + px, px); y = np.arange(box[1], box[3] + px, px)
    fields = []
    for _ in range(n):
        g = gaussian_filter(rng.normal(size=(len(y), len(x))), corr_um / px, mode="wrap")
        fields.append((g - g.mean()) / (g.std() + 1e-12))
    return x, y, fields


def sample_intensity(rng, box, x, y, lam, n_expected):
    """Inhomogeneous Poisson by thinning; lam on grid (y, x) is rescaled so E[N] = n_expected."""
    area = (box[2] - box[0]) * (box[3] - box[1])
    lam = np.clip(lam, 0, None)
    lam = lam / lam.mean() * n_expected / area
    n = rng.poisson(lam.max() * area)
    p = np.column_stack([rng.uniform(box[0], box[2], n), rng.uniform(box[1], box[3], n)])
    f = RegularGridInterpolator((y, x), lam, bounds_error=False, fill_value=0.0)(p[:, ::-1])
    return p[rng.random(n) < f / lam.max()]


# ----------------------------------------------------------------------------- family templates
# A template fixes the tissue-scale structure; `sample_family` draws nuclei from it. Two independent draws from the
# same template share tissue-scale pattern but no nuclei (used by the density-matched negative control).
def make_template(family, rng, box, p):
    t = {"family": family, "box": box}
    if family == "G1":  # Thomas / Neyman-Scott clusters + background
        sig = rng.uniform(14, 32); mu = rng.uniform(8, 16)
        area = (box[2] - box[0] + 6 * sig) * (box[3] - box[1] + 6 * sig)
        n_par = rng.poisson(area * p["latent_density"] * 0.75 * 1.6 / mu)
        t.update(sigma=sig, mu=mu,
                 parents=np.column_stack([rng.uniform(box[0] - 3 * sig, box[2] + 3 * sig, n_par),
                                          rng.uniform(box[1] - 3 * sig, box[3] + 3 * sig, n_par)]))
    elif family == "G2":  # log-Gaussian field x Voronoi step regions x gradient
        x, y, (g,) = smooth_field(rng, box, 8.0, rng.uniform(60, 120))
        k = rng.integers(6, 11)
        seeds = np.column_stack([rng.uniform(box[0], box[2], k), rng.uniform(box[1], box[3], k)])
        mult = np.exp(rng.uniform(np.log(0.25), np.log(2.5), k) * p.get("hetero", 1.0))
        mult[rng.choice(k, 1)] = 0.04                              # one near-empty region (abrupt drop)
        X, Y = np.meshgrid(x, y)
        reg = cKDTree(seeds).query(np.column_stack([X.ravel(), Y.ravel()]))[1].reshape(X.shape)
        u = rng.normal(size=2); u /= np.linalg.norm(u)
        span = max(box[2] - box[0], box[3] - box[1])
        grad = np.clip(1 + rng.uniform(0.5, 1.0) * ((X - X.mean()) * u[0] + (Y - Y.mean()) * u[1]) / span, 0.2, None)
        t.update(x=x, y=y, lam=np.exp(0.6 * p.get("hetero", 1.0) * g) * mult[reg] * grad)
    elif family == "G3":  # epithelium-like bands, gland-like rings, sparse stroma, clefts, nucleus-free areas
        w, h = box[2] - box[0], box[3] - box[1]
        bands = []
        for _ in range(rng.integers(3, 6)):
            bands.append(dict(c=np.array([rng.uniform(box[0], box[2]), rng.uniform(box[1], box[3])]), ang=rng.uniform(0, np.pi),
                              length=rng.uniform(0.5, 1.0) * max(w, h), amp=rng.uniform(20, 90), wl=rng.uniform(250, 600),
                              ph=rng.uniform(0, 2 * np.pi), rows=int(rng.integers(2, 4))))
        n_gl = rng.poisson(w * h / (260.0**2))
        glands = [dict(c=np.array([rng.uniform(box[0], box[2]), rng.uniform(box[1], box[3])]), a=rng.uniform(35, 80),
                       e=rng.uniform(0.6, 1.0), ang=rng.uniform(0, np.pi), rows=int(rng.integers(1, 3))) for _ in range(n_gl)]
        clefts = [dict(c=np.array([rng.uniform(box[0], box[2]), rng.uniform(box[1], box[3])]), ang=rng.uniform(0, np.pi),
                       length=rng.uniform(150, 500), width=rng.uniform(15, 40), bend=rng.uniform(-0.002, 0.002))
                  for _ in range(rng.integers(3, 7))]
        holes = [dict(c=np.array([rng.uniform(box[0], box[2]), rng.uniform(box[1], box[3])]), r=rng.uniform(70, 180))
                 for _ in range(rng.integers(1, 3))]
        t.update(bands=bands, glands=glands, clefts=clefts, holes=holes)
    elif family == "G4":  # one motif tiled on a lattice; per-copy jitter / dropout
        P = rng.uniform(115, 170); ang = rng.uniform(0, 90); shear = rng.uniform(-0.15, 0.15)
        B = rotation(ang) @ np.array([[P, shear * P], [0, P * rng.uniform(0.9, 1.1)]])
        motif = []
        nring = int(2 * np.pi * P / 4.2 / 8.0)
        th = np.linspace(0, 2 * np.pi, nring, endpoint=False)
        motif.append(np.column_stack([P / 4.2 * np.cos(th), P / 4.2 * np.sin(th)]) + P * 0.3)
        cc = rng.uniform(0.55, 0.8, 2) * P
        motif.append(cc + rng.normal(0, P * 0.06, (int(rng.integers(10, 18)), 2)))
        motif.append(rng.uniform(0, P, (int(P * P * p["latent_density"] * 0.7), 2)))
        motif = np.concatenate(motif)
        motif = hard_core_thin(motif, rng, 6.5)
        t.update(period=P, basis=B, motif_unit=np.linalg.solve(np.diag([P, P]), motif.T).T, jitter=p.get("copy_jitter_um", rng.uniform(2.5, 4.0)),
                 keep=p.get("copy_keep", 0.8), extra=0.15)
    else:
        raise ValueError(family)
    return t


def _curve_points(rng, b, spacing=7.5, row_gap=8.0, jit=1.6):
    s = np.arange(-b["length"] / 2, b["length"] / 2, spacing)
    out = []
    for r in range(b["rows"]):
        ss = s + rng.uniform(0, spacing)
        off = (r - (b["rows"] - 1) / 2) * row_gap
        y = b["amp"] * np.sin(2 * np.pi * ss / b["wl"] + b["ph"]) + off
        local = np.column_stack([ss, y]) + rng.normal(0, jit, (len(ss), 2))
        out.append(local @ rotation(np.degrees(b["ang"])).T + b["c"])
    return np.concatenate(out)


def _ring_points(rng, g, spacing=7.5, jit=1.4):
    out = []
    for r in range(g["rows"]):
        a = g["a"] + r * 8.0; bb = g["a"] * g["e"] + r * 8.0
        n = max(6, int(np.pi * (a + bb) / spacing))
        th = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(0, 2 * np.pi / n)
        local = np.column_stack([a * np.cos(th), bb * np.sin(th)]) + rng.normal(0, jit, (n, 2))
        out.append(local @ rotation(np.degrees(g["ang"])).T + g["c"])
    return np.concatenate(out)


def _cleft_mask(points, c):
    u = np.array([np.cos(c["ang"]), np.sin(c["ang"])]); v = np.array([-u[1], u[0]])
    rel = points - c["c"]
    al, ac = rel @ u, rel @ v
    return (np.abs(al) < c["length"] / 2) & (np.abs(ac - c["bend"] * al**2) < c["width"] / 2)


def sample_family(t, rng, n_target):
    """Draw latent nuclei from a template (hard-core 5 um), thinned to n_target if over-produced."""
    box = t["box"]
    area = (box[2] - box[0]) * (box[3] - box[1])
    fam = t["family"]
    if fam == "G1":
        par = t["parents"]
        k = rng.poisson(t["mu"], len(par))
        kids = np.repeat(par, k, axis=0) + rng.normal(0, t["sigma"], (k.sum(), 2))
        bg = np.column_stack([rng.uniform(box[0], box[2], int(0.4 * n_target)), rng.uniform(box[1], box[3], int(0.4 * n_target))])
        pts = np.concatenate([kids, bg])
    elif fam == "G2":
        pts = sample_intensity(rng, box, t["x"], t["y"], t["lam"], 1.5 * n_target)
    elif fam == "G3":
        parts = [_curve_points(rng, b) for b in t["bands"]] + [_ring_points(rng, g) for g in t["glands"]]
        struct = np.concatenate(parts) if parts else np.zeros((0, 2))
        stroma = np.column_stack([rng.uniform(box[0], box[2], int(1.4 * n_target)), rng.uniform(box[1], box[3], int(1.4 * n_target))])
        # stroma is sparse: thin it so structures carry ~half of the nuclei
        stroma = stroma[rng.random(len(stroma)) < 0.55]
        pts = np.concatenate([struct, stroma])
        bad = np.zeros(len(pts), bool)
        for c in t["clefts"]:
            bad |= _cleft_mask(pts, c)
        for hlo in t["holes"]:
            bad |= np.linalg.norm(pts - hlo["c"], axis=1) < hlo["r"]
        for g in t["glands"]:  # lumen is nucleus-free
            rel = (pts - g["c"]) @ rotation(np.degrees(g["ang"]))
            bad |= (rel[:, 0] / (g["a"] - 6)) ** 2 + (rel[:, 1] / (g["a"] * g["e"] - 6)) ** 2 < 1
        pts = pts[~bad]
    elif fam == "G4":
        B = t["basis"]; Binv = np.linalg.inv(B)
        corners = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]) @ Binv.T
        i0, j0 = np.floor(corners.min(0)).astype(int) - 1; i1, j1 = np.ceil(corners.max(0)).astype(int) + 1
        motif = t["motif_unit"] * t["period"]
        out = []
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                keep = rng.random(len(motif)) < t["keep"]
                local = motif[keep] + rng.normal(0, t["jitter"], (keep.sum(), 2))
                out.append(local @ (B / t["period"]).T + B @ np.array([i, j]))
        pts = np.concatenate(out)
        pts = pts[(pts[:, 0] >= box[0]) & (pts[:, 0] <= box[2]) & (pts[:, 1] >= box[1]) & (pts[:, 1] <= box[3])]
        extra = np.column_stack([rng.uniform(box[0], box[2], int(t["extra"] * n_target)), rng.uniform(box[1], box[3], int(t["extra"] * n_target))])
        pts = np.concatenate([pts, extra])
    pts = pts[(pts[:, 0] >= box[0]) & (pts[:, 0] <= box[2]) & (pts[:, 1] >= box[1]) & (pts[:, 1] <= box[3])]
    pts = hard_core_thin(pts, rng)
    return to_target(pts, rng, n_target)


# ----------------------------------------------------------------------------- deformation
def nonaffine_field(rng, box, region_poly, rms_um, corr_um=None):
    """Smooth displacement field with its least-squares affine part (over region_poly) removed, scaled to rms_um."""
    if rms_um <= 0:
        return None
    corr = corr_um or rng.uniform(110, 200)
    x, y, (fx, fy) = smooth_field(rng, box, 10.0, corr, n=2)
    X, Y = np.meshgrid(x, y)
    G = np.column_stack([X.ravel(), Y.ravel()])
    D = np.column_stack([fx.ravel(), fy.ravel()])
    sel = in_polygon(G, region_poly)
    A = np.column_stack([G[sel], np.ones(sel.sum())])
    coef, *_ = np.linalg.lstsq(A, D[sel], rcond=None)
    D = D - np.column_stack([G, np.ones(len(G))]) @ coef
    D *= rms_um / np.sqrt(np.mean(np.sum(D[sel] ** 2, axis=1)))
    ix = RegularGridInterpolator((y, x), D[:, 0].reshape(X.shape)); iy = RegularGridInterpolator((y, x), D[:, 1].reshape(X.shape))
    return lambda p: np.column_stack([ix(p[:, ::-1]), iy(p[:, ::-1])])


# ----------------------------------------------------------------------------- pair
DEFAULTS = dict(
    fixed_fov_um=(1000.0, 1000.0), mov_px=(900, 900), pixel_um=1.10, rot_deg=2.0, reflect=True,
    offset_um=60.0, offset_angle_deg=None, q_shared=0.65, det_fixed=0.93, det_moving=0.90, density=0.0020,
    deform_rms_um=2.0, centroid_noise_um=0.6, loss_patches=0, hetero=1.0,
)


@dataclass
class SectionPair:
    name: str
    family: str
    fixed: np.ndarray
    moving_px: np.ndarray
    width: float
    height: float
    M_true: np.ndarray
    t_true: np.ndarray
    fixed_ids: np.ndarray
    moving_ids: np.ndarray
    true_fixed_index: np.ndarray        # per moving point, index into fixed of the same persistent id, -1 if none
    moving_world_true: np.ndarray       # global-affine image of moving_px (= deformed + noisy section-B position)
    fixed_fov: np.ndarray
    moving_fov_world: np.ndarray
    params: dict = field(default_factory=dict)

    def to_init_pair(self):
        import init_lib as L  # frozen 2026-10-14 container type
        return L.Pair(self.name, self.fixed, self.moving_px, self.width, self.height, self.M_true, self.t_true,
                      self.true_fixed_index, self.moving_world_true, self.params)


def _geometry(p, rng):
    Wf, Hf = p["fixed_fov_um"]
    fixed_fov = rect(0, 0, Wf, Hf)
    W, H = p["mov_px"]
    C = np.diag([1.0, -1.0]) if p["reflect"] else np.eye(2)
    M = p["pixel_um"] * rotation(p["rot_deg"]) @ C
    ang = np.deg2rad(p["offset_angle_deg"] if p["offset_angle_deg"] is not None else rng.uniform(0, 360))
    centre = np.array([Wf / 2, Hf / 2]) + p["offset_um"] * np.array([np.cos(ang), np.sin(ang)])
    t = centre - M @ np.array([W / 2, H / 2])
    mov_fov = rect(0, 0, W, H) @ M.T + t
    return fixed_fov, mov_fov, M, t, float(np.degrees(ang))


def make_pair(family, cond_key, rep, overrides=None, *, mode="normal", name=None):
    """mode: normal | unrelated (independent tissue for moving) | density_matched (same template, independent nuclei)."""
    p = dict(DEFAULTS); p.update(overrides or {})
    g_rng = rng_for("geometry", family, cond_key, rep)
    fixed_fov, mov_fov, M, t, ang = _geometry(p, g_rng)
    allp = np.concatenate([fixed_fov, mov_fov])
    box = (allp[:, 0].min() - 120, allp[:, 1].min() - 120, allp[:, 0].max() + 120, allp[:, 1].max() + 120)
    area = (box[2] - box[0]) * (box[3] - box[1])
    q = p["q_shared"] if mode == "normal" else 0.0
    p_in_fixed = q + (1 - q) / 2 if mode == "normal" else 1.0
    latent_density = p["density"] / (p_in_fixed * p["det_fixed"])
    p["latent_density"] = latent_density
    tmpl = make_template(family, rng_for("template", family, cond_key, rep), box, p)
    n_lat = int(round(latent_density * area))
    lat_a = sample_family(tmpl, rng_for("nuclei", family, cond_key, rep), n_lat)
    m_rng = rng_for("membership", family, cond_key, rep)
    if mode == "normal":
        u = m_rng.random(len(lat_a))
        in_f = u < q + (1 - q) / 2
        in_m = (u < q) | (u >= q + (1 - q) / 2)
        latent = lat_a; ids = np.arange(len(latent))
        lat_f, ids_f, lat_m, ids_m = latent[in_f], ids[in_f], latent[in_m], ids[in_m]
    else:
        if mode == "unrelated":
            tmpl_b = make_template(family, rng_for("template-b", family, cond_key, rep), box, p)
        else:
            tmpl_b = tmpl
        lat_b = sample_family(tmpl_b, rng_for("nuclei-b", family, cond_key, rep), n_lat)
        lat_f, ids_f = lat_a, np.arange(len(lat_a))
        lat_m, ids_m = lat_b, 10_000_000 + np.arange(len(lat_b))   # disjoint persistent ids
    det_f = m_rng.random(len(lat_f)) < p["det_fixed"]
    det_m = m_rng.random(len(lat_m)) < p["det_moving"]
    for poly, pts, keep in ((fixed_fov, lat_f, det_f), (mov_fov, lat_m, det_m)):
        lo, hi = poly.min(0), poly.max(0)
        for _ in range(p["loss_patches"]):
            c = m_rng.uniform(lo, hi); r = m_rng.uniform(40, 110)
            keep &= ~((np.linalg.norm(pts - c, axis=1) < r) & (m_rng.random(len(pts)) < 0.95))
    fixed_w = lat_f + m_rng.normal(0, p["centroid_noise_um"], lat_f.shape)
    fsel = det_f & in_polygon(fixed_w, fixed_fov)
    fixed, fixed_ids = fixed_w[fsel], ids_f[fsel]
    dfield = nonaffine_field(rng_for("deform", family, cond_key, rep), box, mov_fov, p["deform_rms_um"])
    world_b = lat_m + (dfield(lat_m) if dfield is not None else 0.0) + m_rng.normal(0, p["centroid_noise_um"], lat_m.shape)
    px = np.linalg.solve(M, (world_b - t).T).T
    W, H = p["mov_px"]
    crop = det_m & (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
    moving_px, moving_ids, mw = px[crop], ids_m[crop], world_b[crop]
    pos = {k: i for i, k in enumerate(fixed_ids)}
    tfi = np.array([pos.get(k, -1) for k in moving_ids], int)
    inter = clip_convex(fixed_fov, mov_fov)
    a_f, a_m, a_i = polygon_area(fixed_fov), polygon_area(mov_fov), polygon_area(inter) if len(inter) else 0.0
    params = {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()}
    params.update(family=family, condition=cond_key, rep=rep, mode=mode, offset_angle_deg=ang,
                  n_fixed=int(len(fixed)), n_moving=int(len(moving_px)), n_latent=int(len(lat_a)),
                  shared_fraction_of_moving=float(np.mean(tfi >= 0)) if len(tfi) else 0.0,
                  shared_fraction_of_fixed=float(np.isin(fixed_ids, moving_ids).mean()) if len(fixed_ids) else 0.0,
                  overlap_area_um2=a_i, overlap_frac_fixed=a_i / a_f, overlap_frac_moving=a_i / a_m,
                  overlap_iou=a_i / (a_f + a_m - a_i), overlap_min_coverage=a_i / max(a_f, a_m),
                  centroid_shift_um=float(np.linalg.norm(fixed.mean(0) - mw.mean(0))) if len(fixed) and len(mw) else np.nan,
                  observed_fixed_density=float(len(fixed) / a_f), template_meta=_template_meta(tmpl))
    return SectionPair(name or f"{family}:{cond_key}:{rep}", family, fixed, moving_px, float(W), float(H), M, t,
                       fixed_ids, moving_ids, tfi, mw, fixed_fov, mov_fov, params)


def _template_meta(t):
    keep = {"G1": ("sigma", "mu"), "G4": ("period", "jitter", "keep")}.get(t["family"], ())
    out = {k: float(t[k]) for k in keep}
    if t["family"] == "G3":
        out.update(n_bands=len(t["bands"]), n_glands=len(t["glands"]), n_clefts=len(t["clefts"]), n_holes=len(t["holes"]))
    return out


# ----------------------------------------------------------------------------- evaluation geometry
def common_region_grid(pair, n=41):
    """Moving-image grid points (px) whose TRUE world position lies in the fixed FOV (physical common region)."""
    gx, gy = np.meshgrid(np.linspace(0, pair.width, n), np.linspace(0, pair.height, n))
    g = np.column_stack([gx.ravel(), gy.ravel()])
    w = g @ pair.M_true.T + pair.t_true
    return g[in_polygon(w, pair.fixed_fov)], g


def affine_errors_common(M, t, pair):
    """RMS / max |A_est(p) - A_true(p)| (um) over the common physical region; falls back to full field if empty."""
    g_common, g_full = common_region_grid(pair)
    out = {}
    for tag, g in (("common", g_common), ("fullfield", g_full)):
        if len(g) == 0:
            out[f"rms_{tag}_um"], out[f"max_{tag}_um"] = np.nan, np.nan
            continue
        e = np.linalg.norm(g @ np.asarray(M).T + t - (g @ pair.M_true.T + pair.t_true), axis=1)
        out[f"rms_{tag}_um"], out[f"max_{tag}_um"] = float(np.sqrt(np.mean(e**2))), float(e.max())
    out["n_common_grid"] = int(len(g_common))
    return out
