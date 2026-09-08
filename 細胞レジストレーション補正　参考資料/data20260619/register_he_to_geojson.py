#!/usr/bin/env python3
"""
register_he_to_geojson.py

Register an H&E background image to a nuclei-segmentation GeoJSON
(produced by ``ome_zarr_to_geojson.py``) using nucleus centroids as
correspondences, and emit a ``view_state.json`` that places the HE
image at the correct position in the ArcScape world-coordinate
(``μm``, Y-flipped) space.

Pipeline
--------
1.  Detect nucleus centroids in the HE image (StarDist by default).
2.  Load nucleus centroids from the GeoJSON (``properties.centroid_x``
    / ``properties.centroid_y``, already in ``μm`` and Y-flipped to
    match the HE orientation).
3.  Estimate an initial similarity transform (translation + isotropic
    scale; multiple initial rotations are tried) by matching the two
    point-set means and variances.
4.  Refine with ICP using the Umeyama closed-form similarity solver,
    with a trimmed-RANSAC rejection of the worst residuals at each
    iteration.
5.  Express the resulting similarity transform as a ArcScape
    ``view_state.json`` (``bgOffsetX``, ``bgOffsetY``, ``bgScale``,
    ``bgRotation``) that round-trips through
    ``arcscape/visualization/interactive/templates/_init.js.html#buildBgImageCorners``.

The resulting ``view_state.json`` can be:

* Written to disk with ``--output-view-state out.json``
* Injected directly into a ``.arcs`` bundle with ``--inject-arcs``
  (uses ``tools/arcs_view_state.py``).

A diagnostic PNG can be saved with ``--report report.png`` (matplotlib).

Backends
--------
``--method stardist`` (default)
    Uses ``StarDist2D.from_pretrained('2D_versatile_he')``. Requires
    ``stardist`` / ``csbdeep`` / ``tensorflow`` to be importable.
``--method classical``
    Uses scikit-image color deconvolution + thresholding + peak
    detection. Heavy ML deps not required. Less accurate, useful as a
    fallback.

Usage
-----
::

    # 1) Single sample, write a standalone view_state.json
    python tools/register_he_to_geojson.py \\
        --he tmp/seqfish_20260520/sample4_cropped.jpg \\
        --geojson tmp/seqfish_20260520/S2500586-1_dl20_um.geojson \\
        --output-view-state tmp/S2500586-1_view_state.json \\
        --report tmp/S2500586-1_registration.png

    # 2) Bake the result straight into a .arcs bundle
    python tools/register_he_to_geojson.py \\
        --he tmp/seqfish_20260520/sample4_cropped.jpg \\
        --geojson tmp/seqfish_20260520/S2500586-1_dl20_um.geojson \\
        --inject-arcs results/S2500586-1_leiden1.6.arcs \\
        --force

    # 3) Classical (no StarDist) backend
    python tools/register_he_to_geojson.py \\
        --he sample.jpg --geojson seg.geojson --method classical \\
        --output-view-state out.json

Conventions
-----------
The ``view_state.json`` emitted here follows the runtime model in
``_init.js.html#buildBgImageCorners``:

    bx0 = bounds[0] * bgScale + bgOffsetX
    by0 = bounds[1] * bgScale + bgOffsetY
    bx1 = bounds[2] * bgScale + bgOffsetX
    by1 = bounds[3] * bgScale + bgOffsetY
    corners = rotate([(bx0,by1), (bx0,by0), (bx1,by0), (bx1,by1)],
                     center=((bx0+bx1)/2, (by0+by1)/2),
                     angle=bgRotation)

To express a general 4-DOF similarity (translation + isotropic scale +
rotation) under this model with no change to the template, we set:

    bounds   = [cx - s*W/2, cy - s*H/2, cx + s*W/2, cy + s*H/2]
    bgScale  = 1.0
    bgOffset = (0, 0)
    bgRotation = θ_deg

where ``(cx, cy) = t + s * R(θ) · (W/2, H/2)`` is the world location
of the HE image center after applying the similarity. The image bounds
themselves carry the scale and translation; ``bgRotation`` carries the
rotation, applied around the bounds' centre (= ``(cx, cy)``).

The full view_state payload therefore contains the keys consumed by the
viewer (``bgOffsetX/Y``, ``bgScale``, ``bgRotation``, ``bgOpacity``,
``bgDepth``, ``bgVisible``) plus an ``imageBounds`` field that the
caller (or a small wrapper in the build script) can splice into the
HTML's ``BG_IMAGE_BOUNDS`` constant if a freshly-built bundle is being
produced. When injecting into an *existing* bundle, only the runtime
state is changed — the embedded ``BG_IMAGE_BOUNDS`` stays put and the
similarity is realised entirely through ``bgScale`` / ``bgOffset`` /
``bgRotation`` derived from the already-baked bounds. The conversion
between the two parameterisations is handled by
``similarity_to_view_state``.

Standard-library + numpy + scipy + scikit-image only at import time.
StarDist / tensorflow / matplotlib are imported lazily.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class Similarity:
    """4-DOF similarity transform: world = s * R(theta) @ src + t."""
    scale: float
    rotation_rad: float
    translation: Tuple[float, float]

    def apply(self, pts: np.ndarray) -> np.ndarray:
        c, sn = np.cos(self.rotation_rad), np.sin(self.rotation_rad)
        R = self.scale * np.array([[c, -sn], [sn, c]])
        return pts @ R.T + np.asarray(self.translation)


@dataclass
class RegistrationReport:
    method: str
    he_path: str
    geojson_path: str
    n_he_centroids: int
    n_gj_centroids: int
    init_rotation_deg: float
    iterations: int
    final_mean_residual: float
    final_median_residual: float
    final_inliers: int
    similarity: Similarity
    image_size_px: Tuple[int, int]
    stardist_model: Optional[str] = None
    stardist_version: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d['similarity'] = {
            'scale': self.similarity.scale,
            'rotation_deg': float(np.rad2deg(self.similarity.rotation_rad)),
            'translation': list(self.similarity.translation),
        }
        return d


# ---------------------------------------------------------------------------
# Nucleus detection backends
# ---------------------------------------------------------------------------
def _read_he_rgb(jpg_path: str) -> np.ndarray:
    from PIL import Image
    with Image.open(jpg_path) as im:
        return np.asarray(im.convert('RGB'))


def detect_nuclei_stardist(
    jpg_path: str,
    model_name: str = '2D_versatile_he',
    model_basedir: Optional[str] = None,
    prob_thresh: Optional[float] = None,
    nms_thresh: Optional[float] = None,
    n_tiles: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, str]:
    """Return ((N, 2) centroids in HE px (x, y), stardist version str)."""
    # Reduce TF log noise before importing.
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    try:
        import stardist
        from stardist.models import StarDist2D
        from csbdeep.utils import normalize
        from skimage.measure import regionprops
    except ImportError as e:
        raise RuntimeError(
            f"StarDist backend requires `stardist`, `csbdeep`, "
            f"`tensorflow`, `scikit-image`: {e}"
        )

    img = _read_he_rgb(jpg_path)
    img_n = normalize(img, 1.0, 99.8, axis=(0, 1))

    if model_basedir is not None:
        model = StarDist2D(None, name=model_name, basedir=model_basedir)
    else:
        model = StarDist2D.from_pretrained(model_name)

    labels, _ = model.predict_instances(
        img_n,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
        n_tiles=n_tiles,
        verbose=False,
    )
    props = regionprops(labels)
    if not props:
        raise RuntimeError(f"StarDist found no nuclei in {jpg_path}")
    centroids = np.array([rp.centroid for rp in props])  # (row, col)
    return centroids[:, [1, 0]].astype(float), str(stardist.__version__)


def detect_nuclei_classical(
    jpg_path: str,
    smooth_sigma: float = 1.0,
    min_distance: int = 4,
    hematoxylin_percentile: float = 70.0,
) -> np.ndarray:
    """Return (N, 2) centroids in HE pixel coords (x, y).

    Color-deconvolve to hematoxylin, smooth, threshold at a percentile of
    the hematoxylin signal, then take local maxima as nucleus centroids.
    """
    try:
        from skimage.color import rgb2hed
        from skimage.feature import peak_local_max
        from skimage.filters import gaussian
    except ImportError as e:
        raise RuntimeError(f"Classical backend requires scikit-image: {e}")

    img = _read_he_rgb(jpg_path).astype(np.float32) / 255.0
    hed = rgb2hed(img)
    h = gaussian(hed[..., 0], sigma=smooth_sigma)
    thr = float(np.percentile(h, hematoxylin_percentile))
    mask = h > thr
    peaks = peak_local_max(
        h,
        min_distance=min_distance,
        labels=mask.astype(int),
        exclude_border=False,
    )
    if peaks.size == 0:
        raise RuntimeError(
            f"Classical backend found no nuclei peaks in {jpg_path}"
        )
    return peaks[:, [1, 0]].astype(float)  # (x, y)


# ---------------------------------------------------------------------------
# GeoJSON nuclei centroid loader
# ---------------------------------------------------------------------------
def load_geojson_centroids(path: str) -> np.ndarray:
    """Return (N, 2) centroids in world μm coords (x, y).

    Uses ``feature.properties.centroid_x/centroid_y`` if present,
    otherwise falls back to the polygon's vertex mean.
    """
    with open(path) as f:
        gj = json.load(f)

    pts = []
    for feat in gj.get('features', []):
        p = feat.get('properties', {}) or {}
        if 'centroid_x' in p and 'centroid_y' in p:
            pts.append([float(p['centroid_x']), float(p['centroid_y'])])
            continue
        geom = feat.get('geometry', {}) or {}
        if geom.get('type') == 'Polygon':
            coords = np.asarray(geom['coordinates'][0], dtype=float)
            pts.append(coords.mean(axis=0).tolist())
    if not pts:
        raise RuntimeError(f"No centroids extractable from {path}")
    return np.asarray(pts, dtype=float)


# ---------------------------------------------------------------------------
# Similarity estimation (Umeyama)
# ---------------------------------------------------------------------------
def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> Similarity:
    """Closed-form least-squares similarity ``dst ≈ s * R · src + t``.

    See Umeyama (1991). Assumes ``src`` and ``dst`` are paired
    (``src[i]`` ↔ ``dst[i]``) and have the same shape ``(N, 2)``.
    """
    assert src.shape == dst.shape and src.ndim == 2 and src.shape[1] == 2
    n = src.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 correspondences for Umeyama")

    src_mu = src.mean(axis=0)
    dst_mu = dst.mean(axis=0)
    src_c = src - src_mu
    dst_c = dst - dst_mu

    var_src = (src_c ** 2).sum() / n
    H = (dst_c.T @ src_c) / n

    U, S, Vt = np.linalg.svd(H)
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1.0

    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / var_src) if var_src > 0 else 1.0
    t = dst_mu - s * (R @ src_mu)
    theta = float(np.arctan2(R[1, 0], R[0, 0]))
    return Similarity(scale=s, rotation_rad=theta,
                      translation=(float(t[0]), float(t[1])))


def initial_similarity(src: np.ndarray, dst: np.ndarray,
                       rotation_rad: float = 0.0) -> Similarity:
    """Match means and isotropic scale, with a fixed initial rotation."""
    src_mu = src.mean(axis=0)
    dst_mu = dst.mean(axis=0)
    src_scale = float(np.sqrt(((src - src_mu) ** 2).sum(axis=1).mean()))
    dst_scale = float(np.sqrt(((dst - dst_mu) ** 2).sum(axis=1).mean()))
    s = dst_scale / src_scale if src_scale > 0 else 1.0
    c, sn = np.cos(rotation_rad), np.sin(rotation_rad)
    R = np.array([[c, -sn], [sn, c]])
    t = dst_mu - s * (R @ src_mu)
    return Similarity(scale=s, rotation_rad=rotation_rad,
                      translation=(float(t[0]), float(t[1])))


# ---------------------------------------------------------------------------
# ICP with trimmed correspondences
# ---------------------------------------------------------------------------
def icp_similarity(
    src: np.ndarray,
    dst: np.ndarray,
    initial: Similarity,
    *,
    max_iter: int = 100,
    tol: float = 1e-4,
    trim_quantile: float = 0.8,
) -> Tuple[Similarity, dict]:
    """ICP refining a similarity transform ``src -> dst``.

    Each iteration pairs every warped source point to its nearest
    target neighbor, keeps the lowest ``trim_quantile`` fraction of
    distances, and re-fits the similarity via Umeyama on the kept set.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(dst)
    transform = initial
    prev_err = float('inf')
    iters = 0
    info = {'mean_residual': None, 'median_residual': None, 'inliers': 0}

    for iters in range(1, max_iter + 1):
        warped = transform.apply(src)
        dists, idx = tree.query(warped, k=1)
        if trim_quantile >= 1.0:
            keep = np.ones_like(dists, dtype=bool)
        else:
            cutoff = float(np.quantile(dists, trim_quantile))
            keep = dists <= cutoff
        if keep.sum() < 3:
            break
        a = src[keep]
        b = dst[idx[keep]]
        transform = umeyama_similarity(a, b)
        mean_err = float(dists[keep].mean())
        info.update({
            'mean_residual': mean_err,
            'median_residual': float(np.median(dists[keep])),
            'inliers': int(keep.sum()),
        })
        if abs(prev_err - mean_err) < tol:
            break
        prev_err = mean_err

    info['iterations'] = iters
    return transform, info


def best_of_initial_rotations(
    src: np.ndarray,
    dst: np.ndarray,
    rotations_deg: Sequence[float],
    icp_kwargs: dict,
) -> Tuple[Similarity, dict, float]:
    """Run ICP from each initial rotation; return the best result."""
    best: Optional[Tuple[Similarity, dict, float]] = None
    for deg in rotations_deg:
        init = initial_similarity(src, dst, np.deg2rad(deg))
        sim, info = icp_similarity(src, dst, init, **icp_kwargs)
        err = info['mean_residual']
        if err is None:
            continue
        if best is None or err < best[1]['mean_residual']:
            best = (sim, info, float(deg))
    if best is None:
        raise RuntimeError("All ICP starts failed to produce a result")
    return best


# ---------------------------------------------------------------------------
# view_state conversion
# ---------------------------------------------------------------------------
def similarity_to_view_state(
    sim: Similarity,
    image_size_px: Tuple[int, int],
    *,
    base_image_bounds: Optional[Sequence[float]] = None,
    bg_opacity: int = 200,
    bg_depth: float = 0.0,
    bg_visible: bool = True,
) -> dict:
    """Convert a HE-px → world-μm similarity into a ArcScape view_state.

    Two modes:

    * ``base_image_bounds is None``:
        The view_state assumes the HTML's ``BG_IMAGE_BOUNDS`` will be
        rewritten to ``[cx - s*W/2, cy - s*H/2, cx + s*W/2, cy + s*H/2]``
        where ``(cx, cy)`` is the world-space centre of the warped HE.
        ``bgScale = 1``, ``bgOffset = 0``, ``bgRotation = θ_deg``.
        Suitable for *new* bundles built with ``image_bounds`` supplied.

    * ``base_image_bounds is not None``:
        The HTML already carries a fixed ``BG_IMAGE_BOUNDS``
        (typical of an existing ``.arcs``). We solve for the
        ``bgScale``, ``bgOffset``, ``bgRotation`` that realise the
        target similarity when composed with that fixed base. The
        composition in ``buildBgImageCorners`` is:

            world_corner = R(θ) · (bounds * S + O − c) + c
                          where c = ((bounds*S + O).x_mid, .y_mid)

        Decomposition:
            S = s * (W_target / W_base)        # match widths
            θ = sim.rotation_rad
            O chosen so the warped base-rectangle centre lands at
                centre_world = t + s * R(θ) · (W/2, H/2)
    """
    W_px, H_px = image_size_px
    s = sim.scale
    theta = sim.rotation_rad
    tx, ty = sim.translation
    c, sn = np.cos(theta), np.sin(theta)
    R = np.array([[c, -sn], [sn, c]])
    centre_world = np.array([tx, ty]) + s * (R @ np.array([W_px / 2.0,
                                                            H_px / 2.0]))

    if base_image_bounds is None:
        new_bounds = [
            float(centre_world[0] - s * W_px / 2.0),
            float(centre_world[1] - s * H_px / 2.0),
            float(centre_world[0] + s * W_px / 2.0),
            float(centre_world[1] + s * H_px / 2.0),
        ]
        return {
            'bgOffsetX': 0.0,
            'bgOffsetY': 0.0,
            'bgScale': 1.0,
            'bgRotation': float(np.rad2deg(theta)),
            'bgOpacity': int(bg_opacity),
            'bgDepth': float(bg_depth),
            'bgVisible': bool(bg_visible),
            'imageBounds': new_bounds,
        }

    bx0, by0, bx1, by1 = [float(v) for v in base_image_bounds]
    W_base = bx1 - bx0
    H_base = by1 - by0
    if W_base <= 0 or H_base <= 0:
        raise ValueError(f"Degenerate base_image_bounds={base_image_bounds}")
    # Solve scale that maps base rect width to the target world width (= s * W_px)
    bgScale = (s * W_px) / W_base
    # After applying bgScale, the rect midpoint is:
    #   mid_after_scale = ((bx0+bx1)/2 * bgScale, (by0+by1)/2 * bgScale)
    # We want mid_after_scale + bgOffset == centre_world.
    mid_base = np.array([(bx0 + bx1) / 2.0, (by0 + by1) / 2.0])
    mid_after_scale = mid_base * bgScale
    bgOffset = centre_world - mid_after_scale

    return {
        'bgOffsetX': float(bgOffset[0]),
        'bgOffsetY': float(bgOffset[1]),
        'bgScale': float(bgScale),
        'bgRotation': float(np.rad2deg(theta)),
        'bgOpacity': int(bg_opacity),
        'bgDepth': float(bg_depth),
        'bgVisible': bool(bg_visible),
    }


# ---------------------------------------------------------------------------
# .arcs helpers
# ---------------------------------------------------------------------------
def read_image_bounds_from_arcs(arcs_path: Path) -> Optional[list]:
    """Best-effort: read ``BG_IMAGE_BOUNDS`` from the bundle's HTML or
    its external ``_data.js``. Returns ``None`` if not found.
    """
    import re
    import zipfile

    pat = re.compile(r"const\s+BG_IMAGE_BOUNDS\s*=\s*(\[[^\]]+\])\s*;")
    with zipfile.ZipFile(arcs_path) as z:
        for name in z.namelist():
            if not (name.endswith('.html') or name.endswith('.js')):
                continue
            try:
                txt = z.read(name).decode('utf-8', errors='ignore')
            except KeyError:
                continue
            m = pat.search(txt)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    continue
    return None


def inject_view_state(arcs_path: Path, state: dict, *, force: bool) -> bool:
    """Write ``state`` as ``view_state.json`` into the bundle."""
    # Reuse the existing helper rather than re-implementing the atomic
    # ZIP rewrite.
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from arcs_view_state import read_view_state, write_view_state

    existing = read_view_state(arcs_path)
    if existing is not None and not force:
        raise RuntimeError(
            f"{arcs_path} already has a view_state.json; "
            f"pass --force to overwrite"
        )
    return write_view_state(arcs_path, state)


# ---------------------------------------------------------------------------
# Diagnostic report
# ---------------------------------------------------------------------------
def save_report_png(
    out_path: Path,
    he_path: str,
    he_centroids_px: np.ndarray,
    gj_centroids_um: np.ndarray,
    sim: Similarity,
    info: dict,
) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree

    img = _read_he_rgb(he_path)
    warped = sim.apply(he_centroids_px)
    tree = cKDTree(gj_centroids_um)
    dists, _ = tree.query(warped, k=1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0, 0].imshow(img)
    axes[0, 0].scatter(he_centroids_px[:, 0], he_centroids_px[:, 1],
                       s=2, c='red', alpha=0.6)
    axes[0, 0].set_title(f"HE + detected nuclei (n={len(he_centroids_px)})")
    axes[0, 0].axis('off')

    axes[0, 1].scatter(gj_centroids_um[:, 0], gj_centroids_um[:, 1],
                       s=2, c='green', alpha=0.5, label='GeoJSON (μm)')
    axes[0, 1].set_aspect('equal')
    axes[0, 1].invert_yaxis()
    axes[0, 1].legend(loc='lower right')
    axes[0, 1].set_title(f"GeoJSON nuclei (n={len(gj_centroids_um)})")

    axes[1, 0].scatter(gj_centroids_um[:, 0], gj_centroids_um[:, 1],
                       s=2, c='green', alpha=0.5, label='GeoJSON')
    axes[1, 0].scatter(warped[:, 0], warped[:, 1],
                       s=2, c='red', alpha=0.5, label='HE → world')
    axes[1, 0].set_aspect('equal')
    axes[1, 0].invert_yaxis()
    axes[1, 0].legend(loc='lower right')
    axes[1, 0].set_title(
        f"Overlay after registration  "
        f"(scale={sim.scale:.4f}, θ={np.rad2deg(sim.rotation_rad):.2f}°)"
    )

    axes[1, 1].hist(dists, bins=50, color='steelblue')
    axes[1, 1].set_xlabel("Nearest-neighbor residual (μm)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title(
        f"Residuals (mean={info.get('mean_residual', float('nan')):.2f}, "
        f"median={info.get('median_residual', float('nan')):.2f}, "
        f"inliers={info.get('inliers', 0)})"
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
DEFAULT_INIT_ROTATIONS_DEG = [float(d) for d in range(0, 360, 30)]


def _parse_init_rotations(s: str) -> Sequence[float]:
    if not s or s == 'default':
        return list(DEFAULT_INIT_ROTATIONS_DEG)
    if s == 'cardinal':
        return [0.0, 90.0, 180.0, 270.0]
    if s == 'fine':
        return [float(d) for d in range(0, 360, 15)]
    return [float(x) for x in s.split(',') if x.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Register an HE image to a nuclei GeoJSON via "
                    "StarDist + ICP, and emit a ArcScape view_state.json."
    )
    ap.add_argument('--he', required=True,
                    help="Path to HE image (jpg/png/tif)")
    ap.add_argument('--geojson', required=True,
                    help="Path to nuclei GeoJSON (μm, Y-flipped)")

    g_out = ap.add_argument_group("Output")
    g_out.add_argument('--output-view-state', type=Path, default=None,
                       help="Write the resulting view_state.json here.")
    g_out.add_argument('--inject-arcs', type=Path, default=None,
                       help="Write the view_state into this .arcs bundle. "
                            "Reads the bundle's BG_IMAGE_BOUNDS so the "
                            "transform composes correctly with the baked "
                            "bounds.")
    g_out.add_argument('--force', action='store_true',
                       help="Overwrite an existing view_state.json in the "
                            "bundle (only meaningful with --inject-arcs).")
    g_out.add_argument('--report', type=Path, default=None,
                       help="Save a diagnostic PNG (matplotlib required).")
    g_out.add_argument('--report-json', type=Path, default=None,
                       help="Save the registration report as JSON.")

    g_det = ap.add_argument_group("Nucleus detection")
    g_det.add_argument('--method', choices=['stardist', 'classical'],
                       default='stardist',
                       help="Backend used to detect HE nuclei (default: stardist).")
    g_det.add_argument('--stardist-model', default='2D_versatile_he')
    g_det.add_argument('--stardist-model-dir', default=None,
                       help="Local directory containing the StarDist model "
                            "(skip pretrained download).")
    g_det.add_argument('--stardist-prob-thresh', type=float, default=None)
    g_det.add_argument('--stardist-nms-thresh', type=float, default=None)
    g_det.add_argument('--stardist-n-tiles', type=int, nargs=2, default=None,
                       metavar=('NY', 'NX'),
                       help="Tile factor for large images (default: auto).")
    g_det.add_argument('--classical-smooth-sigma', type=float, default=1.0)
    g_det.add_argument('--classical-min-distance', type=int, default=4)
    g_det.add_argument('--classical-hematoxylin-percentile', type=float,
                       default=70.0)

    g_reg = ap.add_argument_group("Registration")
    g_reg.add_argument('--init-rotations-deg', type=_parse_init_rotations,
                       default='default',
                       help="Initial rotations (deg) for the multi-start "
                            "ICP search. Accepts a comma-separated list "
                            "(e.g. '0,30,60'), or one of the presets "
                            "'default' (every 30°, 12 starts), 'cardinal' "
                            "(0,90,180,270), 'fine' (every 15°). "
                            "Pass 'cardinal' for fast runs when HE & "
                            "segmentation share orientation.")
    g_reg.add_argument('--max-iter', type=int, default=100)
    g_reg.add_argument('--tol', type=float, default=1e-4)
    g_reg.add_argument('--trim-quantile', type=float, default=0.8,
                       help="Fraction of nearest-neighbor pairs to keep "
                            "each ICP iteration (0–1, default 0.8).")

    g_pl = ap.add_argument_group("Background appearance defaults")
    g_pl.add_argument('--bg-opacity', type=int, default=200)
    g_pl.add_argument('--bg-depth', type=float, default=0.0)
    g_pl.add_argument('--bg-hidden', action='store_true',
                      help="Set bgVisible=false in the emitted state.")

    args = ap.parse_args(argv)

    if (args.output_view_state is None
            and args.inject_arcs is None
            and args.report_json is None):
        ap.error("Nothing to do: pass at least one of "
                 "--output-view-state / --inject-arcs / --report-json")

    he_path = args.he
    geojson_path = args.geojson

    # 1) Image size
    from PIL import Image
    with Image.open(he_path) as im:
        W_px, H_px = im.size

    # 2) Detect HE nuclei
    t0 = time.time()
    stardist_version: Optional[str] = None
    if args.method == 'stardist':
        n_tiles = tuple(args.stardist_n_tiles) if args.stardist_n_tiles else None
        he_pts, stardist_version = detect_nuclei_stardist(
            he_path,
            model_name=args.stardist_model,
            model_basedir=args.stardist_model_dir,
            prob_thresh=args.stardist_prob_thresh,
            nms_thresh=args.stardist_nms_thresh,
            n_tiles=n_tiles,
        )
    else:
        he_pts = detect_nuclei_classical(
            he_path,
            smooth_sigma=args.classical_smooth_sigma,
            min_distance=args.classical_min_distance,
            hematoxylin_percentile=args.classical_hematoxylin_percentile,
        )
    print(f"  HE nuclei (method={args.method}): {len(he_pts)} "
          f"({time.time() - t0:.1f}s)")

    # 3) Load GeoJSON centroids
    gj_pts = load_geojson_centroids(geojson_path)
    print(f"  GeoJSON nuclei: {len(gj_pts)}")

    # 4) ICP
    sim, info, init_deg = best_of_initial_rotations(
        he_pts, gj_pts,
        rotations_deg=args.init_rotations_deg,
        icp_kwargs=dict(
            max_iter=args.max_iter,
            tol=args.tol,
            trim_quantile=args.trim_quantile,
        ),
    )
    print(f"  ICP: init_rot={init_deg:.0f}°, "
          f"iters={info.get('iterations')}, "
          f"inliers={info.get('inliers')}, "
          f"mean_residual={info.get('mean_residual'):.2f} μm, "
          f"median_residual={info.get('median_residual'):.2f} μm")
    print(f"  similarity: scale={sim.scale:.4f}, "
          f"rotation={np.rad2deg(sim.rotation_rad):.2f}°, "
          f"translation=({sim.translation[0]:.2f}, {sim.translation[1]:.2f}) μm")

    # 5) Build report dataclass
    report = RegistrationReport(
        method=args.method,
        he_path=str(he_path),
        geojson_path=str(geojson_path),
        n_he_centroids=int(len(he_pts)),
        n_gj_centroids=int(len(gj_pts)),
        init_rotation_deg=float(init_deg),
        iterations=int(info.get('iterations', 0)),
        final_mean_residual=float(info.get('mean_residual', float('nan'))),
        final_median_residual=float(info.get('median_residual', float('nan'))),
        final_inliers=int(info.get('inliers', 0)),
        similarity=sim,
        image_size_px=(int(W_px), int(H_px)),
        stardist_model=args.stardist_model if args.method == 'stardist' else None,
        stardist_version=stardist_version,
    )

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"  Report JSON: {args.report_json}")

    # 6) Compose view_state for output / injection
    bg_visible = not args.bg_hidden

    if args.output_view_state is not None:
        state = similarity_to_view_state(
            sim, (W_px, H_px),
            base_image_bounds=None,
            bg_opacity=args.bg_opacity,
            bg_depth=args.bg_depth,
            bg_visible=bg_visible,
        )
        args.output_view_state.parent.mkdir(parents=True, exist_ok=True)
        args.output_view_state.write_text(json.dumps(state, indent=2))
        print(f"  view_state.json (standalone): {args.output_view_state}")

    if args.inject_arcs is not None:
        base_bounds = read_image_bounds_from_arcs(args.inject_arcs)
        if base_bounds is None:
            print("  WARNING: could not read BG_IMAGE_BOUNDS from "
                  f"{args.inject_arcs}; injection skipped.", file=sys.stderr)
        else:
            state = similarity_to_view_state(
                sim, (W_px, H_px),
                base_image_bounds=base_bounds,
                bg_opacity=args.bg_opacity,
                bg_depth=args.bg_depth,
                bg_visible=bg_visible,
            )
            changed = inject_view_state(args.inject_arcs, state,
                                        force=args.force)
            verb = "wrote" if changed else "no-op (identical)"
            print(f"  {args.inject_arcs}: {verb} view_state.json")

    # 7) Diagnostic PNG
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        save_report_png(args.report, he_path, he_pts, gj_pts, sim, info)
        print(f"  Report PNG: {args.report}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
