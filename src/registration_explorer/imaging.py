"""Pyramid OME-TIFF access for the Registration Explorer (research mode).

* Level selection by target um/px; level 0 is never read whole: regions are assembled from the tiles they touch.
* Display conversions keep the modality's pixel meaning explicit (brightfield RGB vs fluorescence display RGB).
* Warp preview resamples with the INVERSE map (target pixel -> source pixel); points use the forward map elsewhere.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

try:
    import tifffile
except ImportError:  # pragma: no cover - tifffile is in requirements.txt
    tifffile = None

MAX_FULL_LEVEL_PIXELS = 4096 * 4096      # whole-level reads above this are refused (use read_region)


class ImageUnavailable(RuntimeError):
    """The image cannot be used (missing file, unreadable, unknown spacing...)."""


@dataclass(frozen=True)
class PyramidInfo:
    path: str
    level_shapes: tuple          # ((h, w, c), ...) level 0 first
    downsamples: tuple           # per level, relative to level 0 (x)
    tiled: bool
    tile: tuple                  # (th, tw) or (0, 0)
    dtype: str


@lru_cache(maxsize=16)
def pyramid_info(path: str) -> PyramidInfo:
    if tifffile is None:
        raise ImageUnavailable("tifffile is not installed")
    p = Path(path)
    if not p.is_file():
        raise ImageUnavailable(f"image file not found: {path}")
    try:
        with tifffile.TiffFile(p) as t:
            s = t.series[0]
            shapes = tuple(tuple(int(v) for v in lv.shape) for lv in s.levels)
            kf = s.levels[0].keyframe
            w0 = shapes[0][1]
            return PyramidInfo(str(p), shapes, tuple(w0 / sh[1] for sh in shapes), bool(kf.is_tiled),
                               (int(kf.tilelength), int(kf.tilewidth)) if kf.is_tiled else (0, 0), str(s.dtype))
    except ImageUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - any read error makes the image unusable, never fatal
        raise ImageUnavailable(f"cannot read pyramid: {exc}") from exc


def pick_level(info: PyramidInfo, level0_um: float, target_um: float) -> int:
    """Coarsest level whose pixel size is still <= target_um (falls back to the finest)."""
    best = 0
    for k, ds in enumerate(info.downsamples):
        if level0_um * ds <= target_um * 1.0001:
            best = k
    return best


def read_region(path: str, level: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Pixels [y0:y1, x0:x1] of a pyramid level, decoding only the tiles that intersect it."""
    info = pyramid_info(path)
    h, w = info.level_shapes[level][:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("empty region")
    with tifffile.TiffFile(path) as t:
        page = t.series[0].levels[level].keyframe
        if not page.is_tiled:
            if h * w > MAX_FULL_LEVEL_PIXELS:
                raise ImageUnavailable("untiled large level; region read refused")
            return t.series[0].levels[level].asarray()[y0:y1, x0:x1]
        th, tw = page.tilelength, page.tilewidth
        ntx = math.ceil(w / tw)
        nchan = info.level_shapes[level][2] if len(info.level_shapes[level]) > 2 else 1
        out = np.zeros((y1 - y0, x1 - x0, nchan), dtype=page.dtype)
        fh = t.filehandle
        jpegtables = getattr(page, "jpegtables", None)
        for ty in range(y0 // th, (y1 - 1) // th + 1):
            for tx in range(x0 // tw, (x1 - 1) // tw + 1):
                idx = ty * ntx + tx
                fh.seek(page.dataoffsets[idx])
                data = fh.read(page.databytecounts[idx])
                tile, _, _ = page.decode(data, idx, jpegtables=jpegtables)
                tile = np.asarray(tile).reshape(th, tw, -1)
                ty0, tx0 = ty * th, tx * tw
                sy0, sy1 = max(y0, ty0), min(y1, ty0 + th)
                sx0, sx1 = max(x0, tx0), min(x1, tx0 + tw)
                out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = tile[sy0 - ty0:sy1 - ty0, sx0 - tx0:sx1 - tx0, :nchan]
    return out if nchan > 1 else out[..., 0]


@lru_cache(maxsize=24)
def read_level_cached(path: str, level: int) -> np.ndarray:
    """A whole (small) pyramid level; refuses levels larger than MAX_FULL_LEVEL_PIXELS (e.g. level 0)."""
    info = pyramid_info(path)
    h, w = info.level_shapes[level][:2]
    if h * w > MAX_FULL_LEVEL_PIXELS:
        raise ImageUnavailable(f"level {level} ({w}x{h}) too large for a whole read; use read_region")
    return read_region(path, level, 0, 0, w, h)


# ----------------------------------------------------------------------------- display
def to_display_rgb(img: np.ndarray, kind: str, mode: str = "rgb") -> np.ndarray:
    """uint8 RGB for display only (never used for computation).

    kind: brightfield_rgb (H&E) | fluorescence_display_rgb (provider-rendered IF)
    mode: rgb | nuclear  (H&E: inverted gray ~ hematoxylin/OD; IF: B-R Hoechst estimate in cyan)
    """
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, -1)
    a = a[..., :3].astype(np.float32)
    if mode == "rgb":
        return np.clip(a, 0, 255).astype(np.uint8)
    if kind == "brightfield_rgb":
        g = 255.0 - a.mean(-1)
        g = np.clip((g - np.percentile(g, 1)) / max(np.percentile(g, 99.5) - np.percentile(g, 1), 1e-6), 0, 1) * 255
        return np.repeat(g[..., None], 3, -1).astype(np.uint8)
    n = np.clip(a[..., 2] - a[..., 0], 0, None)
    n = np.clip(n / max(np.percentile(n, 99.5), 1e-6), 0, 1)
    return (np.stack([0 * n, n, n], -1) * 255).astype(np.uint8)


def encode_jpeg(rgb: np.ndarray, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb, np.uint8)).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def encode_png(rgba: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgba, np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


# ----------------------------------------------------------------------------- warp preview
def level_corner_um(level0_um: float) -> float:
    """World coordinate of the top-left image corner.

    Convention (same as the nuclear coordinates): level-0 pixel centres sit at integer indices and
    world = index * level0_um, so the image corner is at -0.5 * level0_um.  A level-k pixel of size
    um_k has index u = (world - corner) / um_k - 0.5.
    """
    return -0.5 * float(level0_um)


def warp_preview(src_rgb: np.ndarray, src_um_per_px: float, src_corner_um: float, M_src_to_dst: np.ndarray,
                 t_src_to_dst: np.ndarray, dst_corner_um: float, dst_extent_um: tuple, dst_um_per_px: float, order: int = 1):
    """Resample a source-frame image into the target frame with the INVERSE map.

    dst_world = M @ src_world + t.  Each target pixel centre p (world = dst_corner + (j + 0.5) * dst_um) is
    pulled from source location M^-1 (p - t), i.e. source index (world - src_corner) / src_um - 0.5.
    Outside the source image -> 0 and marked invalid.  Returns (image uint8, valid mask, resampling record).
    """
    W, H = int(math.ceil(dst_extent_um[0] / dst_um_per_px)), int(math.ceil(dst_extent_um[1] / dst_um_per_px))
    Mi = np.linalg.inv(np.asarray(M_src_to_dst, float))
    ti = -Mi @ np.asarray(t_src_to_dst, float)
    jj, ii = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    dst_w = np.stack([dst_corner_um + jj * dst_um_per_px, dst_corner_um + ii * dst_um_per_px], -1).reshape(-1, 2)
    src_w = dst_w @ Mi.T + ti
    sx = (src_w[:, 0] - src_corner_um) / src_um_per_px - 0.5
    sy = (src_w[:, 1] - src_corner_um) / src_um_per_px - 0.5
    h, w = src_rgb.shape[:2]
    valid = (sx >= -0.5) & (sx <= w - 0.5) & (sy >= -0.5) & (sy <= h - 0.5)
    chans = src_rgb if src_rgb.ndim == 3 else src_rgb[..., None]
    out = np.zeros((H * W, chans.shape[2]), np.float32)
    for c in range(chans.shape[2]):
        out[:, c] = map_coordinates(chans[..., c].astype(np.float32), [sy, sx], order=order, mode="constant", cval=0.0)
    out[~valid] = 0
    rgb = np.clip(out.reshape(H, W, -1), 0, 255).astype(np.uint8)
    record = {"mapping": "inverse (target pixel centre -> source pixel)", "interpolation": {0: "nearest", 1: "bilinear", 3: "cubic"}[order],
              "outside": "constant 0, marked invalid (no extrapolation)", "target_um_per_px": dst_um_per_px,
              "source_um_per_px": src_um_per_px, "target_shape": [H, W], "target_corner_um": dst_corner_um,
              "pixel_centre_convention": "level-0 pixel centres at integer index, world = index * level0_um",
              "origin": "frame-local (0,0); stage origin UNKNOWN, not inferred"}
    return rgb if src_rgb.ndim == 3 else rgb[..., 0], valid.reshape(H, W), record
