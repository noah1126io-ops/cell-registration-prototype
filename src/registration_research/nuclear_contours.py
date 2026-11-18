"""Nuclear contours from the existing instance segmentation (research, read-only; 2026-11-18).

The nuclei used everywhere (``catalog.load_nuclei``) are centre points.  Their outlines already exist in the segmentation
GeoJSON files that produced those points; this module joins the two by the original nucleus ID and checks that they
really describe the same nucleus.  No segmentation is (re)run.

Discovery (``discover``), in this order:
  1. ``nuclei["contours"] = {"path": ..., "id_property": ...}`` in the sample manifest (optional override);
  2. nuclei format ``geojson_px_props``: polygons of the same file, joined by ``segment_id`` (the ID load_nuclei returns);
  3. nuclei format ``csv_px``: ``<same stem>.geojson`` next to the CSV, joined by ``point_id`` (= the CSV row, the ID
     load_nuclei returns) -- the StarDist output of analysis/he-stardist-seqif-registration-20261019;
  4. otherwise no contours (points-only review still works).

Coordinates: polygon vertices are level-0 pixel positions like the centroids, converted with the same scale as
``load_nuclei`` (world = px x pixel_um).

Per-nucleus check (``status``):
  ok                contour found, the centre point lies inside it, and |area centroid - point| <= MAX_REL_OFFSET x r_eq
  missing           no contour with this ID
  duplicate_id      the ID occurs more than once in the GeoJSON (ambiguous -> not used)
  centroid_outside  the centre point is outside its contour (ID join or frame wrong)
  offset_large      inside, but far from the area centroid (r_eq = sqrt(area / pi))
A file-level ``frame_mismatch`` is reported when fewer than half of the joined contours contain their point
(e.g. contours of another pyramid level).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAX_REL_OFFSET = 1.0          # real data (031-6_2, 031-8_6, H&E StarDist and IF provider): max 0.63
STATUS_OK = "ok"
STATUSES = ("ok", "missing", "duplicate_id", "centroid_outside", "offset_large")
_CACHE: dict = {}


def _key(v):
    """IDs compared as text of the number when numeric (segment_id 4.0 == 4), else as text."""
    v = v.item() if hasattr(v, "item") else v
    try:
        f = float(v)
        return repr(int(f)) if f.is_integer() else repr(f)
    except (TypeError, ValueError):
        return str(v)


def discover(mod) -> dict | None:
    """{"path", "id_property", "how"} or None.  ``mod`` is a catalog.ModalityInfo."""
    n = mod.nuclei or {}
    c = n.get("contours")
    if c and c.get("path") and Path(c["path"]).is_file():
        return {"path": c["path"], "id_property": c.get("id_property", "segment_id"), "how": "manifest nuclei.contours"}
    p = Path(n.get("path") or "")
    if n.get("format") == "geojson_px_props" and p.is_file():
        return {"path": str(p), "id_property": "segment_id", "how": "polygons of the nuclei GeoJSON (segment_id)"}
    if n.get("format") == "csv_px":
        g = p.with_suffix(".geojson")
        if g.is_file():
            return {"path": str(g), "id_property": "point_id", "how": "sibling GeoJSON of the nuclei CSV (point_id = CSV row)"}
    return None


def _inside(poly: np.ndarray, q: np.ndarray) -> bool:
    x, y = q
    xs, ys = poly[:, 0], poly[:, 1]
    xj, yj = np.roll(xs, 1), np.roll(ys, 1)
    cross = ((ys > y) != (yj > y)) & (x < (xj - xs) * (y - ys) / np.where(yj - ys == 0, 1e-30, yj - ys) + xs)
    return bool(cross.sum() % 2)


def _area_centroid(poly: np.ndarray):
    x, y = poly[:, 0], poly[:, 1]
    a = x * np.roll(y, -1) - np.roll(x, -1) * y
    A = a.sum() / 2.0
    if abs(A) < 1e-12:
        return poly.mean(0), 0.0
    return np.array([((x + np.roll(x, -1)) * a).sum() / (6 * A), ((y + np.roll(y, -1)) * a).sum() / (6 * A)]), abs(A)


@dataclass
class Contours:
    """Outlines aligned to the nuclei array of one section (index i = nucleus i of load_nuclei)."""
    key: str
    source: dict                      # discover() + sha256 of the file
    offsets: np.ndarray               # (n + 1,) vertex offsets; nucleus i -> xy[offsets[i]:offsets[i+1]] (empty if none)
    xy: np.ndarray                    # (N, 2) world units of the section frame
    status: np.ndarray                # (n,) one of STATUSES
    rel_offset: np.ndarray            # (n,) |area centroid - point| / r_eq (nan if no contour)
    report: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.status)

    def has(self, i) -> bool:
        return self.offsets[i + 1] > self.offsets[i]

    def polygon(self, i):
        a, b = self.offsets[i], self.offsets[i + 1]
        return self.xy[a:b] if b > a else None

    def bboxes(self) -> np.ndarray:
        if "_bb" not in self.report:
            bb = np.full((self.n, 4), np.nan)
            for i in range(self.n):
                p = self.polygon(i)
                if p is not None:
                    bb[i] = [p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]
            self.report["_bb"] = bb
        return self.report["_bb"]

    def query(self, rect, M=None, t=None) -> np.ndarray:
        """Nucleus indices whose contour intersects rect = [x0, y0, x1, y1] (rect in the frame after the optional affine)."""
        bb = self.bboxes()
        if M is not None:
            M = np.asarray(M, float); t = np.asarray(t, float)
            c = np.stack([bb[:, [0, 1]], bb[:, [2, 1]], bb[:, [2, 3]], bb[:, [0, 3]]], 1) @ M.T + t
            bb = np.concatenate([np.nanmin(c, 1), np.nanmax(c, 1)], 1)
        ok = np.isfinite(bb[:, 0])
        hit = ok & (bb[:, 2] >= rect[0]) & (bb[:, 0] <= rect[2]) & (bb[:, 3] >= rect[1]) & (bb[:, 1] <= rect[3])
        return np.flatnonzero(hit)

    def pack(self, indices, M=None, t=None, decimals=2) -> dict:
        """{"idx": [...], "poly": [[x0, y0, x1, y1, ...], ...]} for the given nuclei that have a usable contour."""
        idx, polys = [], []
        for i in np.asarray(indices, int).ravel():
            p = self.polygon(i)
            if p is None or self.status[i] in ("duplicate_id",):
                continue
            if M is not None:
                p = p @ np.asarray(M, float).T + np.asarray(t, float)
            idx.append(int(i)); polys.append(np.round(p, decimals).ravel().tolist())
        return {"idx": idx, "poly": polys}

    def summary(self) -> dict:
        return {k: v for k, v in self.report.items() if not k.startswith("_")}


def load_contours(mod, pts, ids, scale: float | None = None) -> Contours | None:
    """Join the section's segmentation polygons to its nuclei (pts, ids from catalog.load_nuclei); None if none exist."""
    src = discover(mod)
    if src is None:
        return None
    n = mod.nuclei or {}
    if scale is None:
        scale = mod.pixel_um if (n.get("units", "px") == "px" and mod.pixel_um is not None) else 1.0
    path = Path(src["path"])
    pts = np.asarray(pts, float)
    ck = (str(path), path.stat().st_mtime, float(scale), src["id_property"], hashlib.sha1(pts.tobytes()).hexdigest(),
          hashlib.sha1("|".join(_key(v) for v in ids).encode()).hexdigest())
    if ck in _CACHE:
        return _CACHE[ck]
    raw = path.read_bytes()
    feats = json.loads(raw)["features"]
    by_id, dup, n_bad_geom = {}, set(), 0
    for f in feats:
        g = f.get("geometry") or {}
        if g.get("type") == "Polygon" and g.get("coordinates"):
            ring = np.asarray(g["coordinates"][0], float)
        elif g.get("type") == "MultiPolygon" and g.get("coordinates"):
            ring = np.asarray(max(g["coordinates"], key=lambda r: len(r[0]))[0], float)
        else:
            n_bad_geom += 1
            continue
        k = _key((f.get("properties") or {}).get(src["id_property"]))
        if k in by_id:
            dup.add(k)
        by_id[k] = ring * scale
    keys = [_key(v) for v in ids]
    offsets = np.zeros(len(pts) + 1, np.int64)
    chunks, status, rel = [], np.empty(len(pts), dtype=object), np.full(len(pts), np.nan)
    for i, k in enumerate(keys):
        ring = by_id.get(k)
        if ring is None:
            status[i] = "missing"
        elif k in dup:
            status[i] = "duplicate_id"
        else:
            if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
                ring = ring[:-1]
            g, A = _area_centroid(ring)
            r_eq = np.sqrt(A / np.pi) if A > 0 else np.nan
            rel[i] = float(np.linalg.norm(g - pts[i]) / r_eq) if r_eq > 0 else np.nan
            if not _inside(ring, pts[i]):
                status[i] = "centroid_outside"
            elif not (rel[i] <= MAX_REL_OFFSET):
                status[i] = "offset_large"
            else:
                status[i] = STATUS_OK
            chunks.append(ring)
        offsets[i + 1] = offsets[i] + (len(chunks[-1]) if status[i] not in ("missing", "duplicate_id") else 0)
    xy = np.concatenate(chunks).astype(np.float64) if chunks else np.zeros((0, 2))
    status = status.astype(str)
    counts = {s: int((status == s).sum()) for s in STATUSES}
    joined = len(pts) - counts["missing"] - counts["duplicate_id"]
    inside_frac = (joined - counts["centroid_outside"]) / joined if joined else float("nan")
    rep = {"section": mod.key, "n_nuclei": int(len(pts)), "n_features": len(feats), "n_bad_geometry": n_bad_geom,
           "n_orphan_contours": int(len(set(by_id) - set(keys))), "n_duplicate_ids_in_file": len(dup), "status_counts": counts,
           "inside_fraction": inside_frac, "rel_offset_max": float(np.nanmax(rel)) if np.isfinite(rel).any() else None,
           "frame_mismatch": bool(joined and inside_frac < 0.5), "max_rel_offset_threshold": MAX_REL_OFFSET}
    c = Contours(mod.key, {**src, "sha256": hashlib.sha256(raw).hexdigest(), "scale": float(scale)}, offsets, xy, status, rel, rep)
    _CACHE[ck] = c
    return c


def identifier(c: Contours | None) -> dict | None:
    """Stable identifier of the segmentation used (stored with every review)."""
    if c is None:
        return None
    return {"path": c.source["path"], "sha256": c.source["sha256"], "id_property": c.source["id_property"], "how": c.source["how"],
            "status_counts": c.report["status_counts"]}
