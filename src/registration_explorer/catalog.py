"""Sample registry for the Registration Explorer (research mode).

The registry is an explicit JSON manifest (``viewer_sample_manifest.json``).  Nothing is guessed: a missing
value stays ``None`` and is shown as UNKNOWN.  Coordinates are frame-local world units:
world = level-0 pixel index * pixel_um (x right, y down); when pixel_um is UNKNOWN the frame stays in pixels.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

UNKNOWN = "UNKNOWN"
REPO = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO / "analysis/e0x-difficulty-and-web-integration-20261020/viewer_sample_manifest.json"


def manifest_path() -> Path:
    return Path(os.environ.get("CELLREG_EXPLORER_MANIFEST", str(DEFAULT_MANIFEST)))


@dataclass
class ModalityInfo:
    key: str                         # HE | IF
    label: str
    kind: str                        # brightfield_rgb | fluorescence_display_rgb | points_only
    pixel_um: float | None
    shape_px: tuple | None           # (h, w) level 0
    image_path: str | None
    nuclei: dict                     # {"format": ..., "path": ..., ...}
    channels: str | None = None
    origin: str = f"{UNKNOWN} (frame-local (0,0) used)"
    extent_override: tuple | None = None   # (w, h) in frame units when there is no image (e.g. production crop size)
    tissue_npz: str | None = None    # cached research feature maps (tissue_bin / envelope), optional

    @property
    def units(self) -> str:
        """Display unit of this frame's world coordinates."""
        if self.pixel_um is not None:
            return "µm"
        return {"um": "µm", "px": "px"}.get((self.nuclei or {}).get("world_units", ""), "UNKNOWN units")

    def extent(self):
        """Imaged extent (w, h) in frame units from image size x spacing; None if UNKNOWN (never the point bbox)."""
        if self.shape_px is not None:
            h, w = self.shape_px
            s = self.pixel_um if self.pixel_um is not None else 1.0
            return (w * s, h * s)
        return tuple(self.extent_override) if self.extent_override else None

    def image_on_disk(self) -> bool:
        return bool(self.image_path) and Path(self.image_path).is_file()


@dataclass
class SampleInfo:
    sample_id: str
    label: str
    modalities: dict                 # key -> ModalityInfo
    provenance: dict
    coordinate_system: str
    result_sources: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def modality(self, key) -> ModalityInfo:
        return self.modalities[key]

    def other(self, key) -> str:
        return "IF" if key == "HE" else "HE"


def _mod(key, d) -> ModalityInfo:
    return ModalityInfo(key=key, label=d.get("label", key), kind=d.get("kind", "points_only"), pixel_um=d.get("pixel_um"),
                        shape_px=tuple(d["shape_px"]) if d.get("shape_px") else None, image_path=d.get("image_path"),
                        nuclei=d.get("nuclei", {}), channels=d.get("channels"), origin=d.get("origin", f"{UNKNOWN} (frame-local (0,0) used)"),
                        extent_override=tuple(d["extent_override"]) if d.get("extent_override") else None, tissue_npz=d.get("tissue_npz"))


def load_manifest(path: Path | None = None) -> dict[str, SampleInfo]:
    path = Path(path or manifest_path())
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for s in data["samples"]:
        out[s["sample_id"]] = SampleInfo(s["sample_id"], s.get("label", s["sample_id"]),
                                         {k: _mod(k, v) for k, v in s["modalities"].items()}, s.get("provenance", {}),
                                         s.get("coordinate_system", UNKNOWN), s.get("result_sources", []), s.get("notes", []))
    return out


# ----------------------------------------------------------------------------- nuclei
@lru_cache(maxsize=32)
def _load_nuclei_cached(fmt: str, path: str, x: str, y: str, scale: float, mtime: float):
    p = Path(path)
    if fmt == "csv_px":
        import pandas as pd
        df = pd.read_csv(p)
        return df[[x, y]].to_numpy(float) * scale, df.index.to_numpy()
    if fmt == "geojson_px_props":
        gj = json.loads(p.read_text())
        pts, ids = [], []
        for f in gj["features"]:
            pr = f["properties"]
            pts.append((pr[x], pr[y])); ids.append(pr.get("segment_id", len(ids)))
        return np.asarray(pts, float) * scale, np.asarray(ids)
    if fmt == "npy":
        a = np.load(p)
        return a[:, :2].astype(float) * scale, np.arange(len(a))
    raise ValueError(f"unknown nuclei format {fmt}")


def load_nuclei(mod: ModalityInfo):
    """Nuclei in frame world units (um if spacing known, else px)."""
    n = mod.nuclei
    if not n or not n.get("path") or not Path(n["path"]).is_file():
        raise FileNotFoundError(f"{mod.key}: nuclei file missing ({n.get('path') if n else None})")
    scale = mod.pixel_um if (n.get("units", "px") == "px" and mod.pixel_um is not None) else 1.0
    return _load_nuclei_cached(n["format"], n["path"], n.get("x", ""), n.get("y", ""), float(scale), Path(n["path"]).stat().st_mtime)


def availability(sample: SampleInfo, *, hide_images=False, simulate_image_failure=False) -> dict:
    """What the explorer may use for this sample (images can be hidden for point-only checks)."""
    out = {}
    for k, m in sample.modalities.items():
        nuc_ok = bool(m.nuclei) and bool(m.nuclei.get("path")) and Path(m.nuclei["path"]).is_file()
        img_reason = None
        if hide_images:
            img_reason = "hidden (point-only view)"
        elif simulate_image_failure:
            img_reason = "simulated image read failure (test)"
        elif not m.image_path:
            img_reason = "no image registered"
        elif not Path(m.image_path).is_file():
            img_reason = f"image file not found: {m.image_path}"
        elif m.pixel_um is None:
            img_reason = "pixel spacing UNKNOWN"
        out[k] = {"nuclei": nuc_ok, "image": img_reason is None, "image_reason": img_reason}
    return out


def side_input(sample: SampleInfo, key: str, *, use_image: bool = True):
    """Adapter: one modality of a manifest sample -> registration_core.SideInput (the engine input)."""
    from ..registration_core import SideInput
    m = sample.modality(key)
    pts, _ = load_nuclei(m)
    img = m.image_path if (use_image and m.image_on_disk() and m.pixel_um is not None) else None
    ds = ()
    if img:
        from .imaging import ImageUnavailable, pyramid_info
        try:
            ds = tuple(int(round(d)) for d in pyramid_info(img).downsamples)
        except ImageUnavailable:
            img = None
    return SideInput(key=key, points=np.asarray(pts, float), extent=m.extent() if m.pixel_um is not None or m.extent_override else None,
                     frame_name=f"{sample.sample_id}:{key}", pixel_um=m.pixel_um, image_path=img,
                     image_kind=m.kind if img else "points_only", shape_px=m.shape_px, pyramid_downsamples=ds,
                     nuclei_source=(m.nuclei or {}).get("source", ""))
