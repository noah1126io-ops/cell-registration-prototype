"""Sample loading with explicit availability: each modality has nuclei (required) and an image (optional).

Image *availability* is decided by what the experiment is allowed to see (hide_images), not by what is
on disk, so E0/E1/E2 can run on identical point sets.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from .frames import Frame

PROJECTS = Path("/mnt/mib-dbia08_1/users/noa.ganeko/projects")
HE_NUCLEI_DIR = PROJECTS / "cell-registration-prototype-gpu/analysis/he-stardist-seqif-registration-20261019/out"
SAMPLES = ("031-6_2", "031-8_6")


def ome_info(path):
    with tifffile.TiffFile(path) as t:
        s = t.series[0]
        om = t.ome_metadata or ""
        px = re.search(r'PhysicalSizeX="([^"]+)"', om)
        py = re.search(r'PhysicalSizeY="([^"]+)"', om)
        desc = re.search(r"<Description>(.*?)</Description>", om, re.S)
        h0, w0 = s.levels[0].shape[:2]
        return {"path": str(path), "shape": list(s.levels[0].shape), "axes": s.axes, "dtype": str(s.dtype),
                "levels": [list(l.shape) for l in s.levels],
                "downsamples": [int(round(w0 / l.shape[1])) for l in s.levels],
                "pixel_x_um": float(px.group(1)) if px else None, "pixel_y_um": float(py.group(1)) if py else None,
                "description": desc.group(1) if desc else None,
                "has_stage_origin": bool(re.search(r"<Plane[^>]*PositionX", om)),
                "channel_names": re.findall(r'<Channel [^>]*Name="([^"]+)"', om)}


def read_level(path, k):
    with tifffile.TiffFile(path) as t:
        return t.series[0].levels[k].asarray()


@dataclass
class Modality:
    frame: Frame
    nuclei_um: np.ndarray            # (n, 2) frame world um
    nuclei_ids: np.ndarray
    nuclei_source: str
    image_on_disk: bool
    image_visible: bool
    polygons_px: list | None = None   # IF only (level-0 px)
    extra: dict = field(default_factory=dict)


@dataclass
class Sample:
    name: str
    he: Modality
    iff: Modality
    notes: dict

    def mode(self):
        n = int(self.he.image_visible) + int(self.iff.image_visible)
        return {0: "P0", 1: "P1", 2: "P2"}[n]

    def modality(self, key):
        return {"HE": self.he, "IF": self.iff}[key]


def load_sample(name, *, hide_images=(), he_min_prob=0.0):
    d = PROJECTS / name
    he_tif = next(d.glob(f"{name}_HE*.ome.tif"))
    if_tif = d / f"{name}_seqif.ome.tif"
    hi, ii = ome_info(he_tif), ome_info(if_tif)
    he_frame = Frame(f"{name}:HE", "HE", hi["pixel_x_um"], tuple(hi["shape"][:2]), image_path=str(he_tif),
                     pyramid_downsamples=tuple(hi["downsamples"]))
    if_frame = Frame(f"{name}:IF", "IF", ii["pixel_x_um"], tuple(ii["shape"][:2]), image_path=str(if_tif),
                     pyramid_downsamples=tuple(ii["downsamples"]))
    nuc = pd.read_csv(HE_NUCLEI_DIR / name / "he_nuclei.csv")
    nuc = nuc[nuc.prob >= he_min_prob].reset_index(drop=True)
    he_pts = he_frame.pixel_to_world(nuc[["centroid_x_px", "centroid_y_px"]].to_numpy())
    gj = json.loads((d / f"{name}_seg_seqif_level0.geojson").read_text())
    ids, cen, polys = [], [], []
    for f in gj["features"]:
        p = f["properties"]
        ids.append(p["segment_id"]); cen.append((p["centroid_x"], p["centroid_y"]))
        polys.append(np.asarray(f["geometry"]["coordinates"][0], float))
    if_pts = if_frame.pixel_to_world(np.asarray(cen, float))
    he = Modality(he_frame, he_pts, nuc.point_id.to_numpy(), "StarDist 2D_versatile_he (generated 2026-10-19, analysis he-stardist-seqif-registration-20261019)",
                  True, "HE" not in hide_images, extra={"prob": nuc.prob.to_numpy(), "area_um2": nuc.area_um2.to_numpy()})
    iff = Modality(if_frame, if_pts, np.asarray(ids), "provider GeoJSON *_seg_seqif_level0.geojson (fluorescence segmentation)",
                   True, "IF" not in hide_images, polygons_px=polys)
    notes = {"he_description": hi["description"], "if_description": ii["description"]}
    return Sample(name, he, iff, notes)
