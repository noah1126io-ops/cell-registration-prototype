"""Shared loaders for the 2026-10-27 C1 audit: sides from the Explorer catalog, cached structural feature maps,
the user's manual reference transform (read-only from the research store), probe-distance helpers.

Transform convention (registration_core): fixed_world = M @ moving_world + t  (fixed = HE, moving = IF, um).
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from src.registration_explorer import catalog  # noqa: E402
from src.registration_core.features import FeatureMaps, Modality, extract  # noqa: E402
from src.registration_core.pipeline import Side  # noqa: E402

STORE = REPO / "runs/research-store/research.sqlite"
CACHE = HERE / "cache"
FIG = HERE / "figures"
REF_TF_ID = "tf-20261027064332-5467cd8f"          # user's manual adjustment of 031-8_6 (MANUAL_UNVERIFIED)


def side_input(sample_id, key):
    man = catalog.load_manifest()
    return catalog.side_input(man[sample_id], key)


def features(sample_id, key, res_um=2.0):
    """registration_core.features.extract on the real image, cached as npz (maps + quality)."""
    p = CACHE / f"feat_{sample_id}_{key}_{res_um:g}.npz"
    if p.exists():
        z = np.load(p, allow_pickle=True)
        maps = {k[4:]: z[k] for k in z.files if k.startswith("map_")}
        return FeatureMaps(str(z["frame"]), str(z["modality"]), float(z["res_um"]), maps, z["fov"], json.loads(str(z["quality"])),
                           float(z["runtime_s"]), int(z["source_level"]))
    si = side_input(sample_id, key)
    fm = extract(Modality(si.frame(), np.asarray(si.points, float)), res_um=res_um)
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(p, frame=fm.frame, modality=fm.modality, res_um=fm.res_um, fov=fm.fov, runtime_s=fm.runtime_s,
                        source_level=fm.source_level, quality=json.dumps(fm.quality, default=float),
                        **{f"map_{k}": v for k, v in fm.maps.items()})
    return fm


def side(sample_id, key, with_image=True):
    si = side_input(sample_id, key)
    return Side(key, np.asarray(si.points, float), tuple(si.extent), features(sample_id, key) if with_image else None)


def store_transform(tf_id=REF_TF_ID):
    c = sqlite3.connect(f"file:{STORE}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    r = dict(c.execute("select * from transforms where id = ?", (tf_id,)).fetchone())
    return np.array(json.loads(r["M_json"])), np.array(json.loads(r["t_json"])), r


def params(M):
    det = np.linalg.det(M)
    C = np.diag([1.0, -1.0]) if det < 0 else np.eye(2)
    sR = M @ C
    return {"scale": float(np.sqrt(abs(det))), "theta_deg": float(np.degrees(np.arctan2(sR[1, 0], sR[0, 0])) % 360),
            "reflect": bool(det < 0), "anisotropy": float(np.divide(*np.linalg.svd(M)[1]))}


def probe(moving_extent, r=400.0):
    c = np.asarray(moving_extent) / 2.0
    return c + np.array([[0, 0], [r, 0], [0, r], [-r, 0], [0, -r]], float)


def probe_dist(Ma, ta, Mb, tb, pr):
    """Median displacement (um, fixed frame) of the probe points between two transforms (same as registration_core)."""
    return float(np.median(np.linalg.norm(pr @ np.asarray(Ma).T + ta - (pr @ np.asarray(Mb).T + tb), axis=1)))


def footprint(M, t, extent):
    w, h = extent
    c = np.array([[0, 0], [w, 0], [w, h], [0, h], [0, 0]], float)
    return c @ np.asarray(M).T + np.asarray(t)
