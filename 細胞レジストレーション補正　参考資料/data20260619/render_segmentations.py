#!/usr/bin/env python3
"""
Render the two segmentations as SEPARATE images, in the same world-um frame:
  - geojson_seg.png    : fluorescence GeoJSON nuclei polygons (native world um)
  - stardist_seg.png   : StarDist HE nuclei polygons, mapped to world um via the
                         global affine (so it overlays/compares with the GeoJSON)

GeoJSON render needs nothing special (run in VM). The StarDist render needs the
polygon file he_seg_stardist_<sid>.npz (produced by stardist_seg_export.py on a
machine with StarDist) AND the centroids npy (to recompute the affine).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from register_he_to_geojson import load_geojson_centroids, best_of_initial_rotations
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def affine_lstsq(s, d):
    X = np.hstack([s, np.ones((len(s), 1))]); M, *_ = np.linalg.lstsq(X, d, rcond=None)
    return M[:2].T, M[2]


def affine_icp(s, d, A, b, it=80, trim=0.7, tol=1e-4):
    tree = cKDTree(d); prev = np.inf
    for _ in range(it):
        w = s @ A.T + b; dist, idx = tree.query(w)
        keep = dist <= np.quantile(dist, trim)
        if keep.sum() < 6: break
        A, b = affine_lstsq(s[keep], d[idx[keep]]); e = float(dist[keep].mean())
        if abs(prev - e) < tol: break
        prev = e
    return A, b


def geojson_polys(path):
    gj = json.load(open(path)); polys = []
    for ft in gj.get("features", []):
        g = ft.get("geometry", {})
        if g.get("type") == "Polygon":
            polys.append(np.asarray(g["coordinates"][0], float))
    return polys


def render(polys, extent, out, title, facecolor="#00c8dc"):
    x0, y0, x1, y1 = extent
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_collection(PolyCollection(polys, facecolors=facecolor, edgecolors="#004c5a",
                                     linewidths=0.2, alpha=0.85))
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.set_title(title); ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"-> {Path(out).name}  ({len(polys)} polygons)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geojson", default="S2500586-1_dl20_um.geojson")
    ap.add_argument("--he", default="sample4_cropped.jpg")
    ap.add_argument("--centroids", default="he_nuclei_stardist_S2500586-1.npy")
    ap.add_argument("--stardist-polys", default="he_seg_stardist_S2500586-1.npz")
    a = ap.parse_args()

    gpolys = geojson_polys(HERE / a.geojson)
    gj = load_geojson_centroids(str(HERE / a.geojson))
    # common extent = geojson bbox + pad
    allpts = np.vstack(gpolys); pad = 30.0
    x0, y0 = allpts[:, 0].min() - pad, allpts[:, 1].min() - pad
    x1, y1 = allpts[:, 0].max() + pad, allpts[:, 1].max() + pad
    extent = (x0, y0, x1, y1)
    render(gpolys, extent, HERE / "geojson_seg.png",
           f"GeoJSON (fluorescence) nuclei — {len(gpolys)} polygons")

    sp = HERE / a.stardist_polys
    if not sp.exists():
        print(f"(StarDist polygons {sp.name} not found — run stardist_seg_export.py "
              f"on a machine with StarDist, then re-run this.)")
        return
    # map StarDist polygons (HE px) -> world um via the global affine
    from PIL import Image
    he = np.load(HERE / a.centroids)
    with Image.open(HERE / a.he) as im: W0, H0 = im.size
    rots = [float(d) for d in range(0, 360, 30)]
    icp = dict(max_iter=100, tol=1e-4, trim_quantile=0.8)
    best = None
    for flip in (False, True):
        s = he.copy()
        if flip: s[:, 1] = H0 - s[:, 1]
        sim, info, _ = best_of_initial_rotations(s, gj, rots, icp)
        if best is None or info["mean_residual"] < best[2]["mean_residual"]:
            best = (s, sim, info, flip)
    s, sim, _, FLIP = best
    c, sn = np.cos(sim.rotation_rad), np.sin(sim.rotation_rad)
    A0 = sim.scale * np.array([[c, -sn], [sn, c]])
    A, b = affine_icp(s, gj, A0, np.asarray(sim.translation))

    z = np.load(sp)
    coord = z["coord_hepx"]  # (N, 2, n_rays): [:,0]=y_hepx, [:,1]=x_hepx
    spolys = []
    for i in range(coord.shape[0]):
        y = coord[i, 0]; x = coord[i, 1]
        sy = (H0 - y) if FLIP else y           # apply flip used by the affine
        pts = np.column_stack([x, sy]) @ A.T + b   # HE px -> world um
        spolys.append(pts)
    render(spolys, extent, HERE / "stardist_seg.png",
           f"StarDist (HE) nuclei → world um — {len(spolys)} polygons",
           facecolor="#ff9030")


if __name__ == "__main__":
    main()
