#!/usr/bin/env python3
"""
Align ONE HE image to ONE nuclei GeoJSON (reproduces ALIGN_HE_TO_GEOJSON.md).

Self-contained: all paths are relative to this folder. No SpatialArc / bundling.
Stages: (precomputed StarDist centers) -> global affine -> confidence-gated
fine center-snap -> warp HE onto the GeoJSON world-um grid + overlay.

Run (defaults to the bundled S2500586-1 example):
    python align_he_to_geojson.py
    python align_he_to_geojson.py --he sample4_cropped.jpg \
        --geojson S2500586-1_dl20_um.geojson \
        --centroids he_nuclei_stardist_S2500586-1.npy

If --centroids is missing, pass --allow-classical to use the no-ML
hematoxylin-peak detector (lower quality).

Outputs (this folder):
    warpedHE_<stem>_yflip.png   warped HE on the GeoJSON world-um grid
    warp_<stem>.json            image_bounds [x0,y0,x1,y1], flip, Jacobian min
    overlay_<stem>.png          warped HE + rasterized GeoJSON nuclei (zoom)

Deps: numpy, scipy, scikit-image, pillow, matplotlib  (+ register_he_to_geojson.py
in this folder). StarDist itself is only needed to (re)generate the centroids npy
via stardist_detect.py.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from register_he_to_geojson import (
    load_geojson_centroids, best_of_initial_rotations, detect_nuclei_classical,
)
from PIL import Image
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, map_coordinates

BIN = 6.0  # working grid um/px


def affine_lstsq(s, d):
    X = np.hstack([s, np.ones((len(s), 1))]); M, *_ = np.linalg.lstsq(X, d, rcond=None)
    return M[:2].T, M[2]


def affine_icp(s, d, A, b, it=80, trim=0.7, tol=1e-4):
    tree = cKDTree(d); prev = np.inf
    for _ in range(it):
        w = s @ A.T + b; dist, idx = tree.query(w)
        keep = dist <= np.quantile(dist, trim)
        if keep.sum() < 6:
            break
        A, b = affine_lstsq(s[keep], d[idx[keep]]); e = float(dist[keep].mean())
        if abs(prev - e) < tol:
            break
        prev = e
    return A, b


def report(pts, gj, tag):
    d, _ = cKDTree(gj).query(pts)
    tg, th = cKDTree(gj), cKDTree(pts)
    _, iH = tg.query(pts); _, iG = th.query(gj)
    mut = (iG[iH] == np.arange(len(pts))) & (d <= 10.0)
    md = d[mut]
    print(f"  [{tag}] all-NN med={np.median(d):.2f}um | "
          f"mutual(<10um) n={int(mut.sum())} med={np.median(md):.2f}um "
          f"<3um={100*np.mean(md<3):.0f}%")
    return float(np.median(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--he", default="sample4_cropped.jpg")
    ap.add_argument("--geojson", default="S2500586-1_dl20_um.geojson")
    ap.add_argument("--centroids", default="he_nuclei_stardist_S2500586-1.npy")
    ap.add_argument("--allow-classical", action="store_true")
    ap.add_argument("--sig-k", type=float, default=12.0)   # field bandwidth (um)
    ap.add_argument("--lam", type=float, default=0.3)       # ridge -> identity
    ap.add_argument("--r-match", type=float, default=10.0)  # mutual-NN radius (um)
    ap.add_argument("--sig-c", type=float, default=10.0)    # coherence tolerance
    ap.add_argument("--sig-r", type=float, default=8.0)     # closeness scale
    ap.add_argument("--out-res", type=float, default=1.5)   # output um/px
    a = ap.parse_args()

    he_path = HERE / a.he
    geo_path = HERE / a.geojson
    cen_path = HERE / a.centroids
    stem = Path(a.geojson).stem

    # --- 1. HE nuclei centers (precomputed StarDist, or classical fallback) ---
    if cen_path.exists():
        he_px = np.load(cen_path); src = f"StarDist npy ({cen_path.name})"
    elif a.allow_classical:
        he_px = detect_nuclei_classical(str(he_path), smooth_sigma=1.5,
                                        min_distance=6, hematoxylin_percentile=82.0)
        src = "classical (no-ML)"
    else:
        sys.exit(f"missing {cen_path}; run stardist_detect.py or pass --allow-classical")
    gj = load_geojson_centroids(str(geo_path))
    with Image.open(he_path) as im:
        W0, H0 = im.size
    print(f"HE nuclei: {len(he_px)} via {src} | GeoJSON nuclei: {len(gj)}")

    # --- 2. global affine (try both Y orientations) ---
    rots = [float(d) for d in range(0, 360, 30)]
    icp = dict(max_iter=100, tol=1e-4, trim_quantile=0.8)
    best = None
    for flip in (False, True):
        s = he_px.copy()
        if flip:
            s[:, 1] = H0 - s[:, 1]
        sim, info, _ = best_of_initial_rotations(s, gj, rots, icp)
        if best is None or info["mean_residual"] < best[2]["mean_residual"]:
            best = (s, sim, info, flip)
    s, sim, _, FLIP = best
    c, sn = np.cos(sim.rotation_rad), np.sin(sim.rotation_rad)
    A0 = sim.scale * np.array([[c, -sn], [sn, c]])
    A, b = affine_icp(s, gj, A0, np.asarray(sim.translation)); Ainv = np.linalg.inv(A)
    p = s @ A.T + b
    print(f"affine: flip={FLIP} scale={sim.scale:.4f} rot={np.rad2deg(sim.rotation_rad):.2f}deg")
    report(p, gj, "affine")

    # grid (GeoJSON bbox + pad)
    pad = 30.0
    x0 = min(gj[:, 0].min(), p[:, 0].min()) - pad; y0 = min(gj[:, 1].min(), p[:, 1].min()) - pad
    x1 = max(gj[:, 0].max(), p[:, 0].max()) + pad; y1 = max(gj[:, 1].max(), p[:, 1].max()) + pad
    nx = int(np.ceil((x1 - x0) / BIN)); ny = int(np.ceil((y1 - y0) / BIN))

    # --- 3. fine center-snap field from mutual-NN pairs ---
    tg, th = cKDTree(gj), cKDTree(p)
    dnn, iH = tg.query(p); _, iG = th.query(gj)
    mut = (iG[iH] == np.arange(len(p))) & (dnn <= a.r_match)
    pm, qm = p[mut], gj[iH[mut]]; dvec, resid = qm - pm, dnn[mut]
    tp = cKDTree(pm); coh = np.zeros(len(pm)); supp = np.zeros(len(pm))
    for i, nb in enumerate(tp.query_ball_point(pm, 40.0)):
        nb = [j for j in nb if j != i]; supp[i] = 1.0 - np.exp(-len(nb) / 4.0)
        if nb:
            cons = np.median(dvec[nb], axis=0)
            coh[i] = np.exp(-np.sum((dvec[i] - cons) ** 2) / (2 * a.sig_c ** 2))
    conf = np.exp(-(resid ** 2) / (2 * a.sig_r ** 2)) * coh * supp

    def splat(pts, vals):
        ix = np.floor((pts[:, 0] - x0) / BIN).astype(int); iy = np.floor((pts[:, 1] - y0) / BIN).astype(int)
        m = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny); g = np.zeros((ny, nx))
        np.add.at(g, (iy[m], ix[m]), vals[m]); return g
    num_x = gaussian_filter(splat(pm, conf * dvec[:, 0]), a.sig_k / BIN)
    num_y = gaussian_filter(splat(pm, conf * dvec[:, 1]), a.sig_k / BIN)
    den = gaussian_filter(splat(pm, conf), a.sig_k / BIN)
    lam = a.lam * np.median(den[den > 0]); Fx = num_x / (den + lam); Fy = num_y / (den + lam)
    dxx = np.gradient(Fx, BIN, axis=1); dxy = np.gradient(Fx, BIN, axis=0)
    dyx = np.gradient(Fy, BIN, axis=1); dyy = np.gradient(Fy, BIN, axis=0)
    jac = (1 + dxx) * (1 + dyy) - dxy * dyx
    fi = (p[:, 0] - x0) / BIN; fj = (p[:, 1] - y0) / BIN
    pdef = p + np.column_stack([map_coordinates(Fx, [fj, fi], order=1, mode="nearest"),
                                map_coordinates(Fy, [fj, fi], order=1, mode="nearest")])
    report(pdef, gj, "fine")
    print(f"  jac_min={jac.min():.3f} disp_max={np.hypot(Fx, Fy).max():.1f}um"
          + ("  WARNING folding (raise --sig-k/--lam)" if jac.min() <= 0 else ""))

    # --- 4. warp the HE image onto the GeoJSON world-um grid ---
    he_rgb = np.asarray(Image.open(he_path).convert("RGB"))
    OX, OY = np.meshgrid(np.arange(x0, x1, a.out_res), np.arange(y0, y1, a.out_res))

    def interpF(xw, yw):
        return (map_coordinates(Fx, [(yw - y0) / BIN, (xw - x0) / BIN], order=1, mode="nearest"),
                map_coordinates(Fy, [(yw - y0) / BIN, (xw - x0) / BIN], order=1, mode="nearest"))
    Xc, Yc = OX.copy(), OY.copy()
    for _ in range(6):
        fx, fy = interpF(Xc.ravel(), Yc.ravel())
        Xc = OX - fx.reshape(OX.shape); Yc = OY - fy.reshape(OY.shape)
    sxp = Ainv[0, 0] * (Xc - b[0]) + Ainv[0, 1] * (Yc - b[1])
    syp = Ainv[1, 0] * (Xc - b[0]) + Ainv[1, 1] * (Yc - b[1])
    row = (H0 - syp) if FLIP else syp
    out = np.zeros((*OX.shape, 3), np.uint8)
    for ch in range(3):
        out[..., ch] = map_coordinates(he_rgb[..., ch], [row.ravel(), sxp.ravel()],
                                       order=1, mode="constant", cval=0.0).reshape(OX.shape).astype(np.uint8)
    png = HERE / f"warpedHE_{stem}_yflip.png"
    Image.fromarray(out[::-1]).save(png)   # Y-flip = display/overlay-ready orientation
    bounds = [float(x0), float(y0), float(x1), float(y1)]
    json.dump(dict(he=a.he, geojson=a.geojson, flip=bool(FLIP), image_bounds=bounds,
                   jac_min=float(jac.min()), warped_png=png.name),
              open(HERE / f"warp_{stem}.json", "w"), indent=2)
    print(f"warped HE -> {png.name}  image_bounds={bounds}")

    # --- overlay: warped HE + rasterized GeoJSON nuclei (zoom) ---
    from skimage.draw import polygon as skpoly, polygon_perimeter
    ony, onx = OX.shape
    fill = np.zeros((ony, onx), bool); edge = np.zeros((ony, onx), bool)
    gjson = json.load(open(geo_path))
    for ft in gjson["features"]:
        if ft.get("geometry", {}).get("type") != "Polygon":
            continue
        ring = np.asarray(ft["geometry"]["coordinates"][0], float)
        cc = (ring[:, 0] - x0) / a.out_res; rr = (ring[:, 1] - y0) / a.out_res
        fr, fc = skpoly(rr, cc, shape=(ony, onx)); fill[fr, fc] = True
        pr, pc = polygon_perimeter(rr, cc, shape=(ony, onx)); edge[pr, pc] = True
    comp = out.astype(float).copy()
    comp[fill] = 0.55 * comp[fill] + 0.45 * np.array([0, 200, 220]); comp[edge] = [0, 255, 255]
    comp = np.clip(comp, 0, 255).astype(np.uint8)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(comp, origin="lower", extent=[x0, x1, y0, y1])
    ax.set_xlim(350, 650); ax.set_ylim(400, 700)
    ax.set_title("warped HE + GeoJSON nuclei (cyan)\nzoom 350-650 x 400-700 um")
    ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
    fig.tight_layout(); fig.savefig(HERE / f"overlay_{stem}.png", dpi=140); plt.close(fig)
    print(f"overlay -> overlay_{stem}.png")


if __name__ == "__main__":
    main()
