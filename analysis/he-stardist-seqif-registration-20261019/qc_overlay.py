"""Visual QC: seqIF image + nuclei mapped into HE pixel space with the fitted affine (and the existing placement)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tifffile  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402


def level(path, k):
    with tifffile.TiffFile(path) as t:
        return t.series[0].levels[k].asarray()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--reg", required=True, help="registration output folder")
    ap.add_argument("--nuclei", required=True)
    args = ap.parse_args()
    s, data, reg = args.sample, Path(args.data), Path(args.reg)
    z = np.load(reg / "registration_transform.npz")
    he_um, q_um = float(z["he_pixel_um"]), float(z["seqif_pixel_um"])
    he_tif = next(data.glob(f"{s}_HE*.ome.tif"))
    q_tif = data / f"{s}_seqif.ome.tif"

    def he2q_px(he_px, M, t):            # HE level-0 px -> seqIF level-0 px
        return ((np.asarray(he_px, float) * he_um) @ M.T + t) / q_um

    def q2he_px(q_px, M, t):
        return ((np.asarray(q_px, float) * q_um - t) @ np.linalg.inv(M).T) / he_um

    gj = json.loads((data / f"{s}_seg_seqif_level0.geojson").read_text())
    rings = [np.asarray(f["geometry"]["coordinates"][0], float) for f in gj["features"]]
    nuc = pd.read_csv(args.nuclei)
    he_pts = nuc[["centroid_x_px", "centroid_y_px"]].to_numpy()

    # seqIF field outline in HE px (ours / existing)
    with tifffile.TiffFile(q_tif) as t:
        qh, qw = t.series[0].shape[:2]
    box = np.array([[0, 0], [qw, 0], [qw, qh], [0, qh], [0, 0]], float)
    box_ours = q2he_px(box, z["M"], z["t"]); box_prior = q2he_px(box, z["M_prior"], z["t_prior"])

    # ---------------- overview (HE level 3) + seqIF warped into HE (level 2 grid)
    he3 = level(he_tif, 3); f3 = 8
    he2 = level(he_tif, 2); f2 = 4
    q1 = level(q_tif, 1); fq = 2
    x0, y0 = np.floor(box_ours[:4].min(0)).astype(int) - 200
    x1, y1 = np.ceil(box_ours[:4].max(0)).astype(int) + 200
    xs = np.arange(max(x0, 0), min(x1, he2.shape[1] * f2), f2)
    ys = np.arange(max(y0, 0), min(y1, he2.shape[0] * f2), f2)
    gx, gy = np.meshgrid(xs, ys)
    qq = he2q_px(np.c_[gx.ravel(), gy.ravel()], z["M"], z["t"]) / fq
    warped = np.stack([map_coordinates(q1[..., c].astype(float), [qq[:, 1], qq[:, 0]], order=1, cval=0).reshape(gx.shape)
                       for c in range(3)], -1)
    he_crop = he2[ys[0] // f2: ys[0] // f2 + len(ys), xs[0] // f2: xs[0] // f2 + len(xs)]
    warped = warped[: he_crop.shape[0], : he_crop.shape[1]]

    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    ax[0].imshow(he3); ax[0].plot(*(box_ours / f3).T, c="lime", lw=2, label="seqIF field (this registration)")
    ax[0].plot(*(box_prior / f3).T, c="magenta", lw=1.5, ls="--", label="seqIF field (existing seg_he placement)")
    ax[0].legend(loc="lower left"); ax[0].set_title(f"{s}: HE overview"); ax[0].axis("off")
    ax[1].imshow(he_crop); ax[1].set_title("HE (seqIF field)"); ax[1].axis("off")
    wv = warped / max(np.percentile(warped, 99.5), 1)
    ax[2].imshow(np.clip(wv, 0, 1)); ax[2].set_title("seqIF warped into HE frame (this registration)"); ax[2].axis("off")
    fig.tight_layout(); fig.savefig(reg / "qc_overview.png", dpi=70); plt.close(fig)

    # checkerboard HE / seqIF (grayscale)
    g_he = 1 - he_crop.mean(-1) / 255.0
    g_q = np.clip(wv.max(-1), 0, 1)
    tile = 64
    cb = ((np.arange(g_he.shape[0])[:, None] // tile + np.arange(g_he.shape[1])[None] // tile) % 2).astype(bool)
    img = np.where(cb, g_he / max(np.percentile(g_he, 99.5), 1e-6), g_q)
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(np.clip(img, 0, 1), cmap="gray"); ax.set_title("checkerboard: HE (inverted grey) / seqIF (max channel)"); ax.axis("off")
    fig.tight_layout(); fig.savefig(reg / "qc_checkerboard.png", dpi=70); plt.close(fig)

    # ---------------- zoom crops at level 0: HE + StarDist centroids + seqIF polygons (ours/existing)
    he0 = level(he_tif, 0)
    rng = np.random.default_rng(1)
    cent_q = np.array([r[:-1].mean(0) for r in rings])
    pick = cent_q[rng.choice(len(cent_q), 4, replace=False)]
    fig, ax = plt.subplots(1, 4, figsize=(32, 8))
    half = 400                          # HE px (~75 um)
    for a, cq in zip(ax, pick):
        c = q2he_px(cq[None], z["M"], z["t"])[0].astype(int)
        r0, c0 = max(c[1] - half, 0), max(c[0] - half, 0)
        a.imshow(he0[r0: r0 + 2 * half, c0: c0 + 2 * half], extent=(c0, c0 + 2 * half, r0 + 2 * half, r0))
        for ring in rings:
            ro = q2he_px(ring, z["M"], z["t"]); rp = q2he_px(ring, z["M_prior"], z["t_prior"])
            if np.abs(ro.mean(0) - c).max() < half + 50:
                a.plot(*ro.T, c="lime", lw=1.2)
            if np.abs(rp.mean(0) - c).max() < half + 50:
                a.plot(*rp.T, c="magenta", lw=0.8, ls="--")
        sel = (np.abs(he_pts - c) < half).all(1)
        a.scatter(*he_pts[sel].T, s=12, c="cyan", marker="+")
        a.set_xlim(c0, c0 + 2 * half); a.set_ylim(r0 + 2 * half, r0); a.axis("off")
        a.set_title(f"HE px ({c0},{r0}); green=seqIF nuclei (registered), magenta=existing, cyan=StarDist")
    fig.tight_layout(); fig.savefig(reg / "qc_zoom_nuclei.png", dpi=60); plt.close(fig)
    print("qc done")


if __name__ == "__main__":
    main()
