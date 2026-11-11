"""StarDist (2D_versatile_he) nuclei detection on a full-resolution HE OME-TIFF.

Outputs (in OUT/<sample>/):
  he_nuclei.csv            point_id, centroid_x_px, centroid_y_px, centroid_x_um, centroid_y_um, prob, area_um2
  he_nuclei_xy_px.npy      (n, 2) HE pixel xy (image origin top-left) -- Workflow C HE input format
  he_nuclei.geojson        polygons in HE level-0 pixel coordinates
  he_nuclei_qc.png         crops with outlines
  detection_meta.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from csbdeep.utils import normalize
from stardist.models import StarDist2D


def physical_size(tif: tifffile.TiffFile) -> float:
    m = re.search(r'PhysicalSizeX="([^"]+)"', tif.ome_metadata or "")
    return float(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prob-thresh", type=float, default=None)
    ap.add_argument("--nms-thresh", type=float, default=None)
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--downsample", type=int, default=2,
                    help="predict on level-0 downsampled by this factor (2 -> ~0.38 um/px, closer to the model's training scale)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with tifffile.TiffFile(args.image) as tif:
        px_um = physical_size(tif)
        img = tif.series[0].levels[0].asarray()
    print("image", img.shape, img.dtype, "px_um", px_um, flush=True)

    model = StarDist2D.from_pretrained("2D_versatile_he")
    # Model was trained on ~0.5 um/px-ish 40x HE; keep native resolution, normalise per channel jointly.
    f = args.downsample
    if f > 1:
        from skimage.transform import resize
        small = resize(img, (img.shape[0] // f, img.shape[1] // f, 3), preserve_range=True, anti_aliasing=True).astype(np.float32)
    else:
        small = img.astype(np.float32)
    x = normalize(small, 1.0, 99.8, axis=(0, 1, 2))
    kw = {}
    if args.prob_thresh is not None:
        kw["prob_thresh"] = args.prob_thresh
    if args.nms_thresh is not None:
        kw["nms_thresh"] = args.nms_thresh
    labels, details = model.predict_instances_big(
        x, axes="YXC", block_size=args.block, min_overlap=96, context=96,
        n_tiles=(2, 2, 1), **kw)
    t_pred = time.time() - t0
    print("pred done", t_pred, "s", flush=True)

    pts = details["points"].astype(float) * f + (f - 1) / 2.0   # (n, 2) row, col in level-0 px
    prob = details["prob"].astype(float)
    coord = details["coord"] * f + (f - 1) / 2.0   # (n, 2, n_rays) row/col in level-0 px
    n = len(pts)
    # polygon area via shoelace on ray vertices
    yy, xx = coord[:, 0, :], coord[:, 1, :]
    area_px = 0.5 * np.abs(np.sum(xx * np.roll(yy, -1, 1) - np.roll(xx, -1, 1) * yy, axis=1))
    df = pd.DataFrame({
        "point_id": np.arange(n),
        "centroid_x_px": pts[:, 1], "centroid_y_px": pts[:, 0],
        "centroid_x_um": pts[:, 1] * px_um, "centroid_y_um": pts[:, 0] * px_um,
        "prob": prob, "area_um2": area_px * px_um ** 2,
    })
    df.to_csv(out / "he_nuclei.csv", index=False)
    np.save(out / "he_nuclei_xy_px.npy", pts[:, ::-1].copy())

    feats = []
    for i in range(n):
        ring = np.stack([xx[i], yy[i]], 1)
        ring = np.vstack([ring, ring[:1]])
        feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring.round(2).tolist()]},
                      "properties": {"point_id": int(i), "prob": round(float(prob[i]), 4),
                                     "centroid_x": round(float(pts[i, 1]), 2), "centroid_y": round(float(pts[i, 0]), 2),
                                     "objectType": "detection", "classification": {"name": "Nucleus"}}})
    (out / "he_nuclei.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))

    # QC: 4 crops (800 px) at density quantiles + overview
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    centers = pts[rng.choice(n, size=min(4, n), replace=False)] if n else np.zeros((0, 2))
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    step = 16
    axes[0].imshow(img[::step, ::step])
    axes[0].scatter(pts[:, 1] / step, pts[:, 0] / step, s=0.3, c="lime")
    axes[0].set_title(f"overview n={n}")
    for ax, c in zip(axes[1:], centers):
        r0 = int(np.clip(c[0] - 400, 0, img.shape[0] - 800)); c0 = int(np.clip(c[1] - 400, 0, img.shape[1] - 800))
        ax.imshow(img[r0:r0 + 800, c0:c0 + 800])
        sel = np.where((pts[:, 0] >= r0 - 50) & (pts[:, 0] < r0 + 850) & (pts[:, 1] >= c0 - 50) & (pts[:, 1] < c0 + 850))[0]
        for i in sel:
            ax.plot(np.r_[xx[i], xx[i][:1]] - c0, np.r_[yy[i], yy[i][:1]] - r0, lw=0.6, c="lime")
        ax.set_xlim(0, 800); ax.set_ylim(800, 0)
        ax.set_title(f"crop ({c0},{r0}) px, 800 px = {800 * px_um:.0f} um")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "he_nuclei_qc.png", dpi=90)

    meta = {"image": str(args.image), "shape": list(img.shape), "pixel_size_um": px_um, "model": "2D_versatile_he",
            "normalize": [1.0, 99.8], "downsample": f, "prob_thresh": model.thresholds.prob if args.prob_thresh is None else args.prob_thresh,
            "nms_thresh": model.thresholds.nms if args.nms_thresh is None else args.nms_thresh,
            "n_nuclei": int(n), "area_um2_median": float(np.median(df.area_um2)) if n else None,
            "runtime_s": time.time() - t0}
    (out / "detection_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
