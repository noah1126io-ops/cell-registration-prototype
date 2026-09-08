#!/usr/bin/env python3
"""
Export StarDist HE nuclei POLYGONS (segmentation) for one image, for rendering.

RUN ON A MACHINE WITH INTERNET / StarDist (e.g. local Jupyter). Saves polygon
vertices (in original HE pixel coords) to he_seg_stardist_<sid>.npz with array
'coord_hepx' of shape (N, 2, n_rays): [:,0]=y, [:,1]=x.

    python stardist_seg_export.py            # defaults to the bundled example
"""
import argparse, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--he", default="sample4_cropped.jpg")
    ap.add_argument("--out", default="he_seg_stardist_S2500586-1.npz")
    ap.add_argument("--upscale", type=int, default=4)
    ap.add_argument("--prob-thresh", type=float, default=0.4)
    ap.add_argument("--nms-thresh", type=float, default=0.3)
    a = ap.parse_args()

    from PIL import Image
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize

    img = np.asarray(Image.open(HERE / a.he).convert("RGB"))
    H0, W0 = img.shape[:2]; U = a.upscale
    big = np.asarray(Image.fromarray(img).resize((W0 * U, H0 * U), Image.LANCZOS))
    imn = normalize(big, 1.0, 99.8, axis=(0, 1))
    model = StarDist2D.from_pretrained("2D_versatile_he")
    labels, polys = model.predict_instances(
        imn, prob_thresh=a.prob_thresh, nms_thresh=a.nms_thresh,
        n_tiles=(U, U, 1), verbose=False)
    coord = np.asarray(polys["coord"], float) / U   # (N,2,n_rays) in HE px, [:,0]=y,[:,1]=x
    np.savez_compressed(HERE / a.out, coord_hepx=coord)
    print(f"{a.he}: {coord.shape[0]} nuclei polygons (n_rays={coord.shape[2]}) -> {a.out}")


if __name__ == "__main__":
    main()
