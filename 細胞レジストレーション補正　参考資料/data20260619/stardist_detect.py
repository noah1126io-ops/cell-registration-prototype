#!/usr/bin/env python3
"""
StarDist HE nuclei detection for ONE image (regenerates the centroids npy).

RUN ON A MACHINE WITH INTERNET (e.g. local Jupyter): StarDist downloads the
'2D_versatile_he' model. The Cowork sandbox blocks that download.

One-time:  pip install stardist csbdeep tensorflow scikit-image pillow

Run (defaults to the bundled example):
    python stardist_detect.py
    python stardist_detect.py --he sample4_cropped.jpg --out he_nuclei_stardist_S2500586-1.npy

The 901-px crop is 4x-upscaled (nuclei are ~8-9 px otherwise, too small for the
model); centroids are scaled back to the original HE pixel frame and saved.
"""
import argparse, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--he", default="sample4_cropped.jpg")
    ap.add_argument("--out", default="he_nuclei_stardist_S2500586-1.npy")
    ap.add_argument("--upscale", type=int, default=4)
    ap.add_argument("--prob-thresh", type=float, default=0.4)
    ap.add_argument("--nms-thresh", type=float, default=0.3)
    a = ap.parse_args()

    from PIL import Image
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize
    from skimage.measure import regionprops

    img = np.asarray(Image.open(HERE / a.he).convert("RGB"))
    H0, W0 = img.shape[:2]
    U = a.upscale
    big = np.asarray(Image.fromarray(img).resize((W0 * U, H0 * U), Image.LANCZOS))
    imn = normalize(big, 1.0, 99.8, axis=(0, 1))
    model = StarDist2D.from_pretrained("2D_versatile_he")
    labels, _ = model.predict_instances(
        imn, prob_thresh=a.prob_thresh, nms_thresh=a.nms_thresh,
        n_tiles=(U, U, 1), verbose=False)
    props = regionprops(labels)
    cent = np.array([rp.centroid for rp in props])[:, [1, 0]].astype(float) / U  # (x,y) in HE px
    np.save(HERE / a.out, cent)
    print(f"{a.he}: {len(cent)} nuclei -> {a.out}")


if __name__ == "__main__":
    main()
