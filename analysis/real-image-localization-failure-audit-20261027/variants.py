"""Candidate structural features for the ablation (analysis only; nothing here is used by the engine).

occ    : stain occupancy that does NOT come from nuclei.  HE = optical density (sum RGB); IF = max(G, R) (nuclear
         Hoechst = B excluded).  Smoothed at cell scale, then saturated at the in-field median so dense nuclei / bright
         membranes cannot dominate:  occ = clip((x - p5) / (median_tissue - p5), 0, 1), median_tissue = median of the
         smoothed signal above its Otsu threshold (so a mostly-empty slide does not saturate its background).
         Clefts / lumen / background -> 0, myocardium -> 1.
cleft  : 1 - occ, band-passed (minus a 60 um local mean) -> the interstitial cleft network the reviewer used.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from skimage.filters import threshold_otsu

from common import CACHE, side_input
from src.registration_core.features import read_level


def raw_signal(sample_id, key, res_um=2.0):
    si = side_input(sample_id, key)
    ds = np.asarray(si.pyramid_downsamples)
    lvl = int(np.where(ds * si.pixel_um <= res_um * 0.75 + 1e-9)[0].max())
    src_um = si.pixel_um * ds[lvl]
    img = read_level(si.image_path, lvl).astype(np.float32)
    if key == "HE":
        x = -np.log((img + 1.0) / 256.0).sum(-1)
    else:
        x = np.maximum(img[..., 1], img[..., 0])
    return zoom(x, src_um / res_um, order=1, prefilter=False)


def occupancy(x, res_um=2.0, sigma_um=6.0):
    s = gaussian_filter(x, sigma_um / res_um)
    p5 = np.percentile(s, 5)
    thr = threshold_otsu(s[::2, ::2])
    hi = float(np.median(s[s > thr]))
    return np.clip((s - p5) / max(hi - p5, 1e-9), 0, 1)


def variant_maps(sample_id, key, res_um=2.0):
    p = CACHE / f"var_{sample_id}_{key}_{res_um:g}.npz"
    if p.exists():
        z = np.load(p)
        return {k: z[k] for k in z.files}
    x = raw_signal(sample_id, key, res_um)
    occ = occupancy(x, res_um)
    cl = 1.0 - occ
    cleft = cl - gaussian_filter(cl, 60.0 / res_um)
    occ12 = occupancy(x, res_um, 12.0)
    out = {"occ": occ.astype(np.float32), "cleft": cleft.astype(np.float32), "occ12": occ12.astype(np.float32)}
    np.savez_compressed(p, **out)
    return out
