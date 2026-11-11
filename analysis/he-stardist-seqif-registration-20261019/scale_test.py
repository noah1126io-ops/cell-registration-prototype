import sys, numpy as np, tifffile, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from csbdeep.utils import normalize; from stardist.models import StarDist2D
from skimage.transform import resize
img_path, out, cx, cy = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with tifffile.TiffFile(img_path) as t: full = t.series[0].levels[0].asarray()
x_all = normalize(full[::8, ::8], 1, 99.8, axis=(0, 1, 2))  # just for stats
lo, hi = np.percentile(full[::8, ::8], [1, 99.8])
crop = full[cy:cy+2048, cx:cx+2048]
m = StarDist2D.from_pretrained("2D_versatile_he")
res = {}
for f in (1, 2):
    c = crop if f == 1 else (resize(crop, (1024, 1024, 3), preserve_range=True, anti_aliasing=True))
    xn = np.clip((c.astype(float) - lo) / (hi - lo), 0, None)
    lab, det = m.predict_instances(xn, n_tiles=(2, 2, 1))
    res[f] = det
    a = np.bincount(lab.ravel())[1:] * (0.18872 * f) ** 2
    print(f"factor {f}: n={len(det['points'])} area_um2 median={np.median(a):.1f} p5/p95={np.percentile(a,5):.1f}/{np.percentile(a,95):.1f} prob_med={np.median(det['prob']):.2f}", flush=True)
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
for a, f in zip(ax, (1, 2)):
    a.imshow(crop); co = res[f]["coord"] * f
    for p in co: a.plot(np.r_[p[1], p[1][:1]], np.r_[p[0], p[0][:1]], c="lime", lw=0.5)
    a.set_title(f"factor {f}: n={len(co)}"); a.axis("off")
fig.tight_layout(); fig.savefig(out, dpi=70)
