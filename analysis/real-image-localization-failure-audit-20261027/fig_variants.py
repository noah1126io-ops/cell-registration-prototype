import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import FIG, features
from variants import variant_maps
SID = sys.argv[1]
fig, ax = plt.subplots(2, 4, figsize=(20, 10))
for i, k in enumerate(("HE", "IF")):
    fm = features(SID, k); v = variant_maps(SID, k)
    for j, (n, m) in enumerate((("tissue (engine)", fm.maps["tissue"]), ("nuclear (engine)", fm.maps["nuclear"]), ("occ (no nuclei)", v["occ"]), ("cleft (band-pass)", v["cleft"]))):
        ax[i, j].imshow(m, cmap="gray"); ax[i, j].set_title(f"{SID} {k} {n}"); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
fig.tight_layout(); fig.savefig(FIG / f"feature_variants_{SID}.png", dpi=60)
