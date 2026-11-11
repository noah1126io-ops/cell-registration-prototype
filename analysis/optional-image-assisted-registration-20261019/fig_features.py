"""structural_feature_maps.png: HE and IF structural representations, HE also shown warped into the IF frame
with the provider placement of 031-6_2 (same section) for visual comparability (figure only)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from oir import metrics as mt  # noqa: E402
from run_real import CACHE, load_feats, reference_he_to_if  # noqa: E402

rows = []
for s in ("031-6_2", "031-8_6"):
    he = load_feats(CACHE / f"feats_{s}_HE_2um.npz"); iff = load_feats(CACHE / f"feats_{s}_IF_2um.npz")
    rows.append((s, he, iff, reference_he_to_if(s)))
keys = ("tissue", "nuclear", "edt")
fig, ax = plt.subplots(len(rows) * 3, 3, figsize=(18, 6 * len(rows) * 3 / 1.0))
for r, (s, he, iff, ref) in enumerate(rows):
    for j, k in enumerate(keys):
        ax[3 * r, j].imshow(he.maps[k], cmap="magma"); ax[3 * r, j].set_title(f"{s} HE {k} (full HE field)")
        w = mt.warp_map_into(he.maps[k], he.res_um, ref, iff.shape, iff.res_um)
        ax[3 * r + 1, j].imshow(w, cmap="magma"); ax[3 * r + 1, j].set_title(f"{s} HE {k} in IF frame ({'provider registration' if s == '031-6_2' else 'APPROXIMATE placement, not registered'})", fontsize=9)
        ax[3 * r + 2, j].imshow(iff.maps[k], cmap="magma"); ax[3 * r + 2, j].set_title(f"{s} IF {k}")
for a in ax.ravel():
    a.axis("off")
fig.tight_layout(); fig.savefig(OUT / "structural_feature_maps.png", dpi=45)
print("ok")
