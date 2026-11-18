"""Figures for REPORT.md (env: cell-rg): zoomed nucleus + cell overlays of the full IF section, PNG -> JPEG for the repo."""
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
from skimage.segmentation import find_boundaries

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
F = HERE / "figures"


def zoom():
    wins = {"031-8_6": [(100, 650, 250, 800), (400, 300, 550, 450)], "031-6_2": [(450, 450, 600, 600), (150, 600, 300, 750)]}
    fig, ax = plt.subplots(2, 4, figsize=(20, 10.5))
    for r, (s, ws) in enumerate(wins.items()):
        im = np.asarray(Image.open(HERE / f"cache/crops/{s}/full/IF.png").convert("RGB"))
        lab = np.load(HERE / f"cache/seg/S1_cpsam_if_mem_full/{s}/full/IF.npz")["labels"]
        for c, (x0, y0, x1, y1) in enumerate(ws):
            sl = (slice(int(y0 * 2), int(y1 * 2)), slice(int(x0 * 2), int(x1 * 2)))
            a = im[sl].copy()
            b = find_boundaries(lab[sl], mode="inner")
            for k, raw in enumerate((True, False)):
                o = a.copy()
                if not raw:
                    o[b] = [255, 200, 0]
                axx = ax[r, c * 2 + k]
                axx.imshow(o, extent=(x0, x1, y1, y0))
                axx.set_title(f"{s} IF [{x0},{y0}]-[{x1},{y1}] µm" + ("" if raw else " + cell outlines (orange)"), fontsize=9)
                axx.tick_params(labelsize=7)
    fig.suptitle("S1_cpsam_if_mem_full (Cellpose-SAM, IF G+R membrane proxy). Left of each pair: image only. No ground truth.", fontsize=10)
    fig.tight_layout()
    fig.savefig(F / "zoom_full_section_if_mem.jpg", dpi=70, pil_kwargs={"quality": 82})


def to_jpeg():
    for p in sorted(F.glob("*.png")):
        Image.open(p).convert("RGB").save(p.with_suffix(".jpg"), quality=80, optimize=True)
        p.unlink()


if __name__ == "__main__":
    zoom()
    to_jpeg()
