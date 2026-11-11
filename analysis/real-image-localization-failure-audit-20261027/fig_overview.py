"""original_image_pair.png / feature_map_comparison.png / reference overlay: what the search sees vs what the user sees."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
from common import FIG, HERE, features, footprint, side_input, store_transform, params
from src.registration_core.features import read_level

SID = sys.argv[1] if len(sys.argv) > 1 else "031-8_6"
he, IF = side_input(SID, "HE"), side_input(SID, "IF")


def thumb(si, target_um=4.0):
    ds = np.asarray(si.pyramid_downsamples); lvl = int(np.where(ds * si.pixel_um <= target_um)[0].max())
    return read_level(si.image_path, lvl), si.pixel_um * ds[lvl]


def warp_to_fixed(img, src_um, M, t, out_shape, out_um):
    """Render a moving-frame image in the fixed frame: fixed = M @ moving + t."""
    B = np.linalg.inv(M); P = np.array([[0, 1], [1, 0]], float)
    mat = P @ B @ P * (out_um / src_um); off = (P @ (-B @ t))[::1][::-1] if False else (-(B @ t))[::-1] / src_um
    ch = [affine_transform(img[..., c].astype(np.float32), mat, offset=off, output_shape=out_shape, order=1, cval=np.nan) for c in range(img.shape[-1])]
    return np.stack(ch, -1)


if __name__ == "__main__":
    hi, hu = thumb(he); ii, iu = thumb(IF)
    art = json.load(open(HERE / f"baseline/{'8_6' if SID == '031-8_6' else '6_2'}_auto_artifact_20261022.json"))
    ref = store_transform() if SID == "031-8_6" else None
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(hi, extent=(0, hi.shape[1] * hu, hi.shape[0] * hu, 0)); ax[0].set_title(f"{SID} H&E (fixed), {hu:.2f} um/px")
    cols = {"image": "#ff7f0e", "points": "#1f77b4"}
    for c in art["candidates"]:
        fp = footprint(c["T_search"]["M"], c["T_search"]["t"], IF.extent)
        ax[0].plot(fp[:, 0], fp[:, 1], "-", color=cols[c["source"]], lw=1.5 if c["rank"] == 0 else 0.7, alpha=0.9 if c["rank"] == 0 else 0.5)
        ax[0].text(*fp[0], c["id"], color=cols[c["source"]], fontsize=7)
    if ref is not None:
        fp = footprint(ref[0], ref[1], IF.extent); ax[0].plot(fp[:, 0], fp[:, 1], "-", color="lime", lw=2.5); ax[0].text(*fp[0], "manual reference", color="lime", fontsize=9)
    ax[1].imshow(ii, extent=(0, ii.shape[1] * iu, ii.shape[0] * iu, 0)); ax[1].set_title(f"{SID} fluorescence (moving, display RGB), {iu:.2f} um/px")
    for a in ax: a.set_xlabel("um")
    fig.tight_layout(); fig.savefig(FIG / f"original_image_pair_{SID}.png", dpi=90); plt.close(fig)
    # feature maps
    fh, fi = features(SID, "HE"), features(SID, "IF")
    names = ["tissue", "nuclear", "edt", "tissue_bin"]
    fig, ax = plt.subplots(2, 5, figsize=(25, 10))
    ax[0, 0].imshow(hi); ax[0, 0].set_title("H&E RGB")
    ax[1, 0].imshow(ii); ax[1, 0].set_title("IF display RGB")
    for j, n in enumerate(names, 1):
        ax[0, j].imshow(fh.maps[n], cmap="gray"); ax[0, j].set_title(f"HE {n}")
        ax[1, j].imshow(fi.maps[n], cmap="gray"); ax[1, j].set_title(f"IF {n}")
    for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(FIG / f"feature_map_comparison_{SID}.png", dpi=70); plt.close(fig)
    if ref is not None:
        M, t, _ = ref
        print("reference", params(M), t)
        w = warp_to_fixed(ii, iu, M, t, hi.shape[:2], hu)
        fig, ax = plt.subplots(1, 3, figsize=(24, 8))
        fp = footprint(M, t, IF.extent)
        x0, y0 = fp[:, 0].min() - 100, fp[:, 1].min() - 100; x1, y1 = fp[:, 0].max() + 100, fp[:, 1].max() + 100
        ex = (0, hi.shape[1] * hu, hi.shape[0] * hu, 0)
        ax[0].imshow(hi, extent=ex); ax[0].set_title("H&E (reference footprint)")
        wn = np.nan_to_num(w); wn = wn / max(np.percentile(wn[wn > 0], 99.5), 1)
        ax[1].imshow(np.clip(wn, 0, 1), extent=ex); ax[1].set_title("IF warped by manual reference")
        g = hi.mean(-1) / 255.0
        ax[2].imshow(g, cmap="gray", extent=ex); ax[2].imshow(np.clip(wn, 0, 1), extent=ex, alpha=0.45); ax[2].set_title("overlay")
        for a in ax: a.plot(fp[:, 0], fp[:, 1], "-", color="lime", lw=1.5); a.set_xlim(x0, x1); a.set_ylim(y1, y0)
        fig.tight_layout(); fig.savefig(FIG / f"reference_overlay_{SID}.png", dpi=80); plt.close(fig)
