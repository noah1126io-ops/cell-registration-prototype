"""Figures from the saved smoke results (recomputes fields from the saved auto + review files; no new matching).

fig1_<sample>.png : IF ROI | H&E Affine | H&E Nonrigid (reviewed + RBF, or D when reviewed is not applied) with the forward-mapped
                    H&E nuclei on top (same transform as the image) and the |nonrigid - affine| image difference.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from src.registration_explorer import catalog, imaging  # noqa: E402
from src.registration_research import cell_review as CR  # noqa: E402
from smoke_real import he_reader  # noqa: E402


def one(sample, crop_um=220.0, dst=0.5):
    sm = json.loads((HERE / f"smoke_{sample}.json").read_text())
    S = catalog.load_manifest()[sample]
    he, iff = S.modality("HE"), S.modality("IF")
    he_pts, _ = catalog.load_nuclei(he)
    if_pts, _ = catalog.load_nuclei(iff)
    auto = CR.load_auto(REPO / sm["auto_dir"])
    rv = CR.load_review(REPO / sm["review_file"])
    fields = CR.compute_fields(auto, rv, he_pts, if_pts)
    M, t = np.asarray(auto.meta["global"]["M"]), np.asarray(auto.meta["global"]["t"])
    m = "REVIEWED" if fields["REVIEWED"].applied else "D"
    fr = fields[m]
    mapped = CR.map_points(M, t, he_pts)
    sel = auto.he_idx
    disp = np.linalg.norm(fr.forward_displacement(mapped[sel]), axis=1)
    c = mapped[sel][np.argmax(disp)] if disp.max() > 0 else mapped[sel].mean(0)          # where the field moves most
    roi = [c[0] - crop_um / 2, c[1] - crop_um / 2, c[0] + crop_um / 2, c[1] + crop_um / 2]
    im_if, _, _ = CR.warp_he_roi(he_reader(iff), np.eye(2), np.zeros(2), roi, dst, None)
    im_a, va, ra = CR.warp_he_roi(he_reader(he), M, t, roi, dst, None)
    im_n, vn, rn = CR.warp_he_roi(he_reader(he), M, t, roi, dst, fr)
    pa = CR.nuclei_pixel_positions(he_pts, M, t, None, roi, dst)[0]
    pn = CR.nuclei_pixel_positions(he_pts, M, t, fr, roi, dst)[0]
    pif = np.column_stack([(if_pts[:, 0] - roi[0]) / dst - 0.5, (if_pts[:, 1] - roi[1]) / dst - 0.5])
    H, W = im_a.shape[:2]
    inside = lambda p: (p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)  # noqa: E731
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.4))
    ax[0].imshow(imaging.to_display_rgb(im_if, iff.kind, "nuclear")); ax[0].set_title(f"{sample} IF ROI (fixed)")
    ax[0].scatter(*pif[inside(pif)].T, s=6, facecolors="none", edgecolors="#40e0ff", lw=0.6)
    ax[1].imshow(im_a[..., :3]); ax[1].set_title("H&E Affine (inverse map) + HE nuclei (forward)")
    ax[1].scatter(*pa[inside(pa)].T, s=6, c="#ff4fa3")
    ax[2].imshow(im_n[..., :3]); ax[2].set_title(f"H&E Nonrigid {m}: {rn['status']}\nresidual max {rn['inverse_residual_max_um']:.2e} µm")
    ax[2].scatter(*pn[inside(pn)].T, s=6, c="#ffeb3b")
    diff = np.abs(im_n[..., :3].astype(float) - im_a[..., :3].astype(float)).mean(-1)
    h = ax[3].imshow(diff, cmap="magma"); ax[3].set_title(f"|Nonrigid − Affine| (max field {disp.max():.1f} µm in ROI)")
    plt.colorbar(h, ax=ax[3], fraction=0.046)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(HERE / "figures" / f"fig1_{sample}.png", dpi=110)
    return {"sample": sample, "method": m, "roi": roi, "status": rn["status"], "max_field_um_in_crop_centre_region": float(disp.max()),
            "affine_valid": float(va.mean()), "nonrigid_valid": float(vn.mean())}


if __name__ == "__main__":
    (HERE / "figures").mkdir(exist_ok=True)
    out = [one(s) for s in ("031-6_2", "031-8_6")]
    (HERE / "figures" / "figures_meta.json").write_text(json.dumps(out, indent=1, default=CR._json_default))
    print(json.dumps(out, indent=1, default=CR._json_default))
