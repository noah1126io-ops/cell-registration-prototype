"""candidate_search_map.png, reference_vs_automatic.png, c1_vs_c2_overlay.png (+ copies with the required names)."""
import json, shutil
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import FIG, footprint, side_input, store_transform
from fig_overview import thumb, warp_to_fixed

he, IF = side_input("031-8_6", "HE"), side_input("031-8_6", "IF")
hi, hu = thumb(he); ii, iu = thumb(IF)
ex = (0, hi.shape[1] * hu, hi.shape[0] * hu, 0)
Mr, tr, _ = store_transform()
B = json.load(open("baseline/8_6_auto_artifact_20261022.json")); A = json.load(open("cache/runs/auto_031-8_6_occ+cleft/artifact.json"))
cols = {"image": "#ff7f0e", "points": "#1f77b4"}

# candidate search map: before vs after
fig, ax = plt.subplots(1, 2, figsize=(18, 9))
for a, art, title in ((ax[0], B, "BEFORE: tissue+nuclear (2026-10-22) — image AMBIGUOUS 0.006"), (ax[1], A, "AFTER: occ+cleft — image UNIQUE 0.303")):
    a.imshow(hi, extent=ex)
    for c in art["candidates"]:
        fp = footprint(c["T_search"]["M"], c["T_search"]["t"], IF.extent)
        top = c["rank"] == 0
        a.plot(fp[:, 0], fp[:, 1], "-", color=cols[c["source"]], lw=2.2 if top else 0.7, alpha=1 if top else 0.45)
        if c["rank"] <= 1:
            a.text(*fp[0], f"{c['id']} z={c['search']['score']:.1f}", color=cols[c["source"]], fontsize=8)
    fp = footprint(Mr, tr, IF.extent); a.plot(fp[:, 0], fp[:, 1], "--", color="lime", lw=2.2); a.text(fp[2, 0], fp[2, 1], "manual reference", color="lime")
    a.set_title(title); a.set_xlim(-800, 2900); a.set_ylim(2900, -600)
fig.suptitle("031-8_6: C1 candidates on the H&E (orange = image search, blue = nuclei search, rank 0 thick; lime dashed = manual reference)")
fig.tight_layout(); fig.savefig(FIG / "candidate_search_map.png", dpi=75); plt.close(fig)

# reference vs automatic (after image#0, ICP result) overlays
c0 = next(c for c in A["candidates"] if c["id"] == "image#0")
fig, ax = plt.subplots(1, 3, figsize=(24, 8))
g = hi.mean(-1) / 255.0
for a, (M, t), title in ((ax[0], (Mr, tr), "IF warped by MANUAL reference (θ 15.6°, s 1.10)"),
                         (ax[1], (np.array(c0["T_search"]["M"]), np.array(c0["T_search"]["t"])), "IF warped by AUTOMATIC image#0, C1 search pose (θ 13°, s 1.00)"),
                         (ax[2], (np.array(c0["T_final"]["M"]), np.array(c0["T_final"]["t"])), "AUTOMATIC image#0 after C2 ICP")):
    w = np.nan_to_num(warp_to_fixed(ii, iu, M, t, hi.shape[:2], hu)); w = np.clip(w / max(np.percentile(w[w > 0], 99.5), 1), 0, 1)
    a.imshow(g, cmap="gray", extent=ex); a.imshow(w, extent=ex, alpha=0.5)
    fpr = footprint(Mr, tr, IF.extent); a.plot(fpr[:, 0], fpr[:, 1], "--", color="lime", lw=1.2)
    fpa = footprint(M, t, IF.extent); a.plot(fpa[:, 0], fpa[:, 1], "-", color="#ff7f0e", lw=1.2)
    a.set_xlim(250, 1950); a.set_ylim(2150, 450); a.set_title(title)
fig.suptitle("031-8_6: H&E (gray) + warped IF (colour). lime dashed = manual reference footprint, orange = shown transform (C1→C2 shift 1 µm median; to reference 55 µm)")
fig.tight_layout(); fig.savefig(FIG / "reference_vs_automatic.png", dpi=70); plt.close(fig)

# C1 vs C2 overlay, zoomed tiles: cleft network at search pose vs after ICP (both samples' accepted/selected)
fig, ax = plt.subplots(2, 2, figsize=(16, 16))
for r, (sid, path, cid) in enumerate((("031-8_6", "cache/runs/auto_031-8_6_occ+cleft/artifact.json", "image#0"), ("031-6_2", "cache/runs/auto_031-6_2_occ+cleft/artifact.json", "image#0"))):
    h2, i2 = side_input(sid, "HE"), side_input(sid, "IF"); hh, hhu = thumb(h2, 2.0); i_, iiu = thumb(i2, 2.0)
    c = next(c for c in json.load(open(path))["candidates"] if c["id"] == cid)
    for k, (T, lab) in enumerate(((c["T_search"], "C1 search pose"), (c["T_final"], "C2 after ICP"))):
        M, t = np.array(T["M"]), np.array(T["t"])
        w = np.nan_to_num(warp_to_fixed(i_, iiu, M, t, hh.shape[:2], hhu)); w = np.clip(w / max(np.percentile(w[w > 0], 99.5), 1), 0, 1)
        e2 = (0, hh.shape[1] * hhu, hh.shape[0] * hhu, 0)
        ax[r, k].imshow(hh.mean(-1) / 255.0, cmap="gray", extent=e2); ax[r, k].imshow(w, extent=e2, alpha=0.5)
        cen = M @ np.array([513.0, 513.0]) + t
        ax[r, k].set_xlim(cen[0] - 250, cen[0] + 250); ax[r, k].set_ylim(cen[1] + 250, cen[1] - 250)
        ax[r, k].set_title(f"{sid} {cid}: {lab} (500 um tile at the IF centre)")
fig.tight_layout(); fig.savefig(FIG / "c1_vs_c2_overlay.png", dpi=70); plt.close(fig)
shutil.copy(FIG / "original_image_pair_031-8_6.png", FIG / "original_image_pair.png")
shutil.copy(FIG / "feature_variants_031-8_6.png", FIG / "feature_map_comparison.png")
print("ok")
