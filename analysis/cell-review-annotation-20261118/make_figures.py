"""Visual check (no browser): the same crops / contours / candidates as the comparison panel, drawn with matplotlib.

For 031-6_2 and 031-8_6: three H&E nuclei near the ROI centre (B2 matched if any), left = H&E crop (H&E frame, undeformed),
right = IF crop around the Global-predicted position (IF frame, undeformed) with the top-5 candidate contours."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.registration_explorer import catalog, imaging, results, scene  # noqa: E402
from src.registration_research import cell_review as CR  # noqa: E402
from src.registration_research import nuclear_contours as NC  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
W = 40.0
meta = []
for sample in ("031-6_2", "031-8_6"):
    S = catalog.load_manifest()[sample]
    recs = [r for r in results.load_records(S) if r.M is not None]
    rec = next((r for r in recs if r.method == "WF"), recs[0])
    M, t, _ = results.oriented(rec.M, rec.t, rec.fixed, "IF")
    he, he_ids = catalog.load_nuclei(S.modality("HE"))
    iff, if_ids = catalog.load_nuclei(S.modality("IF"))
    hc = NC.load_contours(S.modality("HE"), he, he_ids)
    ic = NC.load_contours(S.modality("IF"), iff, if_ids)
    mapped = CR.map_points(M, t, he)
    roi = CR.default_roi(mapped, iff, S.modality("IF").extent())
    c = np.array([(roi[0] + roi[2]) / 2, (roi[1] + roi[3]) / 2])
    small = [c[0] - 90, c[1] - 90, c[0] + 90, c[1] + 90]
    auto = CR.compute_auto(he, he_ids, iff, if_ids, M, t, roi=small, if_extent=S.modality("IF").extent(), meta={"sample_id": sample})
    tab = auto.table.copy()
    tab["d"] = np.hypot(tab.mapped_x - c[0], tab.mapped_y - c[1])
    tab = tab.sort_values(["status", "d"], key=lambda s: s.map({"matched": 0, "ambiguous": 1, "unmatched": 2}) if s.name == "status" else s)
    picks = tab["he_index"].to_numpy(int)[:3]
    fig, ax = plt.subplots(len(picks), 2, figsize=(9, 4.4 * len(picks)))
    for r, h in enumerate(picks):
        for col, (key, ctr, cont, pts, ids) in enumerate((("HE", he[h], hc, he, he_ids), ("IF", mapped[h], ic, iff, if_ids))):
            a = ax[r, col]
            mod = S.modality(key)
            got = scene.roi_image(mod, np.eye(2), np.zeros(2), (ctr[0], ctr[1], W), W / 500, "rgb", max_px=900)
            rgb, um, _, lo = got
            c0 = imaging.level_corner_um(mod.pixel_um)
            x0, y0 = c0 + lo[0] * um, c0 + lo[1] * um
            a.imshow(rgb, extent=[x0, x0 + rgb.shape[1] * um, y0 + rgb.shape[0] * um, y0])
            near = cont.query([ctr[0] - W / 2, ctr[1] - W / 2, ctr[0] + W / 2, ctr[1] + W / 2])
            for j in near:
                p = cont.polygon(j)
                a.plot(*np.vstack([p, p[:1]]).T, color="#b0bec5", lw=0.6, alpha=0.7)
            if key == "HE":
                p = cont.polygon(h); a.plot(*np.vstack([p, p[:1]]).T, color="#ffeb3b", lw=2.2)
                a.plot(*he[h], "o", color="#ffeb3b", ms=4)
                a.set_title(f"H&E {CR._scalar(he_ids[h])} ({auto.table.set_index('he_index').loc[h, 'status']})  H&E frame", fontsize=9)
            else:
                cands = auto.candidates[auto.candidates.he_index == h]
                for _, cr in cands.iterrows():
                    f = int(cr.if_index); p = cont.polygon(f)
                    top = cr["rank"] == 1
                    a.plot(*np.vstack([p, p[:1]]).T, color="#ff40ff" if top else "white", lw=2.4 if top else 1.2, ls="-" if top else "--")
                    a.text(iff[f, 0] + 0.8, iff[f, 1] - 0.8, f"#{int(cr['rank'])}", color="#ff40ff" if top else "white", fontsize=8)
                a.plot(*mapped[h], "+", color="#ffeb3b", ms=12, mew=2)
                a.set_title("IF (undeformed, IF frame): top-5 candidates, + = Global prediction", fontsize=9)
            a.set_xlim(ctr[0] - W / 2, ctr[0] + W / 2); a.set_ylim(ctr[1] + W / 2, ctr[1] - W / 2)
            a.set_xlabel("µm"); a.tick_params(labelsize=7)
            meta.append({"sample": sample, "he_index": int(h), "section": key, "level_um_per_px": float(um),
                         "contour_contains_centre": bool(NC._inside(cont.polygon(h if key == "HE" else int(auto.candidates[auto.candidates.he_index == h].iloc[0].if_index)),
                                                                     he[h] if key == "HE" else iff[int(auto.candidates[auto.candidates.he_index == h].iloc[0].if_index)]))})
    fig.suptitle(f"{sample}: comparison panel (display check, not evidence of identity)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"fig_compare_{sample}.png", dpi=110)
    plt.close(fig)
(OUT / "figures_meta.json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta, indent=1))
