"""Result figures + research registration previews (preview only; not a Workflow D output)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from aggregate import T_he_to_if, load_real, status  # noqa: E402
from oir.frames import Affine2D  # noqa: E402
from oir.pipeline import footprint  # noqa: E402
from oir.sample import load_sample, read_level  # noqa: E402
from oir.search import pose_matrix  # noqa: E402
from run_real import reference_he_to_if  # noqa: E402

SAMPLES = ("031-6_2", "031-8_6")
COL = {"E0_HEmov": "#d62728", "E0x_HEfix": "#1f77b4", "E2_HEfix": "#2ca02c", "E2nuc_HEfix": "#17becf", "E3zsum_HEfix": "#9467bd", "E1he_HEfix": "#ff7f0e"}


def he_world_poly_of_if(T_he_to_if_, iff_ext):
    return T_he_to_if_.inverse().apply(footprint(iff_ext))


def cand_poly_he(r, cand, iff_ext):
    """Candidate pose (moving->fixed) -> IF footprint in HE world."""
    A = pose_matrix(cand["chirality"], cand["theta"], cand["scale"]); t = np.asarray(cand["t"])
    T = Affine2D(A, t, r["moving"], r["fixed"])
    T = T if r["fixed"] == "IF" else T.inverse()      # HE->IF
    return he_world_poly_of_if(T, iff_ext)


def warp_rgb(src_rgb, src_um, T_dst_to_src: Affine2D, dst_shape, dst_um):
    """Inverse mapping: for each destination pixel take the source pixel (bilinear, outside = 0)."""
    yy, xx = np.mgrid[0:dst_shape[0], 0:dst_shape[1]]
    w = np.c_[xx.ravel(), yy.ravel()] * dst_um
    s = T_dst_to_src.apply(w) / src_um
    out = np.stack([map_coordinates(src_rgb[..., c].astype(np.float32), [s[:, 1], s[:, 0]], order=1, cval=0.0) for c in range(3)], -1)
    valid = (s[:, 0] >= 0) & (s[:, 0] <= src_rgb.shape[1] - 1) & (s[:, 1] >= 0) & (s[:, 1] <= src_rgb.shape[0] - 1)
    return out.reshape(*dst_shape, 3), valid.reshape(dst_shape)


def main():
    real = load_real()
    df = pd.read_csv(OUT / "mode_comparison.csv")
    S = {s: load_sample(s) for s in SAMPLES}
    he3 = {s: read_level(S[s].he.frame.image_path, 3) for s in SAMPLES}
    he3_um = {s: S[s].he.frame.pixel_um * S[s].he.frame.pyramid_downsamples[3] for s in SAMPLES}

    # ---------------- region_search_candidates.png
    fig, ax = plt.subplots(1, 2, figsize=(22, 11))
    for a, s in zip(ax, SAMPLES):
        a.imshow(he3[s], extent=(0, he3[s].shape[1] * he3_um[s], he3[s].shape[0] * he3_um[s], 0))
        ext = S[s].iff.frame.extent_um()
        ref = reference_he_to_if(s)
        p = he_world_poly_of_if(ref, ext); a.plot(*np.vstack([p, p[:1]]).T, "w--", lw=2, label="provider placement" + (" (registered)" if s == "031-6_2" else " (APPROXIMATE)"))
        for c, col in COL.items():
            r = real.get((s, c))
            if not r or not r["search"].get("top"):
                continue
            if r["method"] == "E0":
                pp = he_world_poly_of_if(T_he_to_if(r), ext)
                a.plot(*np.vstack([pp, pp[:1]]).T, c=col, lw=2, label=f"{c} final (A2)")
                continue
            pt = cand_poly_he(r, r["search"]["top"], ext)
            a.plot(*np.vstack([pt, pt[:1]]).T, c=col, lw=2.2, label=f"{c} top ({r['search']['verdict']}, m={r['search']['margin']:.2f})")
            for k, cd in enumerate(r["search"]["fine"][1:4]):
                pc = cand_poly_he(r, cd, ext)
                a.plot(*np.vstack([pc, pc[:1]]).T, c=col, lw=0.9, ls=":", label=f"{c} candidates 2-4" if k == 0 else None)
        a.set_xlim(-300, he3[s].shape[1] * he3_um[s] + 300); a.set_ylim(he3[s].shape[0] * he3_um[s] + 300, -300)
        a.set_title(f"{s}: IF field candidates in the HE frame (um)"); a.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "region_search_candidates.png", dpi=55); plt.close(fig)

    # ---------------- previews + before/after overlays (IF frame, 1 um/px)
    fig, ax = plt.subplots(2, 4, figsize=(28, 14))
    for i, s in enumerate(SAMPLES):
        if_rgb = read_level(S[s].iff.frame.image_path, 3)            # 0.80 um/px
        if_um = S[s].iff.frame.pixel_um * S[s].iff.frame.pyramid_downsamples[3]
        he_rgb = read_level(S[s].he.frame.image_path, 2); he_um = S[s].he.frame.pixel_um * S[s].he.frame.pyramid_downsamples[2]
        dst_um = 1.0; ext = S[s].iff.frame.extent_um(); shp = (int(ext[1] / dst_um), int(ext[0] / dst_um))
        before = reference_he_to_if(s)                          # provider placement
        r_after = real[(s, "E0x_HEfix")]
        st, acc, reasons, fsrc = status(real[(s, "E3zsum_HEfix")], real)
        after = T_he_to_if(real[(s, fsrc)]) if fsrc else None
        for j, (lab, T) in enumerate((("provider placement" + ("" if s == "031-6_2" else " (APPROXIMATE)"), before), (f"final ({fsrc or 'NONE ACCEPTED'})", after))):
            a = ax[i, 2 * j]
            ifw, _ = warp_rgb(if_rgb, if_um, Affine2D.identity("IF"), shp, dst_um)
            if T is None:
                a.text(0.5, 0.5, "no accepted transform", transform=a.transAxes, ha="center"); a.axis("off"); continue
            hew, valid = warp_rgb(he_rgb, he_um, T.inverse(), shp, dst_um)     # HE resampled into IF frame (inverse map)
            a.imshow(np.clip(hew / 255, 0, 1)); a.set_title(f"{s} HE resampled into IF frame: {lab}", fontsize=9)
            b = ax[i, 2 * j + 1]
            g = 1 - hew.mean(-1) / 255.0
            nuc = np.clip(ifw[..., 2] - ifw[..., 0], 0, None); nuc = nuc / max(np.percentile(nuc, 99.5), 1)
            comp = np.stack([np.clip(g / max(np.percentile(g[valid], 99.5), 1e-6), 0, 1), np.clip(nuc, 0, 1), np.zeros_like(g)], -1)
            comp[~valid] = 0.15
            b.imshow(comp[300:700, 300:700]); b.set_title("zoom 400 um: red = HE darkness, green = IF Hoechst est. (yellow = overlap)", fontsize=9)
        for a in ax[i]:
            a.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "registration_overlay_before_after.png", dpi=50); plt.close(fig)

    # research previews (only for accepted transforms; uncertain ones are labelled)
    (OUT / "previews").mkdir(exist_ok=True)
    for s in SAMPLES:
        for c in ("E0x_HEfix", "E2_HEfix", "E3zsum_HEfix"):
            r = real.get((s, c))
            if not r:
                continue
            st, acc, reasons, fsrc = status(r, real)
            if_rgb = read_level(S[s].iff.frame.image_path, 3); if_um = S[s].iff.frame.pixel_um * 8
            he_rgb = read_level(S[s].he.frame.image_path, 2); he_um = S[s].he.frame.pixel_um * 4
            ext = S[s].iff.frame.extent_um(); shp = (int(ext[1]), int(ext[0]))
            T = T_he_to_if(r)
            hew, valid = warp_rgb(he_rgb, he_um, T.inverse(), shp, 1.0)
            ifw, _ = warp_rgb(if_rgb, if_um, Affine2D.identity("IF"), shp, 1.0)
            fig, ax2 = plt.subplots(1, 3, figsize=(24, 8))
            ax2[0].imshow(np.clip(ifw / 255, 0, 1)); ax2[0].set_title("IF (target frame, 1 um/px)")
            ax2[1].imshow(np.clip(hew / 255, 0, 1)); ax2[1].set_title("HE resampled into IF frame (inverse map, bilinear, outside=0)")
            cb = ((np.arange(shp[0])[:, None] // 100 + np.arange(shp[1])[None] // 100) % 2).astype(bool)
            mix = np.where(cb[..., None], hew / 255, ifw / 255); ax2[2].imshow(np.clip(mix, 0, 1)); ax2[2].set_title("checkerboard 100 um")
            tag = "ACCEPTED" if acc else "NOT ACCEPTED - " + ";".join(x for x in st if x != "IMAGE_AVAILABLE")
            fig.suptitle(f"RESEARCH PREVIEW {s} {c}: {tag}", color="green" if acc else "red", fontsize=14)
            for a in ax2:
                a.axis("off")
            fig.tight_layout(); fig.savefig(OUT / "previews" / f"preview_{s}_{c}.png", dpi=50); plt.close(fig)

    # ---------------- point_only_vs_image_assisted.png
    cfgs = ["E0_HEmov", "E0x_HEfix", "E1he_HEfix", "E1if_HEfix", "E2_HEfix", "E2nuc_HEfix", "E2tn_HEfix", "E3zsum_HEfix", "E3zmin_HEfix", "E3rerank_HEfix"]
    metrics = [("le_3um", "<=3 um proximity (IF nuclei)"), ("nn_median_um", "NN median (um, log)"), ("img_nuclear_corr", "nuclear map corr (IF frame)"), ("search_pose_le_3um", "<=3 um at search pose (before ICP)")]
    fig, ax = plt.subplots(2, 4, figsize=(28, 11))
    for i, s in enumerate(SAMPLES):
        d = df[(df["sample"] == s) & df["config"].isin(cfgs)].set_index("config").reindex(cfgs)
        for j, (m, lab) in enumerate(metrics):
            a = ax[i, j]
            v = d[m].astype(float)
            cols = ["#2ca02c" if acc is True else "#d62728" for acc in d["accepted"]]
            a.bar(range(len(cfgs)), v, color=cols)
            if m == "le_3um":
                a.plot(range(len(cfgs)), d["null_le_3um"], "k_", ms=18, label="mirror null"); a.legend()
            if m == "nn_median_um":
                a.set_yscale("log")
            a.set_xticks(range(len(cfgs))); a.set_xticklabels(cfgs, rotation=60, ha="right", fontsize=8)
            a.set_title(f"{s}: {lab}\n(green = accepted, red = not accepted)", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "point_only_vs_image_assisted.png", dpi=55); plt.close(fig)

    syn_p = OUT / "synthetic_summary.csv"
    if syn_p.exists():
        sm = pd.read_csv(syn_p)
        fig, ax = plt.subplots(1, 3, figsize=(24, 6))
        conds = ["q0.8", "q0.5", "q0.3", "absent_q0.5"]; meths = ["E0", "E0x", "E1", "E2", "E3"]
        for a, (m, lab) in zip(ax, (("region_found", "region found (truth, median err <= 20 um)"), ("correct_accept", "accepted AND correct (rms <= 5 um)"),
                                     ("accepted", "accepted (absent: = false accept)"))):
            for k, me in enumerate(meths):
                v = [sm[(sm.condition == c) & (sm.method == me)][m].mean() for c in conds]
                a.bar(np.arange(len(conds)) + k * 0.16, v, 0.16, label=me)
            a.set_xticks(np.arange(len(conds)) + 0.32); a.set_xticklabels(conds); a.set_ylim(0, 1.05); a.set_title(lab); a.legend()
        fig.tight_layout(); fig.savefig(OUT / "synthetic_method_comparison.png", dpi=60); plt.close(fig)

    # ---------------- failure_and_fallback_examples.png
    fig, ax = plt.subplots(1, 3, figsize=(27, 9))
    s = "031-6_2"; r = real[(s, "E0_HEmov")]
    a = ax[0]; a.imshow(he3[s], extent=(0, he3[s].shape[1] * he3_um[s], he3[s].shape[0] * he3_um[s], 0))
    p = he_world_poly_of_if(T_he_to_if(r), S[s].iff.frame.extent_um()); a.plot(*np.vstack([p, p[:1]]).T, c="r", lw=2, label="E0 (existing A2) final")
    p = he_world_poly_of_if(T_he_to_if(real[(s, "E0x_HEfix")]), S[s].iff.frame.extent_um()); a.plot(*np.vstack([p, p[:1]]).T, c="b", lw=2, label="E0x final")
    a.legend(); a.set_title(f"{s}: existing A2 fails under field-size mismatch (<=3um {df[(df['sample']==s)&(df.config=='E0_HEmov')].le_3um.iloc[0]:.3f})")
    a.set_xlim(-600, 2700); a.set_ylim(2700, -600)
    for a, s in zip(ax[1:], SAMPLES):
        d = df[(df["sample"] == s) & df["config"].str.startswith("NC")]
        txt = [f"{s} negative controls (moving side tile-shuffled 128 um):"]
        for _, x in d.iterrows():
            txt.append(f"{x.config}: verdict {x.search_verdict}, <=3um {x.le_3um:.3f} (null {x.null_le_3um:.3f}), status {x.status}, accepted {x.accepted}")
        d2 = df[(df["sample"] == s) & ~df["config"].str.startswith("NC")]
        txt.append(""); txt.append("fallback decisions:")
        for _, x in d2.iterrows():
            if not x.accepted:
                txt.append(f"{x.config}: {x.status} -> final from {x.final_transform_source}; {str(x.failure_reasons)[:110]}")
        a.text(0.0, 1.0, "\n".join(txt), va="top", family="monospace", fontsize=8, transform=a.transAxes); a.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "failure_and_fallback_examples.png", dpi=55); plt.close(fig)
    print("figures done")


if __name__ == "__main__":
    main()
