"""Phase A input audit: image metadata, nuclei coordinate frames and image<->nuclei consistency.

Writes sample_manifest.json, coordinate_consistency.csv, cache/audit_<sample>.json,
input_image_overview.png, image_nuclei_overlay.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from scipy.ndimage import binary_dilation, shift as nd_shift  # noqa: E402
from skimage.color import rgb2hed  # noqa: E402
from skimage.draw import polygon as draw_polygon  # noqa: E402

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from oir.frames import json_safe  # noqa: E402
from oir.sample import PROJECTS, SAMPLES, load_sample, ome_info, read_level  # noqa: E402

PROVIDER = Path("/home/atsushi.doi/nagasaki_lab/cardio_vascular/05_results/_integrated/20261006_arcscape_planA_bg/work")


def auc(pos, neg, n=200000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.choice(pos, min(n, len(pos))); b = rng.choice(neg, min(n, len(neg)))
    allv = np.r_[a, b]; r = np.argsort(np.argsort(allv)) + 1
    return float((r[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def nuclear_label_mask(polys_px, shape, f):
    m = np.zeros(shape, bool)
    for p in polys_px:
        rr, cc = draw_polygon(p[:, 1] / f, p[:, 0] / f, shape)
        m[rr, cc] = True
    return m


def offset_scan(signal, mask, max_px=8):
    """Mean signal under the (shifted) nuclear mask; peak location tests image<->coordinate registration."""
    res = np.full((2 * max_px + 1, 2 * max_px + 1), np.nan)
    idx = np.argwhere(mask)
    for i, dy in enumerate(range(-max_px, max_px + 1)):
        for j, dx in enumerate(range(-max_px, max_px + 1)):
            r = np.clip(idx[:, 0] + dy, 0, signal.shape[0] - 1); c = np.clip(idx[:, 1] + dx, 0, signal.shape[1] - 1)
            res[i, j] = signal[r, c].mean()
    i, j = np.unravel_index(np.nanargmax(res), res.shape)
    return res, (j - max_px, i - max_px)


def main():
    rows, manifest, figs = [], {"samples": {}}, {}
    for name in SAMPLES:
        S = load_sample(name)
        he_tif, if_tif = S.he.frame.image_path, S.iff.frame.image_path
        hi, ii = ome_info(he_tif), ome_info(if_tif)
        prov_assets = json.loads((PROVIDER / name / "assets.json").read_text())
        prov_ref = PROVIDER / name / "he_reference.json"
        prov_ref = json.loads(prov_ref.read_text()) if prov_ref.exists() else None
        rec = {"he_image": hi, "if_image": ii, "provider_assets": prov_assets, "provider_he_reference": prov_ref}

        # ---------------- IF nuclei: GeoJSON internal consistency
        polys = S.iff.polygons_px
        cen_prop = S.iff.frame.world_to_pixel(S.iff.nuclei_um)
        def area_centroid(p):
            x, y = p[:, 0], p[:, 1]
            cr = x[:-1] * y[1:] - x[1:] * y[:-1]
            a = cr.sum() / 2
            return np.array([((x[:-1] + x[1:]) * cr).sum(), ((y[:-1] + y[1:]) * cr).sum()]) / (6 * a)
        cen_poly = np.array([area_centroid(p) for p in polys])
        d_prop_poly = np.linalg.norm(cen_prop - cen_poly, axis=1)
        gj1 = json.loads((PROJECTS / name / f"{name}_seg_seqif_level1.geojson").read_text())
        c1 = {f["properties"]["segment_id"]: (f["properties"]["centroid_x"], f["properties"]["centroid_y"]) for f in gj1["features"]}
        c1 = np.array([c1[i] for i in S.iff.nuclei_ids])
        d_l1 = np.linalg.norm(c1 * 2 - cen_prop, axis=1)
        H0, W0 = S.iff.frame.shape_px
        inb_if = np.mean((cen_prop[:, 0] >= 0) & (cen_prop[:, 0] < W0) & (cen_prop[:, 1] >= 0) & (cen_prop[:, 1] < H0))
        allv = np.vstack(polys)
        rows += [
            dict(sample=name, check="IF GeoJSON centroid property vs polygon area centroid", value=float(np.median(d_prop_poly)), unit="px L0 (median)", p95=float(np.percentile(d_prop_poly, 95)), verdict="consistent" if np.median(d_prop_poly) < 2 else "CHECK"),
            dict(sample=name, check="IF level1 GeoJSON x2 vs level0 centroid", value=float(np.median(d_l1)), unit="px L0 (median)", p95=float(np.percentile(d_l1, 95)), verdict="level1 = level0/2" if np.median(d_l1) < 2 else "CHECK"),
            dict(sample=name, check="IF nuclei centroids inside IF image", value=float(inb_if), unit="fraction", p95=np.nan, verdict="all inside" if inb_if == 1 else "partial"),
            dict(sample=name, check="IF polygon vertex range x/y", value=float(allv[:, 0].max()), unit=f"px L0 (x max; y max {allv[:, 1].max():.0f}; min {allv.min():.0f}); image {W0}x{H0}", p95=np.nan, verdict="within image" if allv.min() >= -1 and allv[:, 0].max() <= W0 + 1 and allv[:, 1].max() <= H0 + 1 else "CHECK"),
        ]

        # ---------------- IF image: which channel carries nuclear signal (test, do not assume)
        fq = 2
        img_if = read_level(if_tif, 1).astype(float)          # 0.2004 um/px
        mask = nuclear_label_mask(polys, img_if.shape[:2], fq)
        ring = binary_dilation(mask, iterations=10) & ~mask
        chans = {"R (Desmoplakin)": img_if[..., 0], "G (Dystrophin1)": img_if[..., 1], "B (Hoechst+Desmoplakin)": img_if[..., 2],
                 "B-R (Hoechst estimate)": np.clip(img_if[..., 2] - img_if[..., 0], 0, None)}
        ch_stats = {}
        for k, v in chans.items():
            ch_stats[k] = {"mean_in_nuclei": float(v[mask].mean()), "mean_ring": float(v[ring].mean()),
                           "auc_nuclei_vs_ring": auc(v[mask], v[ring])}
            rows.append(dict(sample=name, check=f"IF channel {k}: AUC nuclei-vs-ring (10 px L1)", value=ch_stats[k]["auc_nuclei_vs_ring"], unit="AUC", p95=np.nan,
                             verdict="nuclear-like" if ch_stats[k]["auc_nuclei_vs_ring"] > 0.75 else "not nuclear"))
        hoechst = chans["B-R (Hoechst estimate)"]
        scan, off = offset_scan(hoechst, mask)
        rows.append(dict(sample=name, check="IF B-R signal under nuclear mask: offset of maximum", value=float(np.hypot(*off) * fq * S.iff.frame.pixel_um),
                         unit=f"um (dx,dy)={off} px L1", p95=np.nan, verdict="GeoJSON registered to image" if max(map(abs, off)) <= 1 else "OFFSET"))
        rec["if_channel_stats"] = ch_stats; rec["if_offset_peak_px_L1"] = off

        # ---------------- HE: image + StarDist coordinates
        fh = 2
        he1 = read_level(he_tif, 1)
        hed = rgb2hed(he1)[..., 0]
        hp = S.he.frame.world_to_pixel(S.he.nuclei_um) / fh
        Hh, Wh = he1.shape[:2]
        inb_he = np.mean((hp[:, 0] >= 0) & (hp[:, 0] < Wh) & (hp[:, 1] >= 0) & (hp[:, 1] < Hh))
        hm = np.zeros(he1.shape[:2], bool)
        r = np.clip(np.rint(hp[:, 1]).astype(int), 0, Hh - 1); c = np.clip(np.rint(hp[:, 0]).astype(int), 0, Wh - 1)
        hm[r, c] = True
        hm = binary_dilation(hm, iterations=3)
        scan_he, off_he = offset_scan(hed, hm)
        rows += [dict(sample=name, check="HE StarDist centroids inside HE image", value=float(inb_he), unit="fraction", p95=np.nan, verdict="all inside" if inb_he == 1 else "partial"),
                 dict(sample=name, check="HE hematoxylin under StarDist centroids: offset of maximum", value=float(np.hypot(*off_he) * fh * S.he.frame.pixel_um),
                      unit=f"um (dx,dy)={off_he} px L1", p95=np.nan, verdict="xy order / frame consistent" if max(map(abs, off_he)) <= 1 else "OFFSET")]

        # ---------------- provider transform vs seg_he.geojson
        gjh = json.loads((PROJECTS / name / f"{name}_seg_he.geojson").read_text())
        ch = {f["properties"]["segment_id"]: (f["properties"]["centroid_x"], f["properties"]["centroid_y"]) for f in gjh["features"]}
        chp = np.array([ch[i] for i in S.iff.nuclei_ids])
        A = np.c_[cen_prop, np.ones(len(cen_prop))]; X = np.linalg.lstsq(A, chp, rcond=None)[0]
        res = np.linalg.norm(A @ X - chp, axis=1)
        src = prov_ref if prov_ref else prov_assets.get("he")
        cands = {}
        if src:
            for lab, sc in {"a_x/10 per IF L0 px": src["a_x"] / 10}.items():
                p_ = np.c_[cen_prop[:, 0] * sc + src["b_x"], cen_prop[:, 1] * sc * src["a_y"] / src["a_x"] + src["b_y"]]
                cands[lab] = float(np.median(np.linalg.norm(p_ - chp, axis=1)))
        rows += [dict(sample=name, check="seg_he.geojson = affine(seqIF L0 px) fit residual", value=float(np.median(res)), unit=f"HE px (median); M diag={X[0,0]:.5f},{X[1,1]:.5f} off-diag={X[1,0]:.1e},{X[0,1]:.1e}", p95=float(np.percentile(res, 95)), verdict="pure scale+translation" if abs(X[1, 0]) < 1e-4 and abs(X[0, 1]) < 1e-4 else "general affine"),
                 dict(sample=name, check="seg_he scale vs pixel-spacing ratio (IF/HE)", value=float(X[0, 0] / (S.iff.frame.pixel_um / S.he.frame.pixel_um)), unit="ratio (1 = physical spacing only)", p95=np.nan, verdict="provider scale %.4f" % (src["scale"] if src else np.nan))]
        for lab, v in cands.items():
            rows.append(dict(sample=name, check=f"provider json reproduces seg_he ({lab})", value=v, unit="HE px (median)", p95=np.nan, verdict="exact" if v < 1 else ("approximately (convention of a_* UNKNOWN)" if v < 20 else "no")))
        M_prov_px = X[:2].T; t_prov_px = X[2]
        rec["provider_seg_he_affine_ifpx_to_hepx"] = {"M": M_prov_px, "t": t_prov_px}
        # IF footprint in HE under provider placement
        corners = np.array([[0, 0], [W0, 0], [W0, H0], [0, H0]], float) @ M_prov_px.T + t_prov_px
        Hh0, Wh0 = S.he.frame.shape_px
        g = np.stack(np.meshgrid(np.linspace(0, W0, 60), np.linspace(0, H0, 60)), -1).reshape(-1, 2) @ M_prov_px.T + t_prov_px
        frac_in = np.mean((g[:, 0] >= 0) & (g[:, 0] < Wh0) & (g[:, 1] >= 0) & (g[:, 1] < Hh0))
        rows.append(dict(sample=name, check="IF footprint inside HE image under provider placement", value=float(frac_in), unit="area fraction", p95=np.nan,
                         verdict="IF field contained in HE field" if frac_in > 0.99 else "partial"))
        rows += [dict(sample=name, check="HE physical extent", value=S.he.frame.extent_um()[0], unit=f"um x (y {S.he.frame.extent_um()[1]:.1f})", p95=np.nan, verdict="image size x spacing"),
                 dict(sample=name, check="IF physical extent", value=S.iff.frame.extent_um()[0], unit=f"um x (y {S.iff.frame.extent_um()[1]:.1f})", p95=np.nan, verdict="image size x spacing"),
                 dict(sample=name, check="Stage/crop origin in OME", value=float(hi["has_stage_origin"] or ii["has_stage_origin"]), unit="bool", p95=np.nan, verdict="UNKNOWN: frame-local origins only")]

        manifest["samples"][name] = {
            "he": {**S.he.frame.to_dict(), "description": hi["description"], "channels": "RGB brightfield H&E (uint8)",
                   "nuclei": {"n": int(len(S.he.nuclei_um)), "source": S.he.nuclei_source, "coords": "HE level-0 px (x right, y down); world = px * pixel_um"}},
            "if": {**S.iff.frame.to_dict(), "description": ii["description"],
                   "channels": "uint8 RGB display rendering (provider contrast in assets.json): R=Desmoplakin, G=Dystrophin1, B=Hoechst+Desmoplakin; raw channel intensities not available",
                   "nuclear_channel": "no pure nuclear channel; B-R used as Hoechst estimate after empirical test (coordinate_consistency.csv)",
                   "nuclei": {"n": int(len(S.iff.nuclei_um)), "source": S.iff.nuclei_source, "ids": "segment_id (float)", "coords": "seqIF level-0 px; level1 file = level0/2"}},
            "relationship": {"same_section": name == "031-6_2", "provider_note": hi["description"],
                             "provider_reference_transform": "image NCC, rotation fixed 0, scale grid 0.975-0.995 (he_scan.json)" if name == "031-6_2" else "approximate field-centre placement, not registered",
                             "use_in_this_study": "dual-image same-section pair; provider transform = non-independent reference (not ground truth)" if name == "031-6_2"
                             else "serial-section pair (HE 031-7 vs IF 031-8, spacing UNKNOWN); region-correspondence test, no cell-identity expectation"},
            "he_nuclei_detection": json.loads((OUT.parent / "he-stardist-seqif-registration-20261019/out" / name / "detection_meta.json").read_text()),
        }
        (OUT / "cache").mkdir(exist_ok=True)
        (OUT / "cache" / f"audit_{name}.json").write_text(json.dumps(json_safe(rec), indent=2))
        figs[name] = dict(he3=read_level(he_tif, 3), if3=read_level(if_tif, 3), if1=img_if, he1=he1, mask=mask, hp=hp, corners=corners,
                          polys=polys, scan=scan, scan_he=scan_he, if_pts=cen_prop)

    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT / "coordinate_consistency.csv", index=False)
    manifest["generated"] = "2026-10-19"; manifest["note"] = "Frame origins are frame-local (no stage coordinates in OME). Point-cloud bounding boxes are not used as imaged extents."
    (OUT / "sample_manifest.json").write_text(json.dumps(json_safe(manifest), indent=2))

    # ---------------- figures
    fig, ax = plt.subplots(2, 2, figsize=(16, 16))
    for i, name in enumerate(SAMPLES):
        F = figs[name]
        ax[i, 0].imshow(F["he3"]); ax[i, 0].plot(*np.vstack([F["corners"], F["corners"][:1]]).T / 8, c="lime", lw=2)
        ax[i, 0].set_title(f"{name} HE ({manifest['samples'][name]['he']['extent_um'] if 'extent_um' in manifest['samples'][name]['he'] else ''})\n"
                           f"green = IF field under provider placement ({'registered' if name == '031-6_2' else 'APPROXIMATE, not registered'})", fontsize=10)
        ax[i, 1].imshow(F["if3"]); ax[i, 1].set_title(f"{name} seqIF (R Desmoplakin, G Dystrophin1, B Hoechst+Desmoplakin)", fontsize=10)
        for a in ax[i]:
            a.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "input_image_overview.png", dpi=60); plt.close(fig)

    fig, ax = plt.subplots(2, 3, figsize=(24, 16))
    for i, name in enumerate(SAMPLES):
        F = figs[name]
        cy, cx = np.median(F["if_pts"], 0)[::-1] / 2
        r0, c0 = int(cy - 300), int(cx - 300)
        ax[i, 0].imshow(F["if1"][r0:r0 + 600, c0:c0 + 600].astype(np.uint8), extent=(c0, c0 + 600, r0 + 600, r0))
        for p in F["polys"]:
            q = p / 2
            if (np.abs(q.mean(0) - [cx, cy]) < 320).all():
                ax[i, 0].plot(*q.T, c="yellow", lw=0.7)
        ax[i, 0].set_title(f"{name} IF L1 + IF GeoJSON nuclei (yellow), 600 px = {600 * 0.2004:.0f} um")
        he = F["he1"]; hp = F["hp"]
        hc = np.median(hp, 0)
        r0, c0 = int(hc[1] - 300), int(hc[0] - 300)
        ax[i, 1].imshow(he[r0:r0 + 600, c0:c0 + 600], extent=(c0, c0 + 600, r0 + 600, r0))
        sel = (np.abs(hp - hc) < 300).all(1)
        ax[i, 1].scatter(*hp[sel].T, s=10, c="lime", marker="+")
        ax[i, 1].set_title(f"{name} HE L1 + StarDist centroids, 600 px = {600 * 0.3774:.0f} um")
        ax[i, 2].imshow(F["scan"], extent=(-8.5, 8.5, 8.5, -8.5)); ax[i, 2].set_title("IF B-R mean under shifted nuclear mask (px L1)")
        for a in ax[i, :2]:
            a.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "image_nuclei_overlay.png", dpi=60); plt.close(fig)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
