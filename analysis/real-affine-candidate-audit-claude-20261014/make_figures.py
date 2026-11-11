"""Figures for the Claude Real affine/candidate audit (reads cache/ and frozen inputs only)."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Polygon

import audit_core as ac

C = ac.OUT / "cache"
d = ac.load_real()
corr = np.load(C / "corrected_points.npz")
aff = json.loads((C / "affine.json").read_text())
he_roi = json.loads((C / "he_roi.json").read_text())
blocks = pd.read_csv(C / "correlogram_blocks.csv")
meta = d.warp_meta
EXT = [meta["bounds_um"][0], meta["bounds_um"][2], meta["bounds_um"][1], meta["bounds_um"][3]]
SAVED, CORR, FIX, NULL = "#d1495b", "#2e86ab", "#222222", "#999999"


def save(fig, name):
    fig.savefig(ac.OUT / name, dpi=130, bbox_inches="tight")
    plt.close(fig)


def he(ax):
    ax.imshow(d.he_image, extent=EXT, origin="upper")


# 1. coordinate transform diagram ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 5.2))
ax.axis("off")
boxes = [
    (0.01, 0.62, "HE nuclei (StarDist)\nraw pixel xy, 901×901 px\n1,842 points\nspacing/crop/downsample: UNKNOWN"),
    (0.25, 0.62, "Axis flip (saved)\nflip_x: x' = 901 − x\nflip_y: none"),
    (0.47, 0.62, "Affine (saved)\nworld = p' @ Aᵀ + t\n|det| 1.207, rot −179.4°\n≈1.10 µm/px"),
    (0.71, 0.62, "Fixed world frame (µm)\nfluorescence GeoJSON\n1,979 centroids\nCRS/unit metadata: UNKNOWN"),
    (0.47, 0.12, "Diagnostic re-fit (this audit)\nseed: correlogram offsets\nrefine: production affine_icp\nΔ ≈ (−7.6, −49.9) µm, +1.24°"),
    (0.71, 0.12, "Candidate / NCC / Graph / UOT\nconsume saved-affine subset\n(1,754 × 1,830)\nno second affine applied"),
]
for x, y, t in boxes:
    ax.add_patch(FancyBboxPatch((x, y), 0.2, 0.3, boxstyle="round,pad=0.01", fc="#f4f4f4", ec="#555"))
    ax.text(x + 0.1, y + 0.15, t, ha="center", va="center", fontsize=9)
for (x0, y0), (x1, y1), c in (((0.21, 0.77), (0.25, 0.77), "k"), ((0.45, 0.77), (0.47, 0.77), "k"),
                              ((0.67, 0.77), (0.71, 0.77), "k"), ((0.81, 0.62), (0.81, 0.42), "k"),
                              ((0.57, 0.62), (0.57, 0.42), CORR)):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color=c, lw=1.6))
ax.text(0.01, 0.4, "Checks (all PASS):\n• saved points = flip→affine of raw (max err 0 µm)\n• inverse round trip 1.7e-13 px\n"
        "• subset filters reproduced exactly (1,830 / 1,754)\n• HE nuclei sit on dark HE pixels (gray 96 vs 144)\n"
        "• chirality: only this reflection gives a correlogram peak\n\nFinding: saved affine is a local ICP minimum\n"
        "~50 µm from the shared-nucleus solution", fontsize=9, va="top")
ax.set_title("Coordinate path: HE pixel → fluorescence world (saved run 20260904-161247-698fe6c9)")
save(fig, "coordinate_transform_diagram.png")

# 2. raw vs affine --------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].scatter(d.moving_raw_px[:, 0], d.moving_raw_px[:, 1], s=2, c=SAVED)
axs[0].invert_yaxis(); axs[0].set_title("HE raw pixel xy (y down)"); axs[0].set_aspect("equal")
win = (560, 760, 180, 380)
for ax, pts, name, col in ((axs[1], d.moving_affine_all, "saved affine", SAVED), (axs[2], corr["corrected_all"], "diagnostic re-fit", CORR)):
    he(ax)
    ax.scatter(d.fixed_all[:, 0], d.fixed_all[:, 1], s=40, facecolors="none", edgecolors="cyan", lw=1.3, label="fixed (fluor.)")
    ax.scatter(pts[:, 0], pts[:, 1], s=14, marker="x", c="yellow", label=f"moving ({name})")
    st = ac.nn_stats(pts[d.moving_orig_index], d.fixed)
    ax.set_xlim(win[0], win[1]); ax.set_ylim(win[2], win[3])
    ax.set_title(f"{name}: NN median {st['moving_to_fixed_nn_median_um']:.1f} µm, ≤3 µm {100*st['within_3um']:.1f}%")
    ax.legend(loc="lower left", fontsize=8)
save(fig, "raw_vs_affine.png")

# 3. ROI overlap -----------------------------------------------------------------------------
fr = pd.read_csv(C / "fixed_subset_reasons.csv"); mr = pd.read_csv(C / "moving_subset_reasons.csv")
fig, axs = plt.subplots(1, 2, figsize=(15, 7))
ax = axs[0]
ax.scatter(fr.x[fr.valid], fr.y[fr.valid], s=2, c=FIX, label="fixed subset (1,830)")
for r, col in (("outside_he_footprint", "orange"), ("within_edge_margin", "purple")):
    t = fr[fr.reason == r]; ax.scatter(t.x, t.y, s=10, c=col, label=f"fixed excluded: {r} ({len(t)})")
t = mr[~mr.valid]; ax.scatter(t.x, t.y, s=10, marker="x", c="green", label=f"moving excluded ({len(t)})")
for key, col in (("roi_saved", SAVED), ("roi_corrected", CORR)):
    poly = np.asarray(he_roi[key]["he_image_footprint_world_polygon"])
    ax.add_patch(Polygon(poly, fill=False, ec=col, lw=2, label=f"HE image footprint ({key[4:]})"))
ax.set_aspect("equal"); ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.05)); ax.set_title("Subset filter and HE footprint (world µm)")
ax = axs[1]
scan = pd.read_csv(C / "he_white_translation_scan.csv")
piv = scan.pivot(index="shift_dy", columns="shift_dx", values="fixed_in_white")
im = ax.imshow(piv.values, origin="lower", extent=[piv.columns.min(), piv.columns.max(), piv.index.min(), piv.index.max()], cmap="viridis")
plt.colorbar(im, ax=ax, label="fraction of fixed nuclei in HE white (no-tissue) pixels")
o = -np.array([aff["correction_displacement_um"]["median_dx"], aff["correction_displacement_um"]["median_dy"]])
ax.plot(0, 0, "o", c=SAVED, ms=10, label="saved (no shift)")
ax.plot(o[0], o[1], "*", c="white", ms=16, label="point-correlogram correction")
ax.set_xlabel("shift applied to fixed dx (µm)"); ax.set_ylabel("dy (µm)")
ax.set_title("Image-based check (independent of point matching):\nfixed nuclei landing in HE gaps vs translation"); ax.legend(fontsize=8)
save(fig, "roi_overlap.png")

# 4. affine residual vectors -------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(20, 6.5))
sv = blocks[blocks.condition == "saved_affine_sliding"]
ax = axs[0]
ax.scatter(d.fixed[:, 0], d.fixed[:, 1], s=1, c="#ccc")
q = ax.quiver(sv.center_x, sv.center_y, sv.refined_dx, sv.refined_dy, sv.peak_ratio, angles="xy", scale_units="xy", scale=1, cmap="plasma", width=0.004)
plt.colorbar(q, ax=ax, label="correlogram peak / background")
ax.set_aspect("equal"); ax.set_title("Local shared-nucleus displacement (fixed − moving, true scale)\n200 µm windows, 3 µm kernel")
ax = axs[1]
from scipy.spatial import cKDTree
dist, j = cKDTree(d.fixed).query(d.moving); v = d.fixed[j] - d.moving
ax.scatter(d.fixed[:, 0], d.fixed[:, 1], s=1, c="#ccc")
ax.quiver(d.moving[::3, 0], d.moving[::3, 1], v[::3, 0], v[::3, 1], angles="xy", scale_units="xy", scale=1, color=SAVED, width=0.002)
ax.set_aspect("equal"); ax.set_title("Nearest-neighbour vectors after saved affine (every 3rd)\nNN does not recover the ~50 µm shift")
ax = axs[2]
sel = (np.abs(d.moving_affine_all[:, 0] - 700) < 100) & (np.abs(d.moving_affine_all[:, 1] - 300) < 100)
lists = cKDTree(d.fixed_all).query_ball_point(d.moving_affine_all[sel], 90)
disp = np.concatenate([d.fixed_all[p] - d.moving_affine_all[sel][i] for i, p in enumerate(lists) if p])
H, xe, ye = np.histogram2d(disp[:, 0], disp[:, 1], bins=90, range=[[-90, 90], [-90, 90]])
ax.imshow(H.T, origin="lower", extent=[-90, 90, -90, 90], cmap="magma")
ax.set_title("Displacement histogram, window centred (700, 300)\npeak at ≈(−2, −48) µm; origin = saved affine"); ax.set_xlabel("dx"); ax.set_ylabel("dy")
save(fig, "affine_residual_vectors.png")

# 5. regional NCC boundary map ---------------------------------------------------------------
an = pd.read_csv(C / "ncc_anchor_table.csv")
panels = [("saved_affine", 18), ("saved_affine", 54), ("saved_affine", 72), ("corrected_affine", 18), ("null_mirror_x", 18)]
fig, axs = plt.subplots(1, 5, figsize=(26, 5.6))
for ax, (cnd, R) in zip(axs, panels):
    t = an[(an.condition == cnd) & (an.search_radius_um == R)]
    ax.scatter(d.fixed[:, 0], d.fixed[:, 1], s=1, c="#ddd")
    ax.quiver(t.x, t.y, t.best_dx, t.best_dy, np.where(t.boundary_hit, 1, 0), angles="xy", scale_units="xy", scale=1,
              cmap="coolwarm", clim=(0, 1), width=0.004)
    ax.set_aspect("equal"); ax.set_title(f"{cnd} ±{R} µm\nboundary hit {100*t.boundary_hit.mean():.1f}% of {len(t)} anchors (red)")
save(fig, "regional_ncc_boundary_map.png")

# 6. NCC window comparison ---------------------------------------------------------------------
w = pd.read_csv(ac.OUT / "ncc_window_sensitivity.csv")
w["label"] = w.condition + " ±" + w.search_radius_um.astype(int).astype(str)
fig, axs = plt.subplots(1, 3, figsize=(20, 5.5))
cols = [SAVED if c == "saved_affine" else CORR if c == "corrected_affine" else NULL for c in w.condition]
axs[0].barh(w.label, 100 * w.nucleus_boundary_hit_rate, color=cols); axs[0].set_xlabel("nuclei whose anchor offset hits the window edge (%)")
axs[0].axvline(100 * 48 / 169, ls=":", c="k"); axs[0].text(100 * 48 / 169, -0.6, " edge cells / all cells", fontsize=8)
axs[1].barh(w.label, w.best_ncc_median, color=cols, label="best"); axs[1].barh(w.label, w.peak_margin_median, color="k", alpha=0.5, label="margin")
axs[1].set_xlabel("median NCC"); axs[1].legend()
s = w[w.condition == "saved_affine"]
axs[2].plot(s.search_radius_um, 100 * s.agree_with_correlogram_within_6um_rate, "o-", c=SAVED, label="NCC offset within 6 µm of correlogram offset (%)")
axs[2].plot(s.search_radius_um, s.offset_dy_median, "s--", c="k", label="median NCC dy (µm)")
axs[2].plot(s.search_radius_um, s.neighbor_anchor_offset_diff_median_um, "^:", c="purple", label="neighbouring-anchor offset difference (µm)")
axs[2].axhline(-50, c=CORR, lw=0.8); axs[2].set_xlabel("search radius (µm)"); axs[2].legend(fontsize=8)
axs[2].set_title("Saved affine: wider window converges on the independent correlogram offset")
save(fig, "ncc_window_comparison.png")

# 7. candidate distance distributions ----------------------------------------------------------
dom = pd.read_csv(C / "domain_per_nucleus.csv", low_memory=False)
syn = pd.read_csv(ac.SEP_SRC / "per_nucleus_results.csv", low_memory=False,
                  usecols=["split", "method", "spatial_distance_um", "top1_graph_support", "scenario_id"])
syn = syn[(syn.split == "Test") & (syn.method == "Position + Graph")]
src = pd.read_csv(ac.OUT / "candidate_source_audit.csv")
ks = pd.read_csv(ac.OUT / "candidate_k_sensitivity.csv")
fig, axs = plt.subplots(1, 3, figsize=(20, 5.5))
bins = np.linspace(0, 40, 81)
for case, col in (("Real saved affine (baseline)", SAVED), ("Real corrected affine (diagnostic)", CORR), ("Real null: moving mirrored in x", NULL)):
    t = dom[(dom.case == case) & (dom.method == "Position + Graph")]
    axs[0].hist(t.spatial_distance_um, bins=bins, histtype="step", lw=2, color=col, density=True, label=case)
axs[0].hist(syn.spatial_distance_um, bins=bins, histtype="step", lw=2, ls="--", color="green", density=True, label="Real-like synthetic Test")
axs[0].set_xlabel("top-1 (Position+Graph) candidate distance (µm)"); axs[0].legend(fontsize=8)
p = src[src.putative_partner_fixed >= 0]
axs[1].hist(p.putative_partner_rank_spatial.clip(upper=60), bins=np.arange(0.5, 61.5), color=SAVED)
axs[1].axvline(30.5, c="k", ls="--"); axs[1].axvline(10.5, c="k", ls=":")
axs[1].set_title(f"Spatial rank of correlogram partner under saved affine\n(median {p.putative_partner_rank_spatial.median():.0f}; in final-30 {100*p.putative_partner_in_final30.mean():.1f}%; top-1 {100*p.putative_partner_is_top1_position_graph.mean():.2f}%)")
for cnd, col in (("saved_affine", SAVED), ("corrected_affine", CORR), ("saved_affine_ncc54", "orange"), ("null_mirror_x", NULL)):
    t = ks[ks.condition == cnd]
    axs[2].plot(t.K, t.partner_coverage_at_K, "o-", c=col, label=f"{cnd}: partner in set")
    axs[2].plot(t.K, t.top1_pg_equals_partner_rate, "x--", c=col, label=f"{cnd}: top-1 = partner")
axs[2].set_xscale("log"); axs[2].set_xticks([10, 30, 50, 100]); axs[2].set_xticklabels([10, 30, 50, 100]); axs[2].set_xlabel("K")
axs[2].legend(fontsize=7); axs[2].set_title("Candidate coverage vs selection (proxy partner, not ground truth)")
save(fig, "candidate_distance_distribution.png")

# 8. graph support map ------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(20, 6))
for ax, case in zip(axs[:2], ("Real saved affine (baseline)", "Real corrected affine (diagnostic)")):
    t = dom[(dom.case == case) & (dom.method == "Position + Graph")].sort_values("moving_index")
    sc = ax.scatter(d.moving[:, 0], d.moving[:, 1], c=t.top1_graph_support, s=8, cmap="viridis", vmin=0, vmax=1)
    ax.set_aspect("equal"); ax.set_title(f"{case}\ntop-1 Graph support median {t.top1_graph_support.median():.3f}")
plt.colorbar(sc, ax=axs[1])
b = np.linspace(0, 1, 41)
for case, col in (("Real saved affine (baseline)", SAVED), ("Real corrected affine (diagnostic)", CORR), ("Real null: moving mirrored in x", NULL)):
    t = dom[(dom.case == case) & (dom.method == "Position + Graph")]
    axs[2].hist(t.top1_graph_support, bins=b, histtype="step", lw=2, color=col, density=True, label=case)
axs[2].hist(syn.top1_graph_support, bins=b, histtype="step", lw=2, ls="--", color="green", density=True, label="Real-like synthetic Test")
axs[2].legend(fontsize=8); axs[2].set_xlabel("top-1 Graph support")
save(fig, "graph_support_map.png")

# extra: ICP capture range + HE crops ---------------------------------------------------------
cap = pd.read_csv(C / "icp_capture_range.csv")
fig, axs = plt.subplots(1, 3, figsize=(20, 6.5))
pv = cap.pivot(index="init_dy", columns="init_dx", values="within_3um")
im = axs[0].imshow(pv.values, origin="lower", extent=[-85, 85, -85, 85], cmap="viridis")
plt.colorbar(im, ax=axs[0], label="final fraction ≤3 µm")
axs[0].set_xlabel("initial translation offset dx (µm)"); axs[0].set_ylabel("dy"); axs[0].set_title("Production affine_icp capture range\n(start = saved affine + offset)")
for ax, pts, name in ((axs[1], d.fixed_all, "saved affine"), (axs[2], None, "diagnostic re-fit")):
    he(ax)
    if pts is None:
        Ms, ts, Mc, tc = d.affine_matrix, d.translation, corr["matrix"], corr["translation"]
        pts = (np.linalg.solve(Mc, (d.fixed_all - tc).T).T) @ Ms.T + ts
    ax.scatter(pts[:, 0], pts[:, 1], s=40, facecolors="none", edgecolors="cyan", lw=1.4)
    ax.scatter(d.moving_affine_all[:, 0], d.moving_affine_all[:, 1], s=14, marker="x", c="yellow")
    ax.set_xlim(0, 230); ax.set_ylim(80, 310); ax.set_title(f"Fixed nuclei on HE image, {name}\n(cyan = fluorescence, yellow = HE nuclei)")
save(fig, "icp_capture_and_he_gap_check.png")
print("figures written")
