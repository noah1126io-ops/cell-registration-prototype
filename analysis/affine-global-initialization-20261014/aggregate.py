"""Aggregate cache/*.json into the required CSV/JSON deliverables and figures."""
from __future__ import annotations

import glob
import json
import platform
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import init_lib as L

C = L.CACHE
NAMES = {"A0": "Existing Centroid + ICP", "A1": "Correlogram + ICP", "A2": "Density correlation + ICP", "A3": "Multi-start + ICP"}
COL = {"A0": "#d1495b", "A1": "#2e86ab", "A2": "#e3a72f", "A3": "#3b8b5a"}
SCALAR = ("objective", "runtime_s", "peak_mem_mb", "n_hypotheses_scored", "n_refined", "detector_status", "overlap_moving_fraction",
          "inlier3", "inlier3_null", "inlier_over_null", "block_consistency", "basin_margin", "heldout_inlier3",
          "transform_rms_error_um", "transform_max_error_um", "translation_error_center_um", "rotation_error_deg", "chirality_correct",
          "scale_error_pct", "nuclei_coord_error_median_um", "true_pair_distance_median_um", "true_pairs_within3", "outcome")


def flat(case, m, extra):
    row = {"case": case, "method_key": m["method_key"], "method": NAMES[m["method_key"]], **extra}
    row.update({k: m.get(k) for k in SCALAR})
    d = m.get("diagnostics", {})
    for k in ("peak_z", "peak_ratio", "second_peak_z", "peak_ncc", "peak_robust_z", "second_peak_ncc", "best_short_error", "second_short_error"):
        if k in d:
            row[k] = d[k]
    return row


# ----------------------------------------------------------------------------- synthetic
syn_rows = []
for f in sorted(glob.glob(str(C / "synthetic_*.json"))):
    d = json.loads(open(f).read())
    for m in d["methods"]:
        r = flat(d["case"], m, {"condition": d["condition"], "seed": d["seed"], "truth_basin_objective": d["truth_basin_objective"],
                                **{f"param_{k}": v for k, v in d["params"].items() if k in ("offset_um", "fov_rot_deg", "reflect", "mov_w_px", "mov_h_px", "q_shared", "det_moving", "n_moving", "shared_fraction_of_moving")}})
        r["objective_minus_truth_basin"] = (m["objective"] - d["truth_basin_objective"])
        syn_rows.append(r)
syn = pd.DataFrame(syn_rows)
syn["detector_pass"] = syn.detector_status == "PASS"
syn.to_csv(L.OUT / "synthetic_affine_accuracy.csv", index=False)

# ----------------------------------------------------------------------------- controls
ctl_rows = []
for f in sorted(glob.glob(str(C / "control_*.json"))):
    d = json.loads(open(f).read())
    for m in d["methods"]:
        r = flat(d["case"], m, {"truth_basin_objective": d["truth_basin_objective"]})
        ctl_rows.append(r)
ctl = pd.DataFrame(ctl_rows)
ctl["false_positive_pass"] = ctl.detector_status == "PASS"
ctl.to_csv(L.OUT / "negative_control_metrics.csv", index=False)

# ----------------------------------------------------------------------------- real
real = json.loads((C / "real.json").read_text())
checker = json.loads((C / "real_checker.json").read_text())
rp = L.load_real()
real_rows = []
for m in real["methods"]:
    ck = checker[m["method_key"]]
    ck_mean = float(np.mean([s["heldout_inlier3"] for s in ck]))
    # detector re-evaluated with the checkerboard held-out (the half-field split confounds crop with generalisation)
    det = L.detector(np.asarray(m["M"]), np.asarray(m["t"]), rp["moving_px"], rp["fixed_all"],
                     basin_margin=m["diagnostics"].get("basin_margin") if m["diagnostics"].get("basin_margin") is not None else np.inf,
                     heldout_inlier=ck_mean)
    m.update(det)
    r = flat("real", m, {})
    r.update({"M": json.dumps(np.round(m["M"], 6).tolist()), "t": json.dumps(np.round(m["t"], 4).tolist()),
              "vs_saved_rms_um": m["vs_saved"]["transform_rms_error_um"],
              "vs_reference_rms_um": m["vs_reference_refit"]["transform_rms_error_um"],
              "vs_reference_rotation_deg": m["vs_reference_refit"]["rotation_error_deg"],
              "vs_reference_scale_pct": m["vs_reference_refit"]["scale_error_pct"],
              "heldout_checker_inlier3_min": min(s["heldout_inlier3"] for s in ck),
              "heldout_checker_null_mean": float(np.mean([s["heldout_null"] for s in ck])),
              "heldout_checker_vs_full_rms_um": float(np.max([L.transform_errors(np.asarray(s["M"]), np.asarray(s["t"]), np.asarray(m["M"]), np.asarray(m["t"]), 901, 901)["transform_rms_error_um"] for s in ck])),
              "crop_stress_halffield_inlier3_mean": float(np.mean([s["heldout_inlier3"] for s in m["heldout_splits"]])),
              "crop_stress_halffield_inlier3_max": max(s["heldout_inlier3"] for s in m["heldout_splits"]),
              "he_white_fixed_fraction": m["he_white_fixed_fraction"], "he_white_area_fraction": m["he_white_area_fraction"]})
    real_rows.append(r)
realdf = pd.DataFrame(real_rows)

# ----------------------------------------------------------------------------- downstream
ds = pd.DataFrame([r for f in sorted(glob.glob(str(C / "downstream_*.json"))) for r in json.loads(open(f).read())])
if len(ds):
    ds.to_csv(L.OUT / "downstream_graph_metrics.csv", index=False)
    rd = ds[(ds.case == "real") & (ds.pipeline_method == "Position + Graph")].set_index("method_key")
    for col in ("ncc18_nucleus_boundary_rate", "graph_support_median", "candidate_distance_median_um", "top1_equals_reference_partner"):
        realdf[col] = realdf.method_key.map(rd[col])
    b2 = ds[(ds.case == "real") & (ds.pipeline_method == "Separated Matching / Adoption")].set_index("method_key")
    realdf["b2_matched"] = realdf.method_key.map(b2.matched)
realdf.to_csv(L.OUT / "real_affine_comparison.csv", index=False)

# ----------------------------------------------------------------------------- runtime
rt = pd.concat([syn[["case", "method_key", "runtime_s", "peak_mem_mb", "n_hypotheses_scored", "n_refined"]].assign(dataset="synthetic"),
                ctl[["case", "method_key", "runtime_s", "peak_mem_mb", "n_hypotheses_scored", "n_refined"]].assign(dataset="control"),
                realdf[["case", "method_key", "runtime_s", "peak_mem_mb", "n_hypotheses_scored", "n_refined"]].assign(dataset="real")])
rt["method"] = rt.method_key.map(NAMES)
rt.to_csv(L.OUT / "runtime_metrics.csv", index=False)

# ----------------------------------------------------------------------------- method comparison
mc = []
for k in NAMES:
    s = syn[syn.method_key == k]
    c = ctl[ctl.method_key == k]
    rr = realdf[realdf.method_key == k].iloc[0]
    mc.append({
        "method_key": k, "method": NAMES[k],
        "synthetic_cases": len(s), "synthetic_success_rate": float((s.outcome == "success").mean()),
        "synthetic_imprecise_rate": float((s.outcome == "imprecise").mean()),
        "synthetic_wrong_basin_rate": float((s.outcome == "wrong_basin").mean()),
        "synthetic_transform_rms_median_um": float(s.transform_rms_error_um.median()),
        "synthetic_success_conditions": int(s.groupby("condition").apply(lambda t: (t.outcome == "success").all()).sum()),
        "synthetic_conditions": int(s.condition.nunique()),
        "detector_pass_given_success": float(s[s.outcome == "success"].detector_pass.mean()) if (s.outcome == "success").any() else np.nan,
        "detector_pass_given_failure": float(s[s.outcome != "success"].detector_pass.mean()) if (s.outcome != "success").any() else np.nan,
        "real_objective": rr.objective, "real_inlier3": rr.inlier3, "real_heldout_inlier3_mean": rr.heldout_inlier3,
        "real_heldout_inlier3_min": rr.heldout_checker_inlier3_min, "real_vs_reference_rms_um": rr.vs_reference_rms_um,
        "real_detector": rr.detector_status,
        "runtime_median_s": float(rt[rt.method_key == k].runtime_s.median()), "runtime_max_s": float(rt[rt.method_key == k].runtime_s.max()),
        "peak_mem_median_mb": float(rt[rt.method_key == k].peak_mem_mb.median()),
        "negative_control_cases": len(c), "negative_control_false_positive_pass": int(c.false_positive_pass.sum()),
        "negative_control_ambiguous": int((c.detector_status == "AMBIGUOUS").sum()),
    })
mc = pd.DataFrame(mc)
mc.to_csv(L.OUT / "method_comparison.csv", index=False)

# per-condition table for the report
cond = syn.pivot_table(index="condition", columns="method_key", values="transform_rms_error_um", aggfunc="median")
succ = syn.assign(ok=syn.outcome == "success").pivot_table(index="condition", columns="method_key", values="ok", aggfunc="mean")
cond.to_csv(C / "condition_rms_median.csv"); succ.to_csv(C / "condition_success_rate.csv")

git = lambda *a: subprocess.run(["git", *a], cwd=L.ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
L.save_json(L.OUT / "diagnostic_summary.json", {
    "purpose": "Affine global initialisation research (analysis-only)", "date": "2026-10-14",
    "branch": git("branch", "--show-current"), "git_head": git("rev-parse", "HEAD"), "python": sys.version, "platform": platform.platform(),
    "shared_settings": {k: getattr(L, k) for k in ("SEARCH_RADIUS_UM", "COARSE_ROT_STEP", "FINE_ROT_HALF", "FINE_ROT_STEP", "FINE_SCALE",
                                                   "N_REFINE", "A3_ROT_STEP", "A3_TRANS_STEP", "A3_SHORT_ITERS", "A3_TOPK")},
    "detector_thresholds_prespecified": L.DETECT,
    "synthetic_conditions": L.SYNTHETIC_CONDITIONS, "synthetic_seeds": L.SYNTHETIC_SEEDS,
    "method_comparison": mc.to_dict("records"),
    "condition_success_rate": succ.reset_index().to_dict("records"),
    "real_saved_detector": real["saved_detector"], "real_reference_detector": real["reference_detector"],
    "real_methods": realdf.drop(columns=["M", "t"]).to_dict("records"),
    "controls": ctl[["case", "method_key", "detector_status", "inlier3", "inlier_over_null", "block_consistency", "basin_margin", "objective"]].to_dict("records"),
    "changes_outside_this_directory": "none (verified by sha256 before/after)",
})

# ============================================================================= figures
keys = list(NAMES)
# synthetic transform error
fig, axs = plt.subplots(1, 2, figsize=(18, 6.5), gridspec_kw={"width_ratios": [3, 1]})
conds = list(L.SYNTHETIC_CONDITIONS)
for i, k in enumerate(keys):
    s = syn[syn.method_key == k]
    x = np.array([conds.index(c) for c in s.condition]) + (i - 1.5) * 0.18
    axs[0].scatter(x, s.transform_rms_error_um.clip(lower=0.1), c=COL[k], s=22, label=NAMES[k])
axs[0].set_yscale("log"); axs[0].axhline(5, ls="--", c="k", lw=0.8); axs[0].axhline(20, ls=":", c="k", lw=0.8)
axs[0].set_xticks(range(len(conds))); axs[0].set_xticklabels(conds, rotation=60, ha="right", fontsize=8)
axs[0].set_ylabel("RMS transform error over moving FOV (µm, log)"); axs[0].legend(fontsize=8)
axs[0].set_title("Synthetic (3 seeds/condition): ≤5 µm success, >20 µm or wrong chirality = wrong basin")
oc = syn.groupby(["method_key", "outcome"]).size().unstack(fill_value=0).reindex(columns=["success", "imprecise", "wrong_basin"], fill_value=0)
oc.index = [NAMES[k] for k in oc.index]
oc.plot.barh(stacked=True, ax=axs[1], color=["#3b8b5a", "#e3a72f", "#d1495b"]); axs[1].set_xlabel("cases")
fig.savefig(L.OUT / "synthetic_transform_error.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# initialization comparison (translation offset sweep + Real)
fig, axs = plt.subplots(1, 3, figsize=(20, 5.5))
sw = syn[syn.condition.str.match(r"offset\d")]
for k in keys:
    s = sw[sw.method_key == k].groupby("param_offset_um").apply(lambda t: (t.outcome == "success").mean())
    axs[0].plot(s.index, s.values, "o-", c=COL[k], label=NAMES[k])
axs[0].set_xlabel("FOV offset between sections (µm)"); axs[0].set_ylabel("success rate"); axs[0].legend(fontsize=8)
axs[0].set_title("Robustness to field-of-view offset (synthetic)")
x = np.arange(4)
axs[1].bar(x - 0.2, realdf.inlier3, 0.4, color=[COL[k] for k in realdf.method_key], label="all overlap")
axs[1].bar(x + 0.2, realdf.heldout_inlier3, 0.4, color=[COL[k] for k in realdf.method_key], alpha=0.5, hatch="//", label="checkerboard held-out mean")
axs[1].axhline(real["saved_detector"]["inlier3_null"], c="k", ls=":", label="mirror null")
axs[1].set_xticks(x); axs[1].set_xticklabels([NAMES[k] for k in realdf.method_key], rotation=20, fontsize=8)
axs[1].set_ylabel("Real: fraction of moving nuclei with fixed nucleus ≤3 µm"); axs[1].legend(fontsize=8)
axs[2].barh([NAMES[k] for k in mc.method_key], mc.runtime_median_s, color=[COL[k] for k in mc.method_key])
axs[2].set_xlabel("median runtime per pair (s, 4 CPU)")
fig.savefig(L.OUT / "initialization_comparison.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# ICP convergence basins (reuse prior audit capture map + A3 start objective histogram on Real)
cap = pd.read_csv(L.PRIOR_AUDIT / "cache/icp_capture_range.csv")
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
pv = cap.pivot(index="init_dy", columns="init_dx", values="final_trimmed_error")
im = axs[0].imshow(pv.values, origin="lower", extent=[-85, 85, -85, 85], cmap="viridis_r")
plt.colorbar(im, ax=axs[0], label="final affine-ICP trimmed objective (µm)")
axs[0].set_title("Real: production affine_icp from saved affine + translation offset\n(prior audit; basin of the ~1.04 µm solution is ≈±30 µm wide)")
axs[0].set_xlabel("start offset dx (µm)"); axs[0].set_ylabel("dy")
s = syn.copy()
for k in keys:
    t = s[s.method_key == k]
    axs[1].scatter(t.truth_basin_objective, t.objective, c=COL[k], s=18, label=NAMES[k])
lim = [0, max(10, float(s.objective.max()) * 1.05)]
axs[1].plot(lim, lim, "k--", lw=0.8); axs[1].set_xlabel("objective of ICP started at the true affine"); axs[1].set_ylabel("objective reached")
axs[1].set_title("Synthetic: wrong basins have higher ICP objective than the truth basin"); axs[1].legend(fontsize=8)
fig.savefig(L.OUT / "icp_convergence_basins.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# coarse search peaks on Real (recompute surfaces for the selected orientation; cheap)
r = L.load_real()
best = {m["method_key"]: m for m in real["methods"]}
from scipy.spatial import cKDTree
fig, axs = plt.subplots(1, 3, figsize=(20, 6))
h = best["A1"]["basins"][0][3]
q = L._transform_hyp(r["moving_px"], h["chirality"], h["theta"], h["scale"])
tc = r["fixed_all"].mean(0) - q.mean(0)
tree = cKDTree(r["fixed_all"])
lists = tree.query_ball_point(q + tc, 210)
disp = np.concatenate([r["fixed_all"][p] - (q + tc)[i] for i, p in enumerate(lists) if p])
H, _, _ = np.histogram2d(disp[:, 0], disp[:, 1], bins=400, range=[[-200, 200], [-200, 200]])
from scipy.ndimage import gaussian_filter
Hk, B = gaussian_filter(H, 2, mode="constant"), gaussian_filter(H, 15, mode="constant")
z = (Hk - B) / np.sqrt(np.maximum(B, 1e-3) / (4 * np.pi * 4))
im = axs[0].imshow(z.T, origin="lower", extent=[-200, 200, -200, 200], cmap="magma", vmin=-3)
plt.colorbar(im, ax=axs[0], label="z vs local background")
axs[0].set_title(f"A1 displacement-correlogram z-map, Real (θ={h['theta']:.0f}°, {h['chirality']})\norigin = centroid alignment (A0 start); peak = selected translation")
axs[0].set_xlabel("dx (µm)"); axs[0].set_ylabel("dy (µm)")
co = pd.DataFrame(best["A1"]["diagnostics"]["coarse_top"], columns=["chir", "theta", "scale", "t", "z"])
axs[1].bar(range(len(co)), co.z, color=[COL["A1"] if c == "reflect" else "#999" for c in co.chir])
axs[1].set_xticks(range(len(co))); axs[1].set_xticklabels([f"{c[:4]} {t:.0f}°" for c, t in zip(co.chir, co.theta)], rotation=60, fontsize=8)
axs[1].set_ylabel("coarse peak z"); axs[1].set_title("A1 coarse hypotheses (top 10): peak margin")
co2 = pd.DataFrame(best["A2"]["diagnostics"]["coarse_top"], columns=["chir", "theta", "scale", "t", "ncc"])
axs[2].bar(range(len(co2)), co2.ncc, color=[COL["A2"] if c == "reflect" else "#999" for c in co2.chir])
axs[2].set_xticks(range(len(co2))); axs[2].set_xticklabels([f"{c[:4]} {t:.0f}°" for c, t in zip(co2.chir, co2.theta)], rotation=60, fontsize=8)
axs[2].set_ylabel("masked NCC"); axs[2].set_title("A2 coarse hypotheses (top 10)")
fig.savefig(L.OUT / "coarse_search_peaks.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# Real before/after overlay on HE (warped with the saved affine; fixed mapped into that frame)
meta = r["warp_meta"]; ext = [meta["bounds_um"][0], meta["bounds_um"][2], meta["bounds_um"][1], meta["bounds_um"][3]]
Ms, ts = L.saved_effective(r)
fig, axs = plt.subplots(1, 4, figsize=(24, 6.5))
for ax, k in zip(axs, keys):
    M, t = np.asarray(best[k]["M"]), np.asarray(best[k]["t"])
    fs = L.apply_eff(Ms, ts, np.linalg.solve(M, (r["fixed_all"] - t).T).T)
    ax.imshow(r["he_image"], extent=ext, origin="upper")
    ax.scatter(fs[:, 0], fs[:, 1], s=36, facecolors="none", edgecolors="cyan", lw=1.3)
    ax.scatter(r["moving_saved_all"][:, 0], r["moving_saved_all"][:, 1], s=12, marker="x", c="yellow")
    ax.set_xlim(0, 230); ax.set_ylim(80, 310)
    ax.set_title(f"{NAMES[k]}\n≤3 µm {100*best[k]['inlier3']:.1f}%, HE-white {100*best[k]['he_white_fixed_fraction']:.2f}%")
fig.suptitle("Fluorescence nuclei (cyan) placed on the HE image by each method's affine; yellow = HE nuclei")
fig.savefig(L.OUT / "real_before_after_overlay.png", dpi=120, bbox_inches="tight"); plt.close(fig)

# graph support by initialization
if len(ds):
    fig, axs = plt.subplots(1, 3, figsize=(20, 5.5))
    t = ds[ds.pipeline_method == "Position + Graph"]
    piv = t.pivot_table(index="case", columns="method_key", values="graph_support_median")
    piv.plot.bar(ax=axs[0], color=[COL.get(c, "#777") for c in piv.columns]); axs[0].set_ylabel("top-1 Graph support median")
    axs[0].set_title("Graph support (frozen Graph/UOT) by initialization")
    acc = t.pivot_table(index="case", columns="method_key", values="top1_accuracy_common")
    acc.plot.bar(ax=axs[1], color=[COL.get(c, "#777") for c in acc.columns]); axs[1].set_ylabel("synthetic top-1 accuracy (common nuclei)")
    b2t = ds[ds.pipeline_method == "Separated Matching / Adoption"].pivot_table(index="case", columns="method_key", values="matched")
    b2t.plot.bar(ax=axs[2], color=[COL.get(c, "#777") for c in b2t.columns]); axs[2].set_ylabel("B2 matched count (not accuracy)")
    for a in axs:
        a.tick_params(axis="x", labelrotation=30)
    fig.savefig(L.OUT / "graph_support_by_initialization.png", dpi=130, bbox_inches="tight"); plt.close(fig)
print("aggregated")
