"""Aggregate cache/ -> CSV / JSON / PNG deliverables (analysis only)."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import run_study as R  # noqa: E402

HERE = Path(__file__).resolve().parent
C = HERE / "cache"
L, MX, NR = R.L, R.MX, R.NR
B0, B2 = "Position + Graph", "Separated Matching / Adoption"
STAGES = ("R0", "R1a", "R1b", "R2s", "R2")
COL = {"R0": "#999999", "R1a": "#e08214", "R1b": "#b35806", "R2s": "#5e9ed6", "R2": "#2166ac", "R3": "#1a9850"}


def J(name):
    return json.loads((C / name).read_text())


def gini(x):
    x = np.sort(np.clip(np.asarray(x, float), 0, None))
    if x.sum() == 0:
        return np.nan
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


# ============================================================================ Real
r = L.load_real()
fixed, mov, W, H = r["fixed_all"], r["moving_px"], r["width"], r["height"]
aff = R.real_stage_affines()
mvC, fvC = R.common_sets(r, aff)
met = J("real_metrics.json")
stage_df = pd.DataFrame(met["stages"]).set_index("stage")
down = {k: J(f"real_down_{k}.json")["rows"] for k in R.DOWN_STAGES}
pern = {k: pd.read_csv(C / f"real_down_pernuc_{k}.csv.gz") for k in R.DOWN_STAGES}
held = pd.DataFrame([x for k in range(6) for x in J(f"real_heldout_{k}.json")["rows"]])
resid = J("real_residual.json")
semi = pd.concat([pd.DataFrame(J(f"semisyn_{k}.json")) for k in range(len(R.SEMI_AMPS))], ignore_index=True)


def drow(k, m):
    return next(x for x in down[k] if x["pipeline_method"] == m)


rows = []
for k in STAGES + ("R0_saved_subset",):
    base = k if k in stage_df.index else None
    b0, b2 = drow(k, B0), drow(k, B2)
    b1 = drow(k, "Relative-cost Graph-UOT")
    row = {"stage": k}
    if base:
        s = stage_df.loc[base]
        row.update({c: s[c] for c in stage_df.columns if not c.startswith("param_")})
        hk = "R1a" if k == "R0" else k
        if k != "R0":
            hc = held[(held.stage == k) & held.fold.str.startswith("checker")]
            row["heldout_checker_within3_mean"] = hc.heldout_within_3um.mean()
            row["heldout_checker_within1_mean"] = hc.heldout_within_1um.mean()
            row["heldout_checker_within2_mean"] = hc.heldout_within_2um.mean()
            row["heldout_checker_nn_median_mean"] = hc.heldout_nn_median_um.mean()
            hh = held[(held.stage == k) & ~held.fold.str.startswith("checker")]
            row["heldout_halfimage_within3_mean"] = hh.heldout_within_3um.mean()
        del hk
    row.update({"graph_support_median": b0["graph_support_median"], "candidate_distance_median_um": b0["candidate_distance_median_um"],
                "candidate_distance_p90_um": b0["candidate_distance_p90_um"], "uot_row_mass_median_B1": b1["row_mass_ratio_median"],
                "b2_matched": b2["matched_n"], "b2_ambiguous": b2["ambiguous_n"], "b2_unmatched": b2["predicted_unmatched_n"],
                "b2_confidence_median": b2["confidence_median"], "b2_duplicate_destination_count": b2["duplicate_destination_count"],
                "b2_reciprocal_fraction_matched": b2["reciprocal_fraction_matched"],
                "downstream_n_moving": b2["n_moving_subset"], "downstream_n_fixed": b2["n_fixed_subset"]})
    rows.append(row)
real_tab = pd.DataFrame(rows)
# top-1 agreement with R2 (same moving nucleus, same fixed nucleus) -- agreement, not accuracy
r2top = pern["R2"][pern["R2"].pipeline_method == B0].set_index("moving_orig_index").fixed_orig_index
for k in STAGES:
    t = pern[k][pern[k].pipeline_method == B0].set_index("moving_orig_index").fixed_orig_index
    j = t.to_frame("a").join(r2top.rename("b"), how="inner")
    real_tab.loc[real_tab.stage == k, "top1_agreement_with_R2"] = float((j.a == j.b).mean())
real_tab.to_csv(HERE / "real_stage_comparison.csv", index=False)

# ---------------------------------------------------------------------------- affine parameters
prow = []
for k in STAGES:
    M, t = aff[k]
    prow.append({"kind": "parameters", "stage": k, **MX.decompose(M, t, W, H)})
for d in met["displacements"]:
    prow.append({"kind": "displacement (not an error)", "stage": f"{d['from']}->{d['to']}", **d})
for k in range(6):
    hj = J(f"real_heldout_{k}.json")
    for s, v in hj["stages"].items():
        prow.append({"kind": f"heldout_fit:{hj['fold']}", "stage": s, **MX.decompose(np.asarray(v["M"]), np.asarray(v["t"]), W, H)})
pd.DataFrame(prow).to_csv(HERE / "affine_parameter_comparison.csv", index=False)

# ---------------------------------------------------------------------------- spatial blocks (100 um, R2 frame)
BLK = 100.0
w = {k: MX.apply(*aff[k], mov) for k in STAGES}
ref = w["R2"]                                           # block assignment uses the R2 position of each moving nucleus
bkey = np.floor(ref / BLK).astype(int)
nn = {k: cKDTree(fixed[fvC]).query(w[k])[0] for k in STAGES}
z = np.load(C / "real_residual_vectors_R2.npz")
res_vec = np.full((len(mov), 2), np.nan); res_vec[z["moving_index"]] = z["vec"]
zz = {}
for k in ("R1a", "R2"):
    i, j, u = NR.proxy_pairs(w[k][mvC], fixed[fvC], r=4.0)
    v = np.full((len(mov), 2), np.nan); v[np.flatnonzero(mvC)[i]] = u; zz[k] = v
disp = np.linalg.norm(w["R2"] - w["R1a"], axis=1)
mat = {}
for k in ("R0", "R1a", "R2s", "R2"):
    f = pern[k][(pern[k].pipeline_method == B2)]
    mat[k] = np.zeros(len(mov), bool); mat[k][f.moving_orig_index[f.status == "matched"].to_numpy(int)] = True
gsup = {}
for k in ("R1a", "R2"):
    f = pern[k][pern[k].pipeline_method == B0]
    g = np.full(len(mov), np.nan); g[f.moving_orig_index.to_numpy(int)] = f.top1_graph_support.to_numpy(); gsup[k] = g
cb = pd.DataFrame([b for b in met["blocks"] if b["block_um"] == BLK])
brows = []
for key in sorted(set(map(tuple, bkey[mvC]))):
    sel = mvC & (bkey[:, 0] == key[0]) & (bkey[:, 1] == key[1])
    if sel.sum() < 8:
        continue
    cx, cy = (np.array(key) + 0.5) * BLK
    row = {"block_x0": key[0] * BLK, "block_y0": key[1] * BLK, "center_x": cx, "center_y": cy, "n_moving_common": int(sel.sum())}
    for k in STAGES:
        row[f"within3_{k}"] = float(np.mean(nn[k][sel] <= 3)); row[f"nn_median_{k}"] = float(np.median(nn[k][sel]))
    for k in ("R1a", "R2"):
        v = zz[k][sel]; v = v[np.isfinite(v[:, 0])]
        row[f"proxy_n_{k}"] = len(v)
        row[f"proxy_mean_dx_{k}"], row[f"proxy_mean_dy_{k}"] = (v.mean(0) if len(v) else (np.nan, np.nan))
        row[f"proxy_mean_mag_{k}"] = float(np.linalg.norm(v.mean(0))) if len(v) else np.nan
        row[f"proxy_mag_median_{k}"] = float(np.median(np.linalg.norm(v, axis=1))) if len(v) else np.nan
        row[f"graph_support_median_{k}"] = float(np.nanmedian(gsup[k][sel]))
    row["proxy_noise_floor_R2"] = float(np.sqrt(np.nanmean(np.sum(res_vec[mvC] ** 2, 1))) * np.sqrt(np.pi / 4) / np.sqrt(max(row["proxy_n_R2"], 1)))
    for k in ("R0", "R1a", "R2s", "R2"):
        row[f"b2_matched_{k}"] = int(mat[k][sel].sum())
    row["icp_displacement_median_um"] = float(np.median(disp[sel]))
    c2 = cb[(cb.stage == "R2") & np.isclose(cb.center_x, cx) & np.isclose(cb.center_y, cy)]
    row["correlogram_residual_R2_um"] = float(c2.residual_mag_um.iloc[0]) if len(c2) else np.nan
    c1 = cb[(cb.stage == "R1a") & np.isclose(cb.center_x, cx) & np.isclose(cb.center_y, cy)]
    row["correlogram_residual_R1a_um"] = float(c1.residual_mag_um.iloc[0]) if len(c1) else np.nan
    he_white_d = None  # noqa: F841
    brows.append(row)
blocks = pd.DataFrame(brows)
blocks["delta_within3_R1a_to_R2"] = blocks.within3_R2 - blocks.within3_R1a
blocks["delta_b2_matched_R1a_to_R2"] = blocks.b2_matched_R2 - blocks.b2_matched_R1a
blocks["delta_within3_R2s_to_R2"] = blocks.within3_R2 - blocks.within3_R2s
blocks.to_csv(HERE / "spatial_region_residuals.csv", index=False)

inc = blocks.delta_b2_matched_R1a_to_R2.to_numpy()
pos_inc = np.clip(inc, 0, None)
srt = np.sort(pos_inc)[::-1]
conc = {"blocks": int(len(blocks)), "total_increment_R1a_to_R2": int(inc.sum()),
        "blocks_increase": int((inc > 0).sum()), "blocks_decrease": int((inc < 0).sum()), "blocks_equal": int((inc == 0).sum()),
        "gini_positive_increment": gini(pos_inc),
        "top10pct_blocks_share_of_positive_increment": float(srt[:max(1, int(np.ceil(0.1 * len(srt))))].sum() / max(srt.sum(), 1)),
        "spearman_increment_vs_icp_displacement": float(spearmanr(inc, blocks.icp_displacement_median_um).correlation),
        "spearman_increment_vs_R1a_within3": float(spearmanr(inc, blocks.within3_R1a).correlation),
        "spearman_delta_within3_vs_icp_displacement": float(spearmanr(blocks.delta_within3_R1a_to_R2, blocks.icp_displacement_median_um).correlation),
        "blocks_within3_improved_gt0.05": int((blocks.delta_within3_R1a_to_R2 > 0.05).sum()),
        "blocks_within3_worsened_gt0.05": int((blocks.delta_within3_R1a_to_R2 < -0.05).sum()),
        "blocks_within3_R2s_to_R2_improved_gt0.05": int((blocks.delta_within3_R2s_to_R2 > 0.05).sum()),
        "blocks_within3_R2s_to_R2_worsened_gt0.05": int((blocks.delta_within3_R2s_to_R2 < -0.05).sum())}

# ---------------------------------------------------------------------------- ICP incremental effect (paired)
S = real_tab.set_index("stage")
ICP_FAV = ("common_m2f_within", "common_icp", "common_sym_nn", "common_mutual", "common_f2m", "common_m2f_nn", "all_", "roi_")
spec = [  # (metric, scope, higher_is_better, note)
    ("common_m2f_within_1um", "in-sample COMMON", True, ""), ("common_m2f_within_2um", "in-sample COMMON", True, ""),
    ("common_m2f_within_3um", "in-sample COMMON", True, ""), ("common_m2f_within_5um", "in-sample COMMON", True, ""),
    ("common_m2f_within_10um", "in-sample COMMON", True, ""), ("common_sym_nn_median_um", "in-sample COMMON", False, ""),
    ("common_sym_nn_p90_um", "in-sample COMMON", False, ""), ("common_sym_nn_p95_um", "in-sample COMMON", False, ""),
    ("common_mutual_nn_rate", "in-sample COMMON", True, ""), ("common_mutual_nn_within3_rate", "in-sample COMMON", True, ""),
    ("common_icp_trimmed_objective_q07", "in-sample COMMON", False, "ICP's own objective (favours ICP by construction)"),
    ("heldout_checker_within1_mean", "held-out checkerboard (2 folds)", True, "fit and evaluation nuclei disjoint"),
    ("heldout_checker_within2_mean", "held-out checkerboard (2 folds)", True, "fit and evaluation nuclei disjoint"),
    ("heldout_checker_within3_mean", "held-out checkerboard (2 folds)", True, "fit and evaluation nuclei disjoint"),
    ("heldout_checker_nn_median_mean", "held-out checkerboard (2 folds)", False, "fit and evaluation nuclei disjoint"),
    ("common_density_corr_sigma5", "density map", True, ""), ("common_density_corr_sigma10", "density map", True, ""),
    ("common_density_corr_sigma20", "density map", True, ""),
    ("regional_ncc_best_median", "regional NCC (frozen +-18)", True, ""), ("regional_ncc_nucleus_boundary_rate", "regional NCC (frozen +-18)", False, ""),
    ("corr200_local_residual_median_um", "local correlogram 200 um", False, "not NN based"),
    ("corr100_local_residual_median_um", "local correlogram 100 um", False, "not NN based"),
    ("corr200_peak_ratio_median", "local correlogram 200 um", True, "not NN based"),
    ("he_white_fixed_fraction", "HE image (independent of moving points)", False, "white != guaranteed no-tissue"),
    ("he_density_white_edge_rel", "HE image (independent of moving points)", False, ""),
    ("he_hema_mean_at_fixed_z", "HE image (independent of moving points)", True, ""),
    ("hescan_hema_offset_mag_um", "HE image (independent of moving points)", False, "best-HE offset magnitude"),
    ("graph_support_median", "downstream", True, ""), ("candidate_distance_median_um", "downstream", False, ""),
    ("uot_row_mass_median_B1", "downstream", None, "mass level, not a quality score"),
    ("b2_matched", "downstream", None, "count, not correctness"), ("b2_ambiguous", "downstream", None, ""),
    ("b2_unmatched", "downstream", None, ""), ("b2_confidence_median", "downstream", None, "calibrated on synthetic"),
]
irows = []
for m, scope, hib, note in spec:
    a, s_, b = S.loc["R1a", m], S.loc["R2s", m], S.loc["R2", m]
    irows.append({"metric": m, "scope": scope, "R0": S.loc["R0", m] if m in S.columns else np.nan, "R1a": a, "R1b": S.loc["R1b", m],
                  "R2s": s_, "R2": b, "delta_R1a_to_R2s": s_ - a, "delta_R2s_to_R2": b - s_, "delta_R1a_to_R2": b - a,
                  "direction_R1a_to_R2": ("n/a" if hib is None else ("improved" if (b - a) * (1 if hib else -1) > 0 else "worsened" if b != a else "equal")),
                  "icp_favourable_metric": bool(scope == "in-sample COMMON"), "note": note})
for k, v in conc.items():
    irows.append({"metric": k, "scope": "spatial blocks 100 um (R1a->R2)", "R1a": np.nan, "R2": v, "note": "block-level summary"})
for m in ("param_scale_sqrt_absdet", "param_rotation_deg_after_chirality", "param_tx", "param_ty", "param_anisotropy_ratio", "param_shear_offdiag"):
    a, s_, b = stage_df.loc["R1a", m], stage_df.loc["R2s", m], stage_df.loc["R2", m]
    irows.append({"metric": m, "scope": "affine parameter", "R0": stage_df.loc["R0", m], "R1a": a, "R1b": stage_df.loc["R1b", m], "R2s": s_, "R2": b,
                  "delta_R1a_to_R2s": s_ - a, "delta_R2s_to_R2": b - s_, "delta_R1a_to_R2": b - a, "note": "parameter change"})
pd.DataFrame(irows).to_csv(HERE / "icp_incremental_effect.csv", index=False)

# ---------------------------------------------------------------------------- downstream comparison
drows = []
for k in R.DOWN_STAGES:
    for x in down[k]:
        drows.append({"stage": k, **{c: x.get(c) for c in ("pipeline_method", "n_moving_subset", "n_fixed_subset", "matched_n", "ambiguous_n",
                                                             "predicted_unmatched_n", "coverage", "graph_support_median", "candidate_distance_median_um",
                                                             "candidate_distance_p90_um", "row_mass_ratio_median", "confidence_median",
                                                             "duplicate_destination_count", "duplicate_destinations_matched",
                                                             "top1_duplicate_destination_count_all", "reciprocal_fraction_matched", "solver_converged")}})
ddf = pd.DataFrame(drows)
for k in STAGES:
    ddf.loc[ddf.stage == k, "top1_agreement_with_R2_B0"] = real_tab.set_index("stage").loc[k, "top1_agreement_with_R2"]
for kk, v in conc.items():
    ddf.loc[len(ddf)] = {"stage": "R1a->R2 spatial", "pipeline_method": kk, "matched_n": v}
ddf.to_csv(HERE / "downstream_graph_comparison.csv", index=False)

# ============================================================================ Synthetic
tests = [J(f"syn_test_{i:03d}.json") for i in range(len(R.test_cases()))]
devs = [J(f"syn_dev_{i:03d}.json") for i in range(112)]
sdown = {}
for jj, (i, key) in enumerate(R.syn_down_jobs()):
    d = J(f"syn_down_{jj:03d}.json")
    if d.get("same_as"):
        d = J(f"syn_down_{R.syn_down_jobs().index((i, d['same_as'])):03d}.json")
    sdown[(i, key)] = d
GROUP = {"IV_T1_affine_only": "global affine only", "IV_T2_nonrigid_mild": "mild non-rigid (3 um)", "IV_T3_nonrigid_moderate": "moderate non-rigid (6 um)",
         "IV_T4_nonrigid_strong": "strong non-rigid (12 um)", "IV_T5_fov_contained": "large FOV difference", "IV_T6_fov_smallfixed": "large FOV difference",
         "IV_T7_shared_low": "low shared fraction"}
srows = []
for t in tests:
    for k, v in t["stages"].items():
        row = {"dataset": "Test", "family": t["family"], "condition": t["condition"], "group": GROUP[t["condition"]], "rep": t["rep"], "stage": k,
               "r1a_equals_r1b": t["r1a_equals_r1b"], "selected_ncc_rank": t["selected_ncc_rank"],
               **{c: v[c] for c in ("rms_common_um", "rms_fullfield_um", "chirality_correct", "scale_error_pct", "rotation_error_deg",
                                    "translation_error_center_um", "true_pair_residual_median_um", "true_pairs_within3", "affine_success",
                                    "m2f_within3_all", "icp_trimmed_objective_q07")},
               "overlap_min_coverage": t["params"]["overlap_min_coverage"]}
        dk = {"R1a": "R1a", "R1b": "R1b", "R2": "R2", "R3": "TRUE"}.get(k)
        if dk:
            d = sdown[(t["index"], dk)]
            if "rows" in d:
                b0 = next(x for x in d["rows"] if x["pipeline_method"] == B0); b2 = next(x for x in d["rows"] if x["pipeline_method"] == B2)
                row.update(candidate_recall30=b0["true_candidate_recall"], graph_top1_accuracy=b0["top1_accuracy"],
                           graph_support_median=b0["graph_support_median"], b2_matched_precision=b2["matched_precision"],
                           b2_matched_recall=b2["matched_recall"], b2_matched_f1=b2["matched_f1"], b2_unmatched_f1=b2["unmatched_f1"],
                           b2_matched_n=b2["matched_n"])
            else:
                row["downstream_skipped"] = d.get("skipped")
        srows.append(row)
for t in devs:
    for k, v in t["stages"].items():
        srows.append({"dataset": "Dev (used before; descriptive only)", "family": t["family"], "condition": t["condition"], "group": t["condition"],
                      "rep": t["rep"], "stage": k, "r1a_equals_r1b": t["r1a_equals_r1b"], "selected_ncc_rank": t["selected_ncc_rank"],
                      **{c: v[c] for c in ("rms_common_um", "rms_fullfield_um", "chirality_correct", "scale_error_pct", "rotation_error_deg",
                                           "translation_error_center_um", "true_pair_residual_median_um", "true_pairs_within3", "affine_success",
                                           "m2f_within3_all", "icp_trimmed_objective_q07")},
                      "overlap_min_coverage": t["params"]["overlap_min_coverage"]})
syn = pd.DataFrame(srows)
syn.to_csv(HERE / "synthetic_truth_comparison.csv", index=False)
T = syn[syn.dataset == "Test"]
GORDER = ["global affine only", "mild non-rigid (3 um)", "moderate non-rigid (6 um)", "strong non-rigid (12 um)", "large FOV difference", "low shared fraction"]
syn_sum = (T.groupby(["group", "stage"]).agg(n=("rms_common_um", "size"), rms_median=("rms_common_um", "median"),
                                             success=("affine_success", "sum"), true_pair_res_median=("true_pair_residual_median_um", "median"),
                                             true_within3_median=("true_pairs_within3", "median"), top1_median=("graph_top1_accuracy", "median"),
                                             recall30_median=("candidate_recall30", "median"), b2_prec_median=("b2_matched_precision", "median"),
                                             b2_f1_median=("b2_matched_f1", "median"), unm_f1_median=("b2_unmatched_f1", "median"))
           .reset_index())
syn_sum.to_csv(C / "synthetic_group_summary.csv", index=False)
# paired ICP effect per case (R1a -> R2 and R2s -> R2) on Test
pc = T.pivot_table(index=["family", "condition", "rep", "group"], columns="stage", values="rms_common_um").reset_index()
pc["icp_helps"] = pc.R2 < pc.R1a - 0.5
pc["icp_hurts"] = pc.R2 > pc.R1a + 0.5
pc["affine_step_hurts"] = pc.R2 > pc.R2s + 0.5
icp_by_group = pc.groupby("group").agg(n=("R2", "size"), icp_helps=("icp_helps", "sum"), icp_hurts=("icp_hurts", "sum"),
                                        affine_step_hurts=("affine_step_hurts", "sum"), R1a_med=("R1a", "median"), R1b_med=("R1b", "median"),
                                        R2s_med=("R2s", "median"), R2_med=("R2", "median")).reset_index()
dv = syn[syn.dataset != "Test"].pivot_table(index=["family", "condition", "rep"], columns="stage", values="rms_common_um").reset_index()
dev_sum = {"n": int(len(dv)), "R1a_success_le5": int((dv.R1a <= 5).sum()), "R1b_success_le5": int((dv.R1b <= 5).sum()),
           "R2s_success_le5": int((dv.R2s <= 5).sum()), "R2_success_le5": int((dv.R2 <= 5).sum()),
           "R1a_ok_R2_fail": int(((dv.R1a <= 5) & (dv.R2 > 5)).sum()), "R1a_fail_R2_ok": int(((dv.R1a > 5) & (dv.R2 <= 5)).sum()),
           "R1a_equals_R1b": int(sum(t["r1a_equals_r1b"] for t in devs)),
           "R2_reproduces_prior_max_abs_diff": float(max(t["R2_max_abs_diff_vs_prior"] for t in devs))}
# residual diagnostic on synthetic deformation conditions (truth available)
rd = []
for t in tests:
    if "residual_diagnostic" in t:
        for k, v in t["residual_diagnostic"].items():
            rd.append({"family": t["family"], "condition": t["condition"], "group": GROUP[t["condition"]], "rep": t["rep"], "affine": k,
                       **{c: v.get(c) for c in ("n_vectors", "res_mag_median", "rho_0_20", "rho_20_50", "rho_50_100", "heldout_skill_b100_min",
                                                "heldout_skill_b200_min", "heldout_coherent_rms_b100", "truth_corr_proxy_vs_field",
                                                "truth_field_rms_at_pairs", "proxy_pair_true_id_rate")},
                       "perm_p95_b100_max": max(h.get("skill_perm_p95", np.nan) for h in v["heldout"] if h["block"] == 100)})
rdf = pd.DataFrame(rd)
rdf["detected_coherent"] = rdf.heldout_skill_b100_min > rdf.perm_p95_b100_max
rdf.to_csv(C / "synthetic_residual_diagnostic.csv", index=False)
rd_sum = rdf.groupby(["group", "affine"]).agg(n=("rho_0_20", "size"), rho0_20=("rho_0_20", "median"), skill100=("heldout_skill_b100_min", "median"),
                                             detected=("detected_coherent", "sum"), truth_corr=("truth_corr_proxy_vs_field", "median"),
                                             proxy_true_id=("proxy_pair_true_id_rate", "median"), res_mag=("res_mag_median", "median")).reset_index()

# ============================================================================ decision for non-rigid (pre-registered rule)
P = resid["sources"]["P_R2"]
semi_null = semi[semi.amp == 0]
null95_100 = float(semi_null.heldout_skill_b100_min.quantile(0.95)); null95_200 = float(semi_null.heldout_skill_b200_min.quantile(0.95))
crit = {}
for b in (100.0, 200.0):
    hs = [h for h in P["heldout"] if h["block"] == b]
    crit[f"b{b:g}_both_colours_exceed_perm_p95"] = bool(all(h["skill"] > h["skill_perm_p95"] for h in hs))
    crit[f"b{b:g}_both_colours_exceed_N3_p95"] = bool(all(h["skill"] > (null95_100 if b == 100 else null95_200) for h in hs))
coherent_detected = any(crit[f"b{b:g}_both_colours_exceed_perm_p95"] and crit[f"b{b:g}_both_colours_exceed_N3_p95"] for b in (100.0, 200.0))
# detection floor: smallest amplitude whose 5th-percentile skill exceeds the null 95th percentile
floor = None
for a in sorted(semi.amp.unique()):
    if a > 0 and semi[semi.amp == a].heldout_skill_b100_min.quantile(0.05) > null95_100:
        floor = float(a); break
nonrigid_decision = {
    "rule": "coherent if held-out skill > perm-p95 and > N3(amp 0)-p95 in both checker colours at >=1 block scale, and coherent RMS >= 0.5 um",
    "criteria": crit, "coherent_detected": coherent_detected,
    "real_P_R2_skill_b100": [h["skill"] for h in P["heldout"] if h["block"] == 100],
    "real_P_R2_skill_b200": [h["skill"] for h in P["heldout"] if h["block"] == 200],
    "N3_null_skill_b100_p95": null95_100, "N3_null_skill_b200_p95": null95_200,
    "N3_detection_floor_amp_um_rms": floor,
    "real_rho_0_20": P["rho_0_20"], "N3_null_rho_0_20_p95": float(semi_null.rho_0_20.quantile(0.95)),
    "N3_amp0.5_rho_0_20_median": float(semi[semi.amp == 0.5].rho_0_20.median()),
    "real_coherent_rms_from_cov0_20_um": P["coherent_rms_from_cov0_20"],
    "N3_null_coherent_rms_from_cov0_20_p95": float(semi_null.coherent_rms_from_cov0_20.quantile(0.95)),
    "verdict": None,
}
short_range = P["rho_0_20"] > nonrigid_decision["N3_null_rho_0_20_p95"]
nonrigid_decision["short_range_correlation_above_null"] = bool(short_range)
nonrigid_decision["verdict"] = ("spatially coherent residual" if coherent_detected and P["coherent_rms_from_cov0_20"] >= 0.5 else
                                "detectable but small" if coherent_detected else
                                "random at the >=0.5 um / 100-200 um scale (below detection floor); weak short-range correlation above null"
                                if short_range else "random (below detection floor)")

# ============================================================================ diagnostic_summary.json
semi_tab = semi.groupby("amp").agg(rho0_20_mean=("rho_0_20", "mean"), rho0_20_p95=("rho_0_20", lambda x: x.quantile(0.95)),
                                   skill100_min_mean=("heldout_skill_b100_min", "mean"), skill100_min_p05=("heldout_skill_b100_min", lambda x: x.quantile(0.05)),
                                   skill100_min_p95=("heldout_skill_b100_min", lambda x: x.quantile(0.95)),
                                   coherent_rms_b100=("heldout_coherent_rms_b100", "mean"), truth_corr=("truth_corr_proxy_vs_field", "mean")).reset_index()
summary = {
    "real_stage_table": real_tab.set_index("stage")[["common_m2f_within_3um", "common_sym_nn_median_um", "corr200_local_residual_median_um",
                                                     "graph_support_median", "b2_matched", "b2_ambiguous", "b2_unmatched"]].to_dict("index"),
    "real_r1a_equals_r1b": J("real_stages.json")["r1a_equals_r1b"],
    "real_R2_reproduces_prior_A2_max_abs_diff": J("real_stages.json")["R2_max_abs_diff_vs_prior_A2"],
    "real_common_set": {"n_moving": int(mvC.sum()), "n_fixed": int(fvC.sum())},
    "real_heldout": held.groupby(["fold", "stage"]).heldout_within_3um.first().unstack().to_dict("index"),
    "real_displacements": met["displacements"],
    "real_he_quadrants": met["he_quadrants"],
    "real_spatial_concentration_R1a_to_R2": conc,
    "real_residual_sources": {k: {c: v.get(c) for c in ("n_vectors", "res_mag_median", "rho_0_20", "rho_20_50", "rho_50_100", "rho_100_200",
                                                         "dircons_r20", "dircons_r50", "dircons_r100", "coherent_rms_from_cov0_20",
                                                         "heldout_skill_b100_min", "heldout_skill_b200_min", "heldout_coherent_rms_b100",
                                                         "n_windows", "n_reliable", "mag_median")} for k, v in resid["sources"].items()},
    "real_residual_relations": resid["relations"],
    "semisynthetic_calibration": semi_tab.to_dict("records"),
    "nonrigid_decision": nonrigid_decision,
    "synthetic_test_icp_by_group": icp_by_group.to_dict("records"),
    "synthetic_test_group_summary": syn_sum.to_dict("records"),
    "synthetic_residual_diagnostic": rd_sum.to_dict("records"),
    "synthetic_dev_summary": dev_sum,
    "imaging_range_metadata": "input/metadata.json holds only the warped-HE output canvas (bounds, origin, flip); no acquisition/stage range for either modality",
}
L.save_json(HERE / "diagnostic_summary.json", summary)

# ============================================================================ figures
def fig_overlay():
    fig, ax = plt.subplots(2, 4, figsize=(18, 9))
    win = (fixed[:, 0] > 380) & (fixed[:, 0] < 530) & (fixed[:, 1] > 430) & (fixed[:, 1] < 580)
    for c, k in enumerate(("R0", "R1a", "R2s", "R2")):
        ww = w[k]
        a = ax[0, c]
        a.scatter(fixed[:, 0], fixed[:, 1], s=1.5, c="k", label="fixed")
        a.scatter(ww[:, 0], ww[:, 1], s=1.5, c=COL[k], alpha=.6, label=f"moving {k}")
        a.plot(*np.vstack([MX.he_footprint_world(*aff[k], W, H)] * 1)[[0, 1, 2, 3, 0]].T, c=COL[k], lw=1)
        a.set_title(f"{k}{' (=R1b)' if k == 'R1a' else ''}: <=3um {S.loc[k, 'common_m2f_within_3um']:.3f}"); a.set_aspect("equal"); a.legend(loc="lower left", fontsize=7, markerscale=4)
        b = ax[1, c]
        sel = (ww[:, 0] > 380) & (ww[:, 0] < 530) & (ww[:, 1] > 430) & (ww[:, 1] < 580)
        b.scatter(fixed[win, 0], fixed[win, 1], s=25, facecolors="none", edgecolors="k", label="fixed")
        b.scatter(ww[sel, 0], ww[sel, 1], s=8, c=COL[k], label="moving")
        b.set_xlim(380, 530); b.set_ylim(430, 580); b.set_aspect("equal"); b.set_title(f"{k}: 150x150 um zoom")
    fig.suptitle("Real stage overlay (R1a = R1b on Real). Geometric agreement only, not cell identity.")
    fig.tight_layout(); fig.savefig(HERE / "real_registration_stage_overlay.png", dpi=130); plt.close(fig)


def fig_nn():
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for k in STAGES:
        d = np.sort(cKDTree(fixed[fvC]).query(w[k][mvC])[0])
        ax[0].plot(d, np.arange(1, len(d) + 1) / len(d), c=COL[k], label=k, ls="--" if k == "R1b" else "-")
    ax[0].set_xscale("log"); ax[0].set_xlabel("moving->fixed NN distance (um), COMMON set"); ax[0].set_ylabel("CDF"); ax[0].legend()
    for th in (1, 2, 3, 5, 10):
        ax[0].axvline(th, c="0.85", lw=.8, zorder=0)
    ths = ["1", "2", "3", "5", "10"]; x = np.arange(len(ths))
    for i, k in enumerate(STAGES):
        ax[1].bar(x + (i - 2) * 0.16, [S.loc[k, f"common_m2f_within_{t}um"] for t in ths], 0.16, color=COL[k], label=k)
    ax[1].axhline(S.loc["R2", "common_null_mirror_x_within3"], c="r", ls=":", label="mirror null (<=3um)")
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"<={t}um" for t in ths]); ax[1].set_title("In-sample (ICP-favourable)"); ax[1].legend(fontsize=8)
    hc = held[held.fold.str.startswith("checker")]
    for i, k in enumerate(("R1a", "R1b", "R2s", "R2")):
        v = [hc[hc.stage == k][f"heldout_within_{t}um"].mean() for t in ths]
        e = [float(np.ptp(hc[hc.stage == k][f"heldout_within_{t}um"].to_numpy())) / 2 for t in ths]
        ax[2].bar(x + (i - 1.5) * 0.2, v, 0.2, yerr=e, color=COL[k], label=k)
    ax[2].set_xticks(x); ax[2].set_xticklabels([f"<={t}um" for t in ths]); ax[2].set_title("Held-out checkerboard (fit/eval nuclei disjoint; bar=mean, err=half range of 2 folds)"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(HERE / "nn_distance_by_stage.png", dpi=130); plt.close(fig)


def _quiver_blocks(a, pos, vec, scale, color, title):
    a.quiver(pos[:, 0], pos[:, 1], vec[:, 0], vec[:, 1], angles="xy", scale_units="xy", scale=1 / scale, color=color, width=0.004)
    a.set_aspect("equal"); a.set_title(title, fontsize=9); a.set_xlim(-20, 1040); a.set_ylim(-20, 1040)


def fig_icp_before_after():
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    for a, k in zip(ax[:2], ("R1a", "R2")):
        c, m, n = NR.window_means(w[k][mvC][np.isfinite(zz[k][mvC][:, 0])], zz[k][mvC][np.isfinite(zz[k][mvC][:, 0])], 100.0, min_n=5, origin=np.zeros(2))
        _quiver_blocks(a, c, m, 40, COL[k], f"{k}: 100-um mean proxy residual (x40)\nmedian |mean| {np.median(np.linalg.norm(m, axis=1)):.2f} um")
    sc = ax[2].scatter(blocks.center_x, blocks.center_y, c=blocks.delta_within3_R1a_to_R2, cmap="RdBu", vmin=-0.8, vmax=0.8, s=260, marker="s")
    plt.colorbar(sc, ax=ax[2], label="block <=3um rate: R2 - R1a")
    ax[2].set_aspect("equal"); ax[2].set_title(f"ICP effect per 100-um block: improved {conc['blocks_within3_improved_gt0.05']}, worsened {conc['blocks_within3_worsened_gt0.05']} (|d|>0.05)", fontsize=9)
    fig.suptitle("Before (R1a, NCC only) vs after ICP (R2). Proxy residual = mutual-NN pair vector (<=4 um), not a true displacement.")
    fig.tight_layout(); fig.savefig(HERE / "icp_before_after_residual.png", dpi=130); plt.close(fig)


def fig_spatial_vectors():
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    pos, vec = z["pos"], z["vec"]
    for a, win_ in zip(ax[0], (50.0, 100.0, 200.0)):
        c, m, n = NR.window_means(pos, vec - vec.mean(0), win_, min_n=5, origin=np.zeros(2))
        sd = np.sqrt(np.mean(np.sum((vec - vec.mean(0)) ** 2, 1)))
        floor = np.median(sd * np.sqrt(np.pi / 4) / np.sqrt(n))
        _quiver_blocks(a, c, m, 60, COL["R2"], f"R2 proxy residual, {win_:g}-um windows (x60)\nmedian |mean| {np.median(np.linalg.norm(m, axis=1)):.2f} um vs iid noise floor {floor:.2f} um")
    mags = np.linalg.norm(vec, axis=1)
    ax[1, 0].hist(mags, bins=np.arange(0, 4.01, 0.1), color=COL["R2"], alpha=.7, label=f"R2 proxy (n={len(mags)})")
    gi = resid["sources"]["G_R2_B2matched"]
    ax[1, 0].set_xlabel("|residual| (um)"); ax[1, 0].legend(); ax[1, 0].set_title("Residual magnitude distribution (proxy pairs <=4 um, truncated)")
    rel = resid["relations"]
    for a, (key, lab) in zip((ax[1, 1], ax[1, 2]), (("b2_confidence_bins", "B2 confidence"), ("he_white_signed_dist_bins", "distance to HE white edge (um)"))):
        t = pd.DataFrame(rel[key]).dropna()
        a.bar(range(len(t)), t.res_mag_median, color=COL["R2"], alpha=.7, label="median |res|")
        a.plot(range(len(t)), t.frac_gt2um * 4, "ro-", label="frac >2 um (x4)")
        a.set_xticks(range(len(t))); a.set_xticklabels([f"{lo:g}-{hi:g}\n(n={n})" for lo, hi, n in zip(t.lo.clip(-1, None), t.hi.clip(None, 999), t.n)], fontsize=7)
        a.set_xlabel(lab); a.legend(fontsize=8)
    ax[1, 1].set_title("Residual vs B2 confidence (confidence uses distance; not independent)", fontsize=9)
    ax[1, 2].set_title("Residual vs HE white (cleft) boundary distance", fontsize=9)
    fig.tight_layout(); fig.savefig(HERE / "spatial_residual_vectors.png", dpi=130); plt.close(fig)


def fig_autocorr():
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    lags = ["0_20", "20_50", "50_100", "100_200"]; xs = [10, 35, 75, 150]
    for a_ in sorted(semi.amp.unique()):
        s = semi[semi.amp == a_]
        m = [s[f"rho_{l}"].mean() for l in lags]; lo = [s[f"rho_{l}"].quantile(.05) for l in lags]; hi = [s[f"rho_{l}"].quantile(.95) for l in lags]
        ax[0].fill_between(xs, lo, hi, color=plt.cm.Greys(0.25 + a_ / 6), alpha=.35)
        ax[0].plot(xs, m, c=plt.cm.Greys(0.35 + a_ / 6), ls=":", label=f"N3 semi-synth amp {a_:g} um")
    for k, c, lab in (("P_R1a", COL["R1a"], "Real R1a proxy"), ("P_R2", COL["R2"], "Real R2 proxy"), ("G_R2_B2matched", "#542788", "Real R2 B2 matched"),
                      ("N2_mirror_x", "r", "N2 mirror-x (unrelated)")):
        v = resid["sources"][k]
        ax[0].plot(xs, [v[f"rho_{l}"] for l in lags], "o-", c=c, label=lab)
    ax[0].set_xlabel("lag (um, bin centre)"); ax[0].set_ylabel("vector autocorrelation rho"); ax[0].legend(fontsize=7); ax[0].axhline(0, c="k", lw=.5)
    ax[0].set_title("Residual vector spatial autocorrelation")
    st = semi.groupby("amp").heldout_skill_b100_min
    ax[1].errorbar(st.mean().index, st.mean(), yerr=[st.mean() - st.quantile(.05), st.quantile(.95) - st.mean()], fmt="ks-", label="N3 calibration (5-95%)")
    for h in P["heldout"]:
        if h["block"] == 100:
            ax[1].axhline(h["skill"], c=COL["R2"], ls="-", lw=1, label=f"Real R2, colour {h['colour']}: {h['skill']:.3f}")
    ax[1].axhline(resid["sources"]["P_R1a"]["heldout_skill_b100_min"], c=COL["R1a"], ls="--", label="Real R1a (affine error = positive control)")
    ax[1].set_xlabel("injected coherent field RMS (um)"); ax[1].set_ylabel("held-out skill (100-um checker, min of colours)"); ax[1].legend(fontsize=7)
    ax[1].set_title("Held-out predictability vs known field amplitude")
    sd = rdf.copy()
    for k, mk in (("R3", "o"), ("R2", "s")):
        s = sd[sd.affine == k]
        lv = s.truth_field_rms_at_pairs.fillna(0)
        ax[2].scatter(lv + (0.15 if k == "R2" else 0), s.heldout_skill_b100_min, marker=mk, label=f"independent generator, {k}{' (true affine)' if k == 'R3' else ' (A2)'}")
    ax[2].axhline(max(P['heldout'][0]['skill'], P['heldout'][1]['skill']), c=COL["R2"], label="Real R2 (max colour)")
    ax[2].set_xlabel("true non-affine field RMS at proxy pairs (um)"); ax[2].set_ylabel("held-out skill (100 um)"); ax[2].legend(fontsize=7)
    ax[2].set_title("Synthetic check: diagnostic detects known deformation")
    fig.tight_layout(); fig.savefig(HERE / "residual_spatial_autocorrelation.png", dpi=130); plt.close(fig)


def fig_graph():
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    data = [pern[k][pern[k].pipeline_method == B0].top1_graph_support for k in STAGES]
    ax[0].violinplot(data, showmedians=True)
    ax[0].set_xticks(range(1, len(STAGES) + 1)); ax[0].set_xticklabels(STAGES); ax[0].set_ylabel("top-1 Graph support (Position+Graph)")
    ax[0].set_title("Graph support by stage (not accuracy)")
    ks = list(STAGES) + ["R0_saved_subset"]
    m = [drow(k, B2)["matched_n"] for k in ks]; a_ = [drow(k, B2)["ambiguous_n"] for k in ks]; u = [drow(k, B2)["predicted_unmatched_n"] for k in ks]
    ax[1].bar(ks, m, color="#1a9850", label="matched"); ax[1].bar(ks, a_, bottom=m, color="#fee08b", label="ambiguous")
    ax[1].bar(ks, u, bottom=np.add(m, a_), color="#d73027", label="unmatched")
    for i, v in enumerate(m):
        ax[1].text(i, v / 2, str(v), ha="center", fontsize=8)
    ax[1].set_title("B2 status by stage (counts, not correctness)"); ax[1].legend(); plt.setp(ax[1].get_xticklabels(), rotation=20)
    fig.tight_layout(); fig.savefig(HERE / "graph_support_by_stage.png", dpi=130); plt.close(fig)


def fig_matches():
    fig, ax = plt.subplots(1, 3, figsize=(19, 6))
    for a, k in zip(ax[:2], ("R1a", "R2")):
        p = ref[mat[k]]; q = ref[mvC & ~mat[k]]
        a.scatter(q[:, 0], q[:, 1], s=2, c="0.8", label="not matched")
        a.scatter(p[:, 0], p[:, 1], s=2, c=COL[k], label=f"B2 matched ({mat[k].sum()})")
        a.set_aspect("equal"); a.legend(markerscale=5, fontsize=8); a.set_title(f"{k}: B2 matched (plotted at R2 position)")
    v = np.abs(blocks.delta_b2_matched_R1a_to_R2).max()
    sc = ax[2].scatter(blocks.center_x, blocks.center_y, c=blocks.delta_b2_matched_R1a_to_R2, cmap="PiYG", vmin=-v, vmax=v, s=260, marker="s")
    plt.colorbar(sc, ax=ax[2], label="B2 matched: R2 - R1a per 100-um block")
    ax[2].set_aspect("equal")
    ax[2].set_title(f"Increment +{conc['total_increment_R1a_to_R2']}: up in {conc['blocks_increase']}, down in {conc['blocks_decrease']} blocks;\n"
                    f"Gini {conc['gini_positive_increment']:.2f}, top-10% blocks hold {conc['top10pct_blocks_share_of_positive_increment']:.0%}", fontsize=9)
    fig.tight_layout(); fig.savefig(HERE / "accepted_matches_spatial_map.png", dpi=130); plt.close(fig)


def fig_synthetic():
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    st = ["R1a", "R1b", "R2s", "R2", "R3"]
    for i, k in enumerate(st):
        for gi, g in enumerate(GORDER):
            v = T[(T.group == g) & (T.stage == k)].rms_common_um.clip(lower=0.02)
            ax[0].scatter(np.full(len(v), gi + (i - 2) * 0.15), v, s=14, c=COL[k], label=k if gi == 0 else None)
    ax[0].set_yscale("log"); ax[0].axhline(5, c="r", ls=":", lw=1); ax[0].set_xticks(range(len(GORDER))); ax[0].set_xticklabels(GORDER, rotation=25, ha="right", fontsize=8)
    ax[0].set_ylabel("affine RMS error, common region (um)"); ax[0].legend(); ax[0].set_title("Test (56 new pairs): affine error by stage")
    for i, k in enumerate(st):
        v = [T[(T.group == g) & (T.stage == k)].true_pairs_within3.median() for g in GORDER]
        ax[1].bar(np.arange(len(GORDER)) + (i - 2) * 0.16, v, 0.16, color=COL[k], label=k)
    ax[1].set_xticks(range(len(GORDER))); ax[1].set_xticklabels(GORDER, rotation=25, ha="right", fontsize=8)
    ax[1].set_ylabel("true pairs within 3 um (median)"); ax[1].legend(); ax[1].set_title("True-match spatial residual (R2 > R3 = fits deformation/noise)")
    for i, k in enumerate(("R1a", "R2", "R3")):
        v = [T[(T.group == g) & (T.stage == k)].graph_top1_accuracy.median() for g in GORDER]
        ax[2].bar(np.arange(len(GORDER)) + (i - 1) * 0.25, v, 0.25, color=COL[k], label=k)
    ax[2].set_xticks(range(len(GORDER))); ax[2].set_xticklabels(GORDER, rotation=25, ha="right", fontsize=8)
    ax[2].set_ylabel("Graph top-1 accuracy (median)"); ax[2].legend(); ax[2].set_title("Downstream accuracy by stage")
    fig.tight_layout(); fig.savefig(HERE / "synthetic_affine_vs_nonrigid_error.png", dpi=130); plt.close(fig)


for f in (fig_overlay, fig_nn, fig_icp_before_after, fig_spatial_vectors, fig_autocorr, fig_graph, fig_matches, fig_synthetic):
    f()
print(real_tab.set_index("stage")[["common_m2f_within_3um", "common_sym_nn_median_um", "corr200_local_residual_median_um", "graph_support_median",
                                   "b2_matched", "heldout_checker_within3_mean"]].to_string())
print(json.dumps(conc, indent=1)); print(json.dumps(nonrigid_decision, indent=1, default=str)); print(icp_by_group.to_string()); print(dev_sum)
print(rd_sum.to_string()); print(syn_sum.to_string())
