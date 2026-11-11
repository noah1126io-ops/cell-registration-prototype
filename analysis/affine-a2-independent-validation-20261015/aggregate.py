"""Aggregate ./cache/*.json into the deliverable CSV/JSON files and figures (no algorithm is re-run here)."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import design as D  # noqa: E402
import indep_gen as G  # noqa: E402

C = HERE / "cache"
FAM_NAME = G.FAMILIES
MCOL = {"A0": "#c0392b", "A2": "#d68910", "A3": "#1e8449", "TRUE": "#5d6d7e"}
FCOL = {"G1": "#1f77b4", "G2": "#9467bd", "G3": "#2ca02c", "G4": "#d62728"}
FMARK = {"G1": "o", "G2": "s", "G3": "^", "G4": "D"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, max(0.0, c - h), min(1.0, c + h))


def load(pattern):
    return [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(C / pattern)))]


PARAMS = ("offset_um", "rot_deg", "reflect", "pixel_um", "q_shared", "det_fixed", "det_moving", "deform_rms_um", "density",
          "loss_patches", "hetero", "n_fixed", "n_moving", "shared_fraction_of_moving", "shared_fraction_of_fixed",
          "overlap_frac_fixed", "overlap_frac_moving", "overlap_iou", "overlap_min_coverage", "centroid_shift_um", "observed_fixed_density")
METRICS = ("objective", "runtime_s", "peak_mem_mb", "detector_status", "overlap_moving_fraction", "inlier3", "inlier3_null",
           "inlier_over_null", "block_consistency", "basin_margin", "basin_margin_is_inf", "rms_common_um", "max_common_um",
           "n_common_grid", "transform_rms_error_um", "transform_max_error_um", "translation_error_center_um", "rotation_error_deg",
           "chirality_correct", "scale_error_pct", "true_pair_distance_median_um", "true_pairs_within3", "affine_success")


def init_error(d, m):
    """Min common-region RMS of the pre-refine similarity hypotheses that A1/A2/A3 kept (NaN for A0)."""
    if "params" not in d or m["method_key"] == "A0":
        return np.nan
    pair = _pair_cache(d)
    errs = []
    for b in m.get("basins", []):
        h = b[3]
        if "theta" not in h:
            continue
        C = np.diag([1.0, -1.0]) if h["chirality"] == "reflect" else np.eye(2)
        a = np.deg2rad(h["theta"])
        M0 = h["scale"] * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ C
        errs.append(G.affine_errors_common(M0, np.asarray(h["translation"]), pair)["rms_common_um"])
    return float(np.nanmin(errs)) if errs else np.nan


_PAIRS = {}


def _pair_cache(d):
    if d["name"] not in _PAIRS:
        p = d["params"]
        if d.get("kind") == "main":
            _PAIRS[d["name"]] = G.make_pair(p["family"], p["condition"], p["rep"], D.CONDITIONS[p["condition"]])
        else:
            name, fam, rep, ov, mode, _, _ = D.CONTROLS[d["index"]]
            _PAIRS[d["name"]] = G.make_pair(fam, name, rep, ov, mode=mode, name=f"{fam}:{name}:{rep}")
    return _PAIRS[d["name"]]


def flatten(d, extra):
    rows = []
    for m in d["methods"]:
        r = {"case": d["name"], "family": d["family"], "family_name": FAM_NAME[d["family"]], **extra,
             "method_key": m["method_key"], "truth_basin_objective": d["truth_basin_objective"]}
        r.update({f"p_{k}": d["params"].get(k) for k in PARAMS})
        r.update({k: m.get(k) for k in METRICS})
        dg = m.get("diagnostics", {})
        r["peak_ncc"], r["second_peak_ncc"], r["peak_robust_z"] = dg.get("peak_ncc"), dg.get("second_peak_ncc"), dg.get("peak_robust_z")
        r["ncc_peak_margin"] = (r["peak_ncc"] - r["second_peak_ncc"]) if r["peak_ncc"] is not None and r["second_peak_ncc"] is not None else np.nan
        rb = [x for x in m.get("refined_basin_rms_common_um", []) if x is not None]
        r["n_distinct_basins"] = len(m.get("basins", []))
        r["truth_basin_refined"] = bool(rb and min(rb) <= D.AFFINE_SUCCESS_RMS_UM)
        r["init_rms_common_min_um"] = init_error(d, m)
        to = d["truth_basin_objective"]
        # post-hoc diagnostic taxonomy (does not change the pre-registered success definition)
        if r["affine_success"]:
            r["failure_type"] = "success"
        elif not r["chirality_correct"]:
            r["failure_type"] = "wrong chirality"
        elif r["rms_common_um"] <= 20 and abs(r["objective"] - to) <= 0.02 * to:
            r["failure_type"] = "imprecise: truth basin, deformation bias (RMS 5-20)"
        elif r["truth_basin_refined"]:
            r["failure_type"] = "selection: truth basin refined but worse objective"
        elif np.isfinite(r["init_rms_common_min_um"]) and r["init_rms_common_min_um"] <= 20:
            r["failure_type"] = "refine drift: NCC init near truth, ICP left the basin"
        else:
            r["failure_type"] = "initialisation: no hypothesis near truth"
        r["detector_pass"] = r["detector_status"] == "PASS"
        rows.append(r)
    return rows


main = pd.DataFrame([r for d in load("main_*.json") for r in flatten(d, {"kind": "main", "condition": d["condition"], "rep": d["rep"],
                                                                              "factor": d["factor"], "index": d["index"]})])
ctl = pd.DataFrame([r for d in load("control_*.json") for r in flatten(d, {"kind": "control", "control": d["control"], "expect": d["expect"],
                                                                              "restrict_chirality": d["restrict_chirality"], "index": d["index"],
                                                                              "rep": d["params"]["rep"]})])
main["oor_centroid"] = main.p_centroid_shift_um > 200
main.to_csv(HERE / "affine_accuracy.csv", index=False)


def is_false_accept(r):
    if r["expect"] in ("no_shared", "wrong_chir"):
        return bool(r["detector_pass"])
    return bool(r["detector_pass"] and not r["affine_success"])


ctl["false_accept"] = ctl.apply(is_false_accept, axis=1)
ctl.to_csv(HERE / "negative_control_metrics.csv", index=False)

# ----------------------------------------------------------------------------- detector confusion
conf_rows = []


def confusion(df, method, stratum, value):
    s = df[df.method_key == method]
    if s.empty:
        return
    ok, ps = s.affine_success.astype(bool), s.detector_pass.astype(bool)
    ta, fr, fa, cr = int((ok & ps).sum()), int((ok & ~ps).sum()), int((~ok & ps).sum()), int((~ok & ~ps).sum())
    n_ok, n_bad = ta + fr, fa + cr
    row = {"method_key": method, "stratum": stratum, "value": value, "n": len(s), "true_accept": ta, "false_reject": fr,
           "false_accept": fa, "correct_reject": cr, "n_correct_affine": n_ok, "n_wrong_affine": n_bad,
           "n_pass": int(ps.sum()), "n_ambiguous": int((s.detector_status == "AMBIGUOUS").sum()), "n_fail": int((s.detector_status == "FAIL").sum())}
    for name, k, n in (("true_acceptance_rate", ta, n_ok), ("false_rejection_rate", fr, n_ok), ("false_acceptance_rate", fa, n_bad),
                       ("pass_precision", ta, ta + fa), ("fail_or_ambiguous_rate", fr + cr, len(s))):
        p, lo, hi = wilson(k, n)
        row.update({name: p, f"{name}_ci95_lo": lo, f"{name}_ci95_hi": hi})
    row["false_acceptance_rule_of_three_upper"] = 3 / n_bad if fa == 0 and n_bad > 0 else np.nan
    conf_rows.append(row)


for mk in ("A2", "A0", "A3"):
    confusion(main, mk, "all_main", "all")
    for f in D.FAMILIES:
        confusion(main[main.family == f], mk, "family", f)
    for fac in sorted(main.factor.unique()):
        confusion(main[main.factor == fac], mk, "factor", fac)
    for c in D.CONDITIONS:
        confusion(main[main.condition == c], mk, "condition", c)
    truth_ctl = ctl[ctl.expect == "truth"]
    confusion(pd.concat([main, truth_ctl]), mk, "main+truth_controls", "all")
    confusion(truth_ctl, mk, "truth_controls", "NC3/NC4/NC6")
conf = pd.DataFrame(conf_rows)
# no-shared / wrong-chirality controls: every PASS is a false accept
nc_rows = []
for mk in ("A2", "A0", "A3"):
    for name, s in [("no_shared+wrong_chir", ctl[(ctl.method_key == mk) & ctl.expect.isin(["no_shared", "wrong_chir"])])] + \
            [(c, ctl[(ctl.method_key == mk) & (ctl.control == c)]) for c in sorted(ctl.control.unique())]:
        if s.empty:
            continue
        k = int(s.false_accept.sum()); p, lo, hi = wilson(k, len(s))
        nc_rows.append({"method_key": mk, "stratum": "control", "value": name, "n": len(s), "false_accept": k, "false_acceptance_rate": p,
                        "false_acceptance_rate_ci95_lo": lo, "false_acceptance_rate_ci95_hi": hi,
                        "false_acceptance_rule_of_three_upper": 3 / len(s) if k == 0 else np.nan,
                        "n_pass": int(s.detector_pass.sum()), "n_ambiguous": int((s.detector_status == "AMBIGUOUS").sum()),
                        "n_fail": int((s.detector_status == "FAIL").sum()), "affine_success_n": int(s.affine_success.sum())})
conf = pd.concat([conf, pd.DataFrame(nc_rows)], ignore_index=True)
conf.to_csv(HERE / "detector_confusion_matrix.csv", index=False)

# ----------------------------------------------------------------------------- downstream
ds_rows, ds_dist = [], []
for d in load("downstream_[0-9]*.json"):
    for r in d["rows"]:
        ds_rows.append({"kind": d["kind"], "index": d["index"], "case": d["name"], "affine_key": d["affine_key"], **r})
    if d["dist"]:
        ds_dist.append({"kind": d["kind"], "index": d["index"], "case": d["name"], "affine_key": d["affine_key"], **d["dist"]})
ds = pd.DataFrame(ds_rows)
B2 = "Separated Matching / Adoption"
if not ds.empty:
    key = main[["case", "method_key", "affine_success", "detector_status", "rms_common_um", "family", "condition", "factor"]]
    ds = ds.merge(key.rename(columns={"method_key": "affine_key"}), on=["case", "affine_key"], how="left")
    tr = ds.affine_key == "TRUE"
    ds.loc[tr, "affine_success"] = True
    ds.loc[tr, "rms_common_um"] = 0.0
    fam = main.drop_duplicates("case").set_index("case")
    for col in ("family", "condition", "factor"):
        ds.loc[tr, col] = ds.loc[tr, "case"].map(fam[col])
    ds["graph_success"] = (ds.top1_accuracy >= D.GRAPH_SUCCESS["top1_min"]) & (ds.matched_precision >= D.GRAPH_SUCCESS["b2_precision_min"])

    def state(r):
        a, g = bool(r.affine_success), bool(r.graph_success)
        return {(True, True): "1 correct affine + graph success", (True, False): "2 correct affine + graph failure",
                (False, False): "3 wrong affine + graph failure", (False, True): "4 wrong affine + graph success"}[(a, g)]
    ds["state"] = ds.apply(state, axis=1)
    ctl_ds = ds.kind == "control"            # no-shared controls: no truth affine/ids -> not part of the state table
    ds.loc[ctl_ds, "state"] = "control (no shared nuclei)"
    cstat = ctl[ctl.method_key == "A2"].set_index("case").detector_status
    ds.loc[ctl_ds, "detector_status"] = ds.loc[ctl_ds, "case"].map(cstat)
    # operational adoption: detector FAIL/AMBIGUOUS stops the pipeline (no matches adopted)
    stop = (ds.affine_key != "TRUE") & (ds.detector_status != "PASS")
    ds["operational_stopped"] = stop
    ds["operational_matched_n"] = np.where(stop, 0, ds.matched_n)
    ds["operational_wrong_matched_n"] = np.where(stop, 0, ds.wrong_matched_n)
    ds["operational_coverage"] = np.where(stop, 0.0, ds.coverage)
ds.to_csv(HERE / "downstream_correspondence_metrics.csv", index=False)

# ----------------------------------------------------------------------------- real + domain gap
real = json.loads((C / "real_a2.json").read_text()) if (C / "real_a2.json").exists() else None
real_ds = json.loads((C / "downstream_real_a2.json").read_text()) if (C / "downstream_real_a2.json").exists() else None
dist = pd.DataFrame(ds_dist)
gap_rows = []
if real is not None and real_ds is not None and not dist.empty:
    rr = {r["pipeline_method"]: r for r in real_ds["rows"]}
    rdist = real_ds["dist"]
    qi = rdist["quantiles"].index(0.5)

    def case_metrics(drow, dsub, arow):
        b0 = dsub.get("Position + Graph", {}); b1 = dsub.get("Relative-cost Graph-UOT", {}); b2 = dsub.get(B2, {})
        return {"fixed_density_subset": drow["density_fixed_subset"], "knn_median_um": drow["knn_um_q"][qi],
                "candidate_distance_median_um": drow["candidate_distance_um_q"][qi], "graph_support_median": drow["graph_support_q"][qi],
                "uot_row_mass_median": drow["row_mass_ratio_q"][qi], "b2_confidence_median": drow["b2_confidence_q"][qi],
                "b2_coverage": b2.get("coverage"), "ncc18_boundary_rate": b0.get("ncc18_boundary_rate"),
                "peak_ncc": arow.get("peak_ncc"), "ncc_peak_margin": arow.get("ncc_peak_margin"), "inlier3": arow.get("inlier3"),
                "block_consistency": arow.get("block_consistency"), "icp_objective_um": arow.get("objective")}
    dg = real["diagnostics"]
    real_vals = case_metrics(rdist, rr, {"peak_ncc": dg.get("peak_ncc"), "ncc_peak_margin": dg["peak_ncc"] - dg["second_peak_ncc"],
                                         "inlier3": real["inlier3"], "block_consistency": real["block_consistency"], "objective": real["objective"]})
    syn_case = []
    a2 = main[main.method_key == "A2"].set_index("case")
    for _, drow in dist[(dist.kind == "main") & (dist.affine_key == "A2")].iterrows():
        sub = {r["pipeline_method"]: r for r in ds_rows if r["tag"] == f"{drow.case}:A2"}
        ar = a2.loc[drow.case]
        v = case_metrics(drow, sub, ar.to_dict())
        syn_case.append({"case": drow.case, "family": ar.family, "affine_success": bool(ar.affine_success),
                         "detector_pass": bool(ar.detector_pass), **v})
    syn_case = pd.DataFrame(syn_case)
    syn_case.to_csv(HERE / "cache/synthetic_case_domain_metrics.csv", index=False)
    for metric, rv in real_vals.items():
        for grp, sub in [("all A2 PASS", syn_case[syn_case.detector_pass])] + \
                [(f"{f} A2 PASS", syn_case[(syn_case.family == f) & syn_case.detector_pass]) for f in D.FAMILIES] + \
                [("all A2 (incl. failures)", syn_case)]:
            x = pd.to_numeric(sub[metric], errors="coerce").dropna().to_numpy()
            if not len(x):
                continue
            q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
            gap_rows.append({"metric": metric, "synthetic_group": grp, "n_cases": len(x), "real_a2_value": rv,
                             "syn_p05": q[0], "syn_p25": q[1], "syn_median": q[2], "syn_p75": q[3], "syn_p95": q[4],
                             "real_percentile_in_synthetic": float(np.mean(x < rv) + 0.5 * np.mean(x == rv)) if rv is not None else np.nan,
                             "real_outside_p05_p95": bool(rv is not None and (rv < q[0] or rv > q[4]))})
gap = pd.DataFrame(gap_rows)
gap.to_csv(HERE / "synthetic_real_distribution.csv", index=False)

# ----------------------------------------------------------------------------- runtime
rt_rows = []
for mk in ("A0", "A2", "A3"):
    for grp, s in [("all_main", main[main.method_key == mk])] + [(f, main[(main.method_key == mk) & (main.family == f)]) for f in D.FAMILIES] + \
            [("controls", ctl[ctl.method_key == mk])]:
        if s.empty:
            continue
        rt_rows.append({"method_key": mk, "group": grp, "n": len(s), "runtime_median_s": s.runtime_s.median(),
                        "runtime_p90_s": s.runtime_s.quantile(0.9), "runtime_max_s": s.runtime_s.max(),
                        "peak_traced_mem_median_mb": s.peak_mem_mb.median(), "peak_traced_mem_max_mb": s.peak_mem_mb.max(),
                        "n_moving_median": s.p_n_moving.median() if "p_n_moving" in s else np.nan})
if real is not None:
    rt_rows.append({"method_key": "A2", "group": "real", "n": 1, "runtime_median_s": real["runtime_s"], "runtime_max_s": real["runtime_s"]})
pd.DataFrame(rt_rows).to_csv(HERE / "runtime_metrics.csv", index=False)

# ----------------------------------------------------------------------------- headline table
head = []
for f in D.FAMILIES:
    a2 = main[(main.method_key == "A2") & (main.family == f)]
    b2 = ds[(ds.affine_key == "A2") & (ds.family == f) & (ds.pipeline_method == B2)] if not ds.empty else pd.DataFrame()
    head.append({"family": f, "generator": FAM_NAME[f], "cases": len(a2), "a2_affine_success": int(a2.affine_success.sum()),
                 "a2_affine_success_rate": a2.affine_success.mean(),
                 "detector_false_accept": int((a2.detector_pass & ~a2.affine_success.astype(bool)).sum()),
                 "n_wrong_affine": int((~a2.affine_success.astype(bool)).sum()),
                 "true_accept": int((a2.detector_pass & a2.affine_success.astype(bool)).sum()),
                 "n_correct_affine": int(a2.affine_success.sum()),
                 "a0_affine_success": int(main[(main.method_key == "A0") & (main.family == f)].affine_success.sum()),
                 "a3_affine_success": f"{int(main[(main.method_key == 'A3') & (main.family == f)].affine_success.sum())}/{len(main[(main.method_key == 'A3') & (main.family == f)])}",
                 "downstream_top1_median": b2.top1_accuracy.median() if len(b2) else np.nan,
                 "downstream_top1_median_pass_only": b2[b2.detector_status == "PASS"].top1_accuracy.median() if len(b2) else np.nan,
                 "matched_precision_pooled": (b2.matched_n - b2.wrong_matched_n).sum() / b2.matched_n.sum() if len(b2) and b2.matched_n.sum() else np.nan,
                 "matched_precision_pooled_pass_only": ((b2.matched_n - b2.wrong_matched_n)[b2.detector_status == "PASS"].sum() /
                                                        b2.matched_n[b2.detector_status == "PASS"].sum()) if len(b2) and b2.matched_n[b2.detector_status == "PASS"].sum() else np.nan})
headline = pd.DataFrame(head)
headline.to_csv(HERE / "cache/headline_by_generator.csv", index=False)

# condition x family success (A2)
cond_tab = main[main.method_key == "A2"].pivot_table(index="condition", columns="family", values="affine_success", aggfunc="sum")
cond_tab.to_csv(HERE / "cache/a2_success_condition_by_family.csv")
for mk in ("A0", "A3"):
    main[main.method_key == mk].pivot_table(index="condition", columns="family", values="affine_success", aggfunc="sum").to_csv(HERE / f"cache/{mk.lower()}_success_condition_by_family.csv")
main[main.method_key == "A2"].groupby(["condition", "failure_type"]).size().unstack(fill_value=0).to_csv(HERE / "cache/a2_failure_type_by_condition.csv")
main[main.method_key == "A2"].groupby(["condition", "detector_status", "affine_success"]).size().unstack(fill_value=0).to_csv(HERE / "cache/a2_detector_by_condition.csv")

# ----------------------------------------------------------------------------- figures
def fig_examples():
    fig, ax = plt.subplots(2, 4, figsize=(18, 9.2))
    for i, f in enumerate(D.FAMILIES):
        p = G.make_pair(f, "C01_base", 0, D.CONDITIONS["C01_base"])
        w = p.moving_world_true
        a = ax[0, i]
        a.scatter(*p.fixed.T, s=1.2, c="k", lw=0, label="fixed")
        a.scatter(*w.T, s=1.2, c="#e74c3c", lw=0, alpha=0.6, label="moving (true affine)")
        a.plot(*np.vstack([p.fixed_fov, p.fixed_fov[:1]]).T, "k-", lw=0.8)
        a.plot(*np.vstack([p.moving_fov_world, p.moving_fov_world[:1]]).T, "-", c="#e74c3c", lw=0.8)
        a.set_title(f"{f} {FAM_NAME[f]}  (C01_base rep0)\nfixed {len(p.fixed)}, moving {len(p.moving_px)}, shared {p.params['shared_fraction_of_moving']:.2f}", fontsize=9)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        b = ax[1, i]
        sh = p.true_fixed_index >= 0
        lo = np.array([400, 400]); hi = lo + 200
        sel_f = np.all((p.fixed > lo) & (p.fixed < hi), 1); sel_m = np.all((w > lo) & (w < hi), 1)
        b.scatter(*p.fixed[sel_f].T, s=10, c="k", lw=0)
        b.scatter(*w[sel_m & ~sh].T, s=10, facecolors="none", edgecolors="#e74c3c", lw=0.8)
        b.scatter(*w[sel_m & sh].T, s=10, c="#e74c3c", lw=0)
        for j in np.flatnonzero(sel_m & sh):
            b.plot([w[j, 0], p.fixed[p.true_fixed_index[j], 0]], [w[j, 1], p.fixed[p.true_fixed_index[j], 1]], c="#888", lw=0.5)
        b.set_xlim(lo[0], hi[0]); b.set_ylim(lo[1], hi[1]); b.set_aspect("equal")
        b.set_title("200 µm zoom: black fixed, red moving\n(filled = shared id, hollow = moving-only)", fontsize=8)
    ax[0, 0].legend(markerscale=6, fontsize=7, loc="lower left")
    fig.suptitle("Independent generator examples (2D partial-sharing model; artificial tissue-like patterns, not real histology)", fontsize=11)
    fig.tight_layout(); fig.savefig(HERE / "synthetic_generator_examples.png", dpi=110); plt.close(fig)


def fig_success():
    fig, ax = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.25, 1]})
    a2 = main[main.method_key == "A2"].pivot_table(index="condition", columns="family", values="affine_success", aggfunc="mean")
    im = ax[0].imshow(a2.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    n = main[main.method_key == "A2"].pivot_table(index="condition", columns="family", values="affine_success", aggfunc="size")
    k = main[main.method_key == "A2"].pivot_table(index="condition", columns="family", values="affine_success", aggfunc="sum")
    for i in range(a2.shape[0]):
        for j in range(a2.shape[1]):
            ax[0].text(j, i, f"{int(k.values[i, j])}/{int(n.values[i, j])}", ha="center", va="center", fontsize=9)
    ax[0].set_xticks(range(a2.shape[1]), [f"{c}\n{FAM_NAME[c]}" for c in a2.columns], fontsize=8)
    ax[0].set_yticks(range(a2.shape[0]), a2.index, fontsize=8)
    ax[0].set_title("A2 affine success (common-region RMS ≤5 µm & chirality) by condition × generator")
    fig.colorbar(im, ax=ax[0], fraction=0.03)
    x = np.arange(len(D.FAMILIES)); wd = 0.26
    for o, mk in zip((-wd, 0, wd), ("A0", "A2", "A3")):
        s = main[main.method_key == mk]
        rates = [wilson(int(s[s.family == f].affine_success.sum()), int((s.family == f).sum())) for f in D.FAMILIES]
        p = np.array([r[0] for r in rates]); lo = np.array([r[1] for r in rates]); hi = np.array([r[2] for r in rates])
        ax[1].bar(x + o, p, wd, color=MCOL[mk], label=f"{mk} (n/gen {int((s.family == 'G1').sum())})", yerr=[p - lo, hi - p], capsize=3)
    ax[1].set_xticks(x, [f"{f}\n{FAM_NAME[f]}" for f in D.FAMILIES], fontsize=8); ax[1].set_ylim(0, 1.05)
    ax[1].set_ylabel("affine success rate (Wilson 95% CI)"); ax[1].legend(fontsize=8)
    ax[1].set_title("By generator (A3: replicate 0 only)")
    fig.tight_layout(); fig.savefig(HERE / "affine_success_by_generator.png", dpi=110); plt.close(fig)


def _scatter_err(axis, xcol, xlabel, s, vline=None):
    for f in D.FAMILIES:
        q = s[s.family == f]
        ok = q.affine_success.astype(bool); ps = q.detector_pass.astype(bool)
        axis.scatter(q[xcol][ok & ps], q.rms_common_um[ok & ps].clip(lower=0.05), marker=FMARK[f], c=FCOL[f], s=30, label=f"{f} PASS" if True else None)
        axis.scatter(q[xcol][~ps], q.rms_common_um[~ps].clip(lower=0.05), marker=FMARK[f], facecolors="none", edgecolors=FCOL[f], s=30)
        fa = ~ok & ps
        if fa.any():
            axis.scatter(q[xcol][fa], q.rms_common_um[fa], marker="X", c="k", s=90, zorder=5)
    axis.axhline(5, c="k", ls="--", lw=0.8)
    if vline is not None:
        axis.axvline(vline, c="gray", ls=":", lw=1)
    axis.set_yscale("log"); axis.set_xlabel(xlabel); axis.set_ylabel("A2 common-region RMS error (µm, log)")


def fig_overlap():
    s = main[main.method_key == "A2"]
    sc = pd.concat([s, ctl[(ctl.method_key == "A2") & (ctl.expect == "truth")]])
    fig, ax = plt.subplots(1, 3, figsize=(20, 5.8))
    _scatter_err(ax[0], "p_overlap_min_coverage", "pre-registered: FOV overlap (intersection / larger FOV area)", sc)
    _scatter_err(ax[1], "p_overlap_frac_moving", "post-hoc: moving-side coverage (intersection / moving FOV area)", sc, vline=0.7)
    _scatter_err(ax[2], "p_centroid_shift_um", "point-cloud centroid shift after true alignment (µm)", sc, vline=200)
    ax[2].text(203, 1000, "A2 search ±200 µm", fontsize=8, color="gray")
    ax[0].legend(fontsize=7, ncol=2)
    fig.suptitle("A2 error vs spatial overlap (main + NC3/NC4/NC6). filled = detector PASS, hollow = FAIL/AMBIGUOUS, black X = false accept", fontsize=10)
    fig.tight_layout(); fig.savefig(HERE / "affine_error_vs_overlap.png", dpi=110); plt.close(fig)


def fig_shared():
    s = main[main.method_key == "A2"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    _scatter_err(ax[0], "p_shared_fraction_of_moving", "observed shared fraction of moving nuclei", s)
    q = s[s.condition.isin(["C01_base", "C07_shared_low", "C08_detect_loss"])]
    for c, col in (("C01_base", "#555"), ("C07_shared_low", "#2e86c1"), ("C08_detect_loss", "#ca6f1e")):
        z = q[q.condition == c]
        ax[1].scatter(z.p_shared_fraction_of_moving, z.inlier3, c=col, label=c, s=35)
    ax[1].axhline(0.25, c="k", ls="--", lw=0.8); ax[1].text(0.41, 0.26, "PASS min inlier3 0.25", fontsize=8)
    ax[1].set_xlabel("observed shared fraction of moving nuclei"); ax[1].set_ylabel("detector inlier3 (≤3 µm)"); ax[1].legend(fontsize=8)
    ax[1].set_title("Low q_shared vs detection loss (matched observed sharing)", fontsize=9)
    ax[0].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(HERE / "affine_error_vs_shared_fraction.png", dpi=110); plt.close(fig)


def fig_confusion():
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))
    for a, (title, s) in zip(ax, [("A2 main (112)", main[main.method_key == "A2"]), ("A0 main (112)", main[main.method_key == "A0"]),
                                  ("A3 main rep0 (56)", main[main.method_key == "A3"]),
                                  ("A2 truth controls NC3/NC4/NC6", ctl[(ctl.method_key == "A2") & (ctl.expect == "truth")])]):
        ok = s.affine_success.astype(bool); ps = s.detector_pass.astype(bool)
        m = np.array([[(ok & ps).sum(), (ok & ~ps).sum()], [(~ok & ps).sum(), (~ok & ~ps).sum()]])
        a.imshow([[1, 0.3], [0, 1]], cmap="RdYlGn", vmin=0, vmax=1)
        lab = [["true accept", "false reject"], ["FALSE ACCEPT", "correct reject"]]
        for i in range(2):
            for j in range(2):
                a.text(j, i, f"{lab[i][j]}\n{m[i, j]}", ha="center", va="center", fontsize=10, weight="bold" if (i, j) == (1, 0) else None)
        a.set_xticks([0, 1], ["PASS", "AMBIG/FAIL"]); a.set_yticks([0, 1], ["correct affine", "wrong affine"]); a.set_title(title, fontsize=10)
    nc = ctl[(ctl.method_key == "A2") & ctl.expect.isin(["no_shared", "wrong_chir"])]
    fig.suptitle(f"Frozen detector on the independent generator.  No-shared / wrong-chirality controls (A2): PASS {int(nc.detector_pass.sum())}/{len(nc)}", fontsize=11)
    fig.tight_layout(); fig.savefig(HERE / "detector_confusion_matrix.png", dpi=110); plt.close(fig)


def _overlay(a, pair, M, t, title):
    w = np.asarray(pair.moving_px) @ np.asarray(M).T + np.asarray(t)
    wt = pair.moving_world_true
    a.scatter(*pair.fixed.T, s=1, c="k", lw=0)
    a.scatter(*w.T, s=1, c="#e74c3c", lw=0, alpha=0.6)
    a.plot(*np.vstack([pair.fixed_fov, pair.fixed_fov[:1]]).T, "k-", lw=0.8)
    poly = pair.moving_fov_world
    a.plot(*np.vstack([poly, poly[:1]]).T, "--", c="#2e86c1", lw=1, label="true moving FOV")
    corners = G.rect(0, 0, pair.width, pair.height) @ np.asarray(M).T + t
    a.plot(*np.vstack([corners, corners[:1]]).T, "-", c="#e74c3c", lw=1, label="estimated moving FOV")
    a.set_aspect("equal"); a.set_title(title, fontsize=8); a.set_xticks([]); a.set_yticks([])


def fig_false_accepts():
    allr = pd.concat([main, ctl])
    fa = allr[allr.detector_pass & ((~allr.affine_success.astype(bool)) | allr.get("expect", pd.Series(index=allr.index)).isin(["no_shared", "wrong_chir"]))]
    note = "false accepts (detector PASS with wrong affine or without cell-level correspondence)"
    if fa.empty:
        bad = allr[~allr.affine_success.astype(bool)].copy()
        fa = bad.sort_values("inlier_over_null", ascending=False).head(6)
        note = "NO false accept occurred. Shown: the 6 wrong-affine results with the highest inlier/null ratio (closest calls)"
    fa = fa.head(8)
    nc = max(1, min(4, len(fa))); nr = int(np.ceil(len(fa) / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(4.6 * nc, 4.6 * nr), squeeze=False)
    for a, (_, r) in zip(ax.ravel(), fa.iterrows()):
        if r.kind == "main":
            pair = G.make_pair(r.family, r.condition, int(r.rep), D.CONDITIONS[r.condition])
            src = json.loads((C / f"main_{int(r['index']):03d}.json").read_text())
        else:
            name, fam, rep, ov, mode, _, _ = D.CONTROLS[int(r["index"])]
            pair = G.make_pair(fam, name, rep, ov, mode=mode, name=f"{fam}:{name}:{rep}")
            src = json.loads((C / f"control_{int(r['index']):02d}.json").read_text())
        m = next(x for x in src["methods"] if x["method_key"] == r.method_key)
        _overlay(a, pair, m["M"], m["t"], f"{r.case} {r.method_key} [{r.detector_status}]\nRMS {r.rms_common_um:.1f} µm, inlier3 {r.inlier3:.2f}, "
                                          f"null ratio {r.inlier_over_null:.1f}, block {r.block_consistency:.2f}")
    for a in ax.ravel()[len(fa):]:
        a.axis("off")
    ax.ravel()[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(note, fontsize=10)
    fig.tight_layout(); fig.savefig(HERE / "false_accept_examples.png", dpi=105); plt.close(fig)


def fig_graph():
    if ds.empty:
        return
    b2 = ds[ds.pipeline_method == B2]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    for mk in ("A0", "A2", "TRUE"):
        s = b2[(b2.affine_key == mk) & (b2.kind == "main")]
        ax[0].scatter(s.rms_common_um.clip(lower=0.05) * (1 if mk != "TRUE" else 1), s.top1_accuracy, c=MCOL[mk], s=22, alpha=0.75,
                      label=f"{mk} (n={len(s)})", marker="o" if mk != "TRUE" else "|")
    ax[0].set_xscale("log"); ax[0].axvline(5, c="k", ls="--", lw=0.8); ax[0].axhline(0.8, c="gray", ls=":", lw=0.8)
    ax[0].set_xlabel("common-region affine RMS error (µm, log; TRUE plotted at 0.05)"); ax[0].set_ylabel("top-1 accuracy (common cells in subset)")
    ax[0].legend(fontsize=8); ax[0].set_title("Downstream true-ID top-1 vs affine error")
    st = b2[b2.affine_key.isin(["A0", "A2", "TRUE"]) & (b2.kind == "main")].groupby(["affine_key", "state"]).size().unstack(fill_value=0)
    st.plot.barh(stacked=True, ax=ax[1], color=["#27ae60", "#f39c12", "#c0392b", "#8e44ad"][: st.shape[1]])
    ax[1].set_xlabel("cases"); ax[1].set_title("Affine × Graph states (graph success: top-1 ≥0.80 & B2 precision ≥0.90)", fontsize=9)
    ax[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(HERE / "graph_accuracy_vs_affine_error.png", dpi=110); plt.close(fig)


def fig_gap():
    if gap.empty:
        return
    sc = pd.read_csv(HERE / "cache/synthetic_case_domain_metrics.csv")
    metrics = list(dict.fromkeys(gap.metric))
    nc = 5; nr = int(np.ceil(len(metrics) / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(3.6 * nc, 3.2 * nr), squeeze=False)
    rv = gap.drop_duplicates("metric").set_index("metric").real_a2_value
    for a, mname in zip(ax.ravel(), metrics):
        for j, f in enumerate(D.FAMILIES):
            v = sc[(sc.family == f)]
            y = pd.to_numeric(v[mname], errors="coerce")
            jit = np.random.default_rng(j).uniform(-0.18, 0.18, len(v))
            a.scatter(j + jit, y, s=9, c=np.where(v.detector_pass, FCOL[f], "#cccccc"), lw=0)
        if np.isfinite(rv[mname]):
            a.axhline(rv[mname], c="k", lw=1.6, label="Real (A2)")
        a.set_xticks(range(4), D.FAMILIES, fontsize=8); a.set_title(mname, fontsize=8)
    for a in ax.ravel()[len(metrics):]:
        a.axis("off")
    ax[0, 0].legend(fontsize=7)
    fig.suptitle("Synthetic (A2; coloured = detector PASS, grey = not PASS) vs Real after A2 correction (black line). "
                 "Matching distributions do not imply matching accuracy on Real.", fontsize=9)
    fig.tight_layout(); fig.savefig(HERE / "synthetic_real_domain_gap.png", dpi=105); plt.close(fig)


for fn in (fig_examples, fig_success, fig_overlap, fig_shared, fig_confusion, fig_false_accepts, fig_graph, fig_gap):
    fn()

# ----------------------------------------------------------------------------- summary json
a2m, a0m, a3m = (main[main.method_key == k] for k in ("A2", "A0", "A3"))
a2c = ctl[ctl.method_key == "A2"]


def rate(k, n):
    p, lo, hi = wilson(k, n)
    return {"k": int(k), "n": int(n), "rate": p, "ci95": [lo, hi]}


summary = {
    "plan": {"main_cases": len(D.main_cases()), "controls": len(D.CONTROLS), "downstream_jobs": len(D.downstream_jobs()),
             "main_results_found": int(len(a2m)), "control_results_found": int(len(a2c)),
             "downstream_results_found": int(ds[["case", "affine_key"]].drop_duplicates().shape[0]) if not ds.empty else 0},
    "affine_success": {"A2": rate(a2m.affine_success.sum(), len(a2m)), "A0": rate(a0m.affine_success.sum(), len(a0m)),
                       "A3_rep0": rate(a3m.affine_success.sum(), len(a3m)),
                       "A2_rep0_same_cases_as_A3": rate(a2m[a2m.rep == 0].affine_success.sum(), int((a2m.rep == 0).sum())),
                       "A2_in_search_range_conditions": rate(a2m[a2m.condition != "C05_oor_off320"].affine_success.sum(), int((a2m.condition != "C05_oor_off320").sum()))},
    "a2_by_family": {f: rate(a2m[a2m.family == f].affine_success.sum(), int((a2m.family == f).sum())) for f in D.FAMILIES},
    "a2_failure_types": a2m.failure_type.value_counts().to_dict(),
    "a2_success_rms_common_median_um": float(a2m[a2m.affine_success.astype(bool)].rms_common_um.median()),
    "detector_A2_main": conf[(conf.method_key == "A2") & (conf.stratum == "all_main")].iloc[0].to_dict() if len(conf) else {},
    "detector_all_methods_main_plus_truth_controls": conf[conf.stratum == "main+truth_controls"].to_dict("records"),
    "controls_A2": a2c.groupby("control").agg(n=("case", "size"), pass_n=("detector_pass", "sum"), false_accept=("false_accept", "sum"),
                                              affine_success=("affine_success", "sum")).reset_index().to_dict("records"),
    "controls_all_methods_false_accept": ctl.groupby("method_key").false_accept.agg(["sum", "size"]).reset_index().to_dict("records"),
    "headline_by_generator": headline.to_dict("records"),
    "real_a2": None if real is None else {"detector_status": real["detector_status"], "objective": real["objective"],
                                          "vs_prior_a2_rms_um": real["vs_prior_a2_rms_um"], "heldout_checker_mean": real["heldout_checker_mean"],
                                          "inlier3": real["inlier3"], "inlier_over_null": real["inlier_over_null"],
                                          "block_consistency": real["block_consistency"], "runtime_s": real["runtime_s"]},
}
if not ds.empty:
    b2 = ds[ds.pipeline_method == B2]
    b2c = b2[b2.kind == "control"]
    summary["downstream_no_shared_controls_A2_B2"] = b2c[["case", "detector_status", "matched_n", "wrong_matched_n", "coverage", "unmatched_f1"]].to_dict("records")
    b2 = b2[b2.kind == "main"]
    summary["downstream_states_B2"] = {k: b2[b2.affine_key == k].state.value_counts().to_dict() for k in ("A0", "A2", "TRUE")}
    summary["downstream_B2_medians"] = b2.groupby("affine_key")[["true_candidate_recall", "top1_accuracy", "matched_precision", "matched_recall",
                                                                  "matched_f1", "unmatched_f1", "coverage", "false_match_rate"]].median().to_dict("index")
    op = b2[b2.affine_key.isin(["A0", "A2"])].groupby("affine_key").agg(stopped=("operational_stopped", "sum"), n=("case", "size"),
                                                                         adopted_matches=("operational_matched_n", "sum"),
                                                                         adopted_wrong=("operational_wrong_matched_n", "sum"))
    op["adopted_precision"] = 1 - op.adopted_wrong / op.adopted_matches.replace(0, np.nan)
    summary["operational_B2"] = op.reset_index().to_dict("records")
L_json = HERE / "diagnostic_summary.json"


def js(v):
    if isinstance(v, dict):
        return {str(k): js(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [js(x) for x in v]
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else float(v)
    return v


L_json.write_text(json.dumps(js(summary), indent=2, ensure_ascii=False) + "\n")
print(headline.to_string())
print(json.dumps(js(summary["affine_success"]), indent=1))
