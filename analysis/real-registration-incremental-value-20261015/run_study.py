"""Runner: python -B run_study.py <mode> [--index i]   (writes ./cache only).

modes: real_stages | real_heldout (0-5) | real_metrics | real_downstream (0-5) | real_residual | semisyn (0-4)
       syn_test (0-55) | syn_dev (0-111) | syn_down (0-223)
Frozen components are imported, never modified. Truth is used only after a method has returned.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
IV = ROOT / "analysis/affine-a2-independent-validation-20261015"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(IV))
import stages as S  # noqa: E402
import metrics as MX  # noqa: E402
import nonrigid as NR  # noqa: E402
import design as D  # noqa: E402  (2026-10-15 plan; read-only)
import indep_gen as G  # noqa: E402  (independent generator; read-only)

L = S.L
CACHE = HERE / "cache"
STAGE_KEYS = ("R0", "R1a", "R1b", "R2s", "R2")
DOWN_STAGES = ("R0", "R1a", "R1b", "R2s", "R2", "R0_saved_subset")

# ----------------------------------------------------------------------------- synthetic Test plan (pre-registered)
TEST_CONDITIONS = {
    "IV_T1_affine_only": dict(deform_rms_um=0.0),
    "IV_T2_nonrigid_mild": dict(deform_rms_um=3.0),
    "IV_T3_nonrigid_moderate": dict(deform_rms_um=6.0),
    "IV_T4_nonrigid_strong": dict(deform_rms_um=12.0),
    "IV_T5_fov_contained": dict(offset_um=40.0, mov_px=(1230, 1230)),
    "IV_T6_fov_smallfixed": dict(offset_um=60.0, fixed_fov_um=(520.0, 520.0)),
    "IV_T7_shared_low": dict(q_shared=0.30),
}
TEST_REPS = (0, 1)


def test_cases():
    return [(f, c, r) for f in D.FAMILIES for c in TEST_CONDITIONS for r in TEST_REPS]


def syn_down_jobs():
    return [(i, k) for i in range(len(test_cases())) for k in ("R1a", "R1b", "R2", "TRUE")]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def save(name, obj):
    L.save_json(CACHE / name, obj)


def load(name):
    return json.loads((CACHE / name).read_text())


# ----------------------------------------------------------------------------- Real helpers
def real_stage_affines():
    st = load("real_stages.json")
    out = {k: (np.asarray(v["M"]), np.asarray(v["t"])) for k, v in st["stages"].items()}
    r = L.load_real()
    out["R0"] = L.saved_effective(r)
    return out


def common_sets(r, aff):
    fixed, mov = r["fixed_all"], r["moving_px"]
    mv = np.ones(len(mov), bool); fv = np.ones(len(fixed), bool)
    for k in STAGE_KEYS:
        M, t = aff[k]
        mv &= MX.eroded_hull_contains(fixed, MX.apply(M, t, mov), 15.0)
        fv &= MX.in_footprint(M, t, r["width"], r["height"], fixed, 10.0)
    return mv, fv


# ----------------------------------------------------------------------------- Real: stages and held-out
def run_real_stages():
    r = L.load_real()
    out = S.run_a2_stages(r["moving_px"], r["fixed_all"])
    prior = json.loads((IV / "cache/real_a2.json").read_text())
    out["R2_max_abs_diff_vs_prior_A2"] = float(max(np.abs(np.asarray(out["stages"]["R2"]["M"]) - prior["M"]).max(),
                                                  np.abs(np.asarray(out["stages"]["R2"]["t"]) - prior["t"]).max()))
    log(f"real stages: R1a==R1b {out['r1a_equals_r1b']} R2 diff {out['R2_max_abs_diff_vs_prior_A2']:.2e}")
    save("real_stages.json", out)


def heldout_folds(mov, W, H):
    colour = (np.floor(mov[:, 0] / 100.0) + np.floor(mov[:, 1] / 100.0)) % 2 == 0
    return [("checker_0", colour), ("checker_1", ~colour), ("x_low", mov[:, 0] < W / 2), ("x_high", mov[:, 0] >= W / 2),
            ("y_low", mov[:, 1] < H / 2), ("y_high", mov[:, 1] >= H / 2)]


def run_real_heldout(k):
    r = L.load_real()
    fixed, mov = r["fixed_all"], r["moving_px"]
    name, tr = heldout_folds(mov, r["width"], r["height"])[k]
    st = S.run_a2_stages(mov[tr], fixed)
    rows = []
    for key, v in st["stages"].items():
        M, t = np.asarray(v["M"]), np.asarray(v["t"])
        te = MX.apply(M, t, mov[~tr]); te = te[L.overlap_moving(te, fixed)]
        d = cKDTree(fixed).query(te)[0]
        row = {"fold": name, "stage": key, "n_heldout": len(te), "heldout_nn_median_um": float(np.median(d))}
        for th in MX.THRESH:
            row[f"heldout_within_{th:g}um"] = float(np.mean(d <= th))
        row.update(MX.null_within3(te, fixed))
        row.update({f"param_{a}": b for a, b in MX.decompose(M, t, r["width"], r["height"]).items()})
        rows.append(row)
    save(f"real_heldout_{k}.json", {"fold": name, "rows": rows, "r1a_equals_r1b": st["r1a_equals_r1b"],
                                    "selected_ncc_rank": st["selected_ncc_rank"], "stages": st["stages"]})
    log(f"heldout {name}: " + ", ".join(f"{x['stage']} {x['heldout_within_3um']:.3f}" for x in rows))


# ----------------------------------------------------------------------------- Real: metrics
def run_real_metrics():
    r = L.load_real()
    fixed, mov, W, H = r["fixed_all"], r["moving_px"], r["width"], r["height"]
    aff = real_stage_affines()
    mvC, fvC = common_sets(r, aff)
    he = MX.HE(r)
    stage_rows, block_rows, scans = [], [], {}
    hull_mask = lambda g: MX.eroded_hull_contains(fixed, g, 15.0)  # noqa: E731  (same mask for every stage)
    for k in STAGE_KEYS:
        M, t = aff[k]
        w = MX.apply(M, t, mov)
        mv_roi, fv_roi = L.subset_rule(M, t, mov, fixed, W, H)
        row = {"stage": k}
        row.update(MX.nn_block(w, fixed, "all_"))
        row.update(MX.nn_block(w[mvC], fixed[fvC], "common_"))
        row.update(MX.nn_block(w[mv_roi], fixed[fv_roi], "roi_"))
        row.update({"common_" + a: b for a, b in MX.null_within3(w[mvC], fixed[fvC]).items()})
        row.update({"common_" + a: b for a, b in MX.density_correlation(w[mvC], fixed[fvC], hull_mask).items()})
        for blk in (200.0, 100.0):
            rows = MX.block_correlogram(w[mvC], fixed, block=blk, search=72.0, kernel=3.0, step=1.0, min_n=25 if blk == 200 else 12)
            row.update(MX.correlogram_summary(rows, prefix=f"corr{blk:g}_"))
            for x in rows:
                block_rows.append({"stage": k, **x})
        row.update(MX.regional_ncc_summary(fixed[fv_roi], w[mv_roi], 18.0))
        row.update(he.metrics(M, t, fixed))
        sc, grids = he.translation_scan(M, t, fixed)
        row.update({"hescan_" + a: b for a, b in sc.items()})
        scans[k] = grids
        row.update({"param_" + a: b for a, b in MX.decompose(M, t, W, H).items()})
        row["roi_n_moving"], row["roi_n_fixed"] = int(mv_roi.sum()), int(fv_roi.sum())
        # image footprint vs point ROI
        poly = MX.he_footprint_world(M, t, W, H)
        row["fixed_in_he_footprint_fraction"] = float(MX.in_footprint(M, t, W, H, fixed, 0.0).mean())
        row["moving_in_fixed_hull_fraction"] = float(MX.eroded_hull_contains(fixed, w, 0.0).mean())
        row["he_footprint_center_minus_fixed_centroid_um"] = float(np.linalg.norm(poly.mean(0) - fixed.mean(0)))
        stage_rows.append(row)
        log(f"metrics {k}: common within3 {row['common_m2f_within_3um']:.3f} corr200 {row['corr200_local_residual_median_um']:.2f}")
    # quadrant-wise HE scans under each A2 stage (independent coarse residual field check)
    quad = []
    for k in ("R1a", "R2"):
        M, t = aff[k]
        for qx in range(3):
            for qy in range(3):
                reg = (fixed[:, 0] >= qx * 340) & (fixed[:, 0] < (qx + 1) * 340) & (fixed[:, 1] >= qy * 340) & (fixed[:, 1] < (qy + 1) * 340)
                if reg.sum() < 80:
                    continue
                sc, _ = he.translation_scan(M, t, fixed, half=12.0, step=1.0, region=reg)
                quad.append({"stage": k, "qx": qx, "qy": qy, **sc})
    displacements = []
    gx, gy = np.meshgrid(np.linspace(0, W, 41), np.linspace(0, H, 41))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    for a, b in (("R1a", "R2"), ("R1b", "R2"), ("R1b", "R2s"), ("R2s", "R2"), ("R0", "R2"), ("R0", "R1a")):
        for tag, pts in (("moving_nuclei", mov), ("image_grid", grid)):
            displacements.append({"from": a, "to": b, "points": tag, **MX.displacement_between(*aff[a], *aff[b], pts)})
    np.savez_compressed(CACHE / "real_he_scans.npz", **{f"{k}_{n}": v for k, g in scans.items() for n, v in zip(("offs", "W", "H"), g)})
    save("real_metrics.json", {"stages": stage_rows, "blocks": block_rows, "he_quadrants": quad, "displacements": displacements,
                               "common_n_moving": int(mvC.sum()), "common_n_fixed": int(fvC.sum())})


# ----------------------------------------------------------------------------- downstream (frozen Graph/UOT/B2)
_B2 = None


def b2_models():
    global _B2
    if _B2 is None:
        import experiment_lib as lib
        per = pd.read_csv(L.SEP_SRC / "per_nucleus_results.csv", low_memory=False)
        mm, um, th, _ = lib.fit_separated_models(per, source_method=lib.METHOD_B1, feature_names=lib.UOT_FEATURES, output_method=lib.METHOD_B2)
        _B2 = (mm, um, th)
    return _B2


def downstream_frames(fixed, moving, fidx, midx, truth, tag):
    import experiment_lib as lib
    mm, um, th = b2_models()
    ctx = lib.build_fixed_context(fixed, moving)
    runs = lib.method_runs(ctx)
    sample = SimpleNamespace(fixed_points=fixed, moving_points=moving, fixed_ids=np.array([f"f{i}" for i in fidx], dtype=object),
                             moving_ids=np.array([f"m{i}" for i in midx], dtype=object), true_fixed_index=truth,
                             common_moving_mask=truth >= 0, seed=0, difficulty=tag)
    scen = {"scenario_id": tag, "split": "IncrementalValue", "strength": tag, "template_group": tag}
    frames = {m: lib.result_frame(sample, scen, ctx, runs[m], method=m) for m in (lib.METHOD_B0, lib.METHOD_B1, lib.METHOD_B3)}
    frames[lib.METHOD_B2] = lib.apply_separated_adoption(frames[lib.METHOD_B1], mm, um, th["match"]["threshold"], th["unmatched"]["threshold"], method=lib.METHOD_B2)
    rows = []
    for m, fr in frames.items():
        met = lib.method_metrics(fr)
        matched = fr.status == "matched"
        met.update(pipeline_method=m, tag=tag, n_fixed_subset=len(fixed), n_moving_subset=len(moving),
                   candidate_distance_median_um=float(fr.spatial_distance_um.median()),
                   candidate_distance_p90_um=float(fr.spatial_distance_um.quantile(0.9)),
                   duplicate_destinations_matched=int(fr.loc[matched, "top1_fixed_index"].duplicated(keep=False).sum()),
                   solver_converged=bool(fr.solver_converged.iloc[0]))
        rows.append(met)
    return rows, frames


def subset_and_truth(M, t, moving_px, fixed_all, W, H, true_fixed_index=None):
    mv, fv = L.subset_rule(M, t, moving_px, fixed_all, W, H)
    fixed, moving = fixed_all[fv], L.apply_eff(M, t, moving_px[mv])
    fidx, midx = np.flatnonzero(fv), np.flatnonzero(mv)
    if true_fixed_index is None:
        truth = -np.ones(len(moving), int)
    else:
        remap = -np.ones(len(fixed_all), int); remap[fidx] = np.arange(len(fidx))
        tf = true_fixed_index[mv]
        truth = np.where(tf >= 0, remap[np.maximum(tf, 0)], -1)
    return fixed, moving, fidx, midx, truth


def run_real_downstream(i):
    import experiment_lib as lib
    key = DOWN_STAGES[i]
    r = L.load_real()
    if key == "R0_saved_subset":
        fixed, moving = r["fixed_subset"], r["moving_subset"]
        fidx = cKDTree(r["fixed_all"]).query(fixed)[1]; midx = cKDTree(r["moving_saved_all"]).query(moving)[1]
        truth = -np.ones(len(moving), int)
    else:
        M, t = real_stage_affines()[key]
        fixed, moving, fidx, midx, truth = subset_and_truth(M, t, r["moving_px"], r["fixed_all"], r["width"], r["height"])
    rows, frames = downstream_frames(fixed, moving, fidx, midx, truth, f"real:{key}")
    keep = ["moving_index", "top1_fixed_index", "status", "calibrated_confidence", "top1_graph_support", "row_mass_ratio",
            "spatial_distance_um", "absolute_min_cost", "reciprocal"]
    per = []
    for m in (lib.METHOD_B0, lib.METHOD_B2):
        f = frames[m][keep].copy()
        f["pipeline_method"] = m; f["stage"] = key
        f["moving_orig_index"] = midx[f.moving_index.to_numpy(int)]
        f["fixed_orig_index"] = fidx[f.top1_fixed_index.to_numpy(int)]
        f["x"], f["y"] = moving[f.moving_index.to_numpy(int)].T
        per.append(f)
    pd.concat(per).to_csv(CACHE / f"real_down_pernuc_{key}.csv.gz", index=False)
    save(f"real_down_{key}.json", {"stage": key, "rows": rows})
    b2 = next(x for x in rows if x["pipeline_method"] == lib.METHOD_B2)
    log(f"real downstream {key}: matched {b2['matched_n']} amb {b2['ambiguous_n']} unm {b2['predicted_unmatched_n']} gs {b2['graph_support_median']:.3f}")


# ----------------------------------------------------------------------------- Real: residual structure
def run_real_residual():
    r = L.load_real()
    fixed, mov, W, H = r["fixed_all"], r["moving_px"], r["width"], r["height"]
    aff = real_stage_affines()
    mvC, fvC = common_sets(r, aff)
    rng = np.random.default_rng(20261015)
    origin = np.array([0.0, 0.0])
    out = {"sources": {}, "relations": {}}
    for k in ("R1a", "R2s", "R2"):
        w = MX.apply(*aff[k], mov)[mvC]
        i, j, u = NR.proxy_pairs(w, fixed[fvC], r=4.0)
        d = NR.full_diagnostic(w[i], u, origin, rng=rng, n_perm=200)
        out["sources"][f"P_{k}"] = d
        log(f"residual P {k}: n {len(u)} rho0-20 {d['rho_0_20']:.3f} skill100 {d['heldout_skill_b100_min']:.3f} perm95 {d['heldout'][0].get('skill_perm_p95')}")
        if k == "R2":
            np.savez_compressed(CACHE / "real_residual_vectors_R2.npz", pos=w[i], vec=u, moving_index=np.flatnonzero(mvC)[i],
                                fixed_index=np.flatnonzero(fvC)[j])
    # N2: unrelated pattern at the R2 pose (mirror-x of the moving cloud) -> chance NN 'residuals'
    w = MX.apply(*aff["R2"], mov)[mvC]
    for nk, nv in (("N2_mirror_x", MX.ac.null_transforms(w)["mirror_x"]), ("N2_rotate_180", MX.ac.null_transforms(w)["rotate_180"])):
        i, j, u = NR.proxy_pairs(nv, fixed[fvC], r=4.0)
        out["sources"][nk] = NR.full_diagnostic(nv[i], u, origin, rng=rng, n_perm=50) if len(u) > 40 else {"n_vectors": int(len(u))}
    # C: correlogram windows (not NN based) under R2
    for blk in (50.0, 100.0, 200.0):
        rows = MX.block_correlogram(w, fixed, block=blk, search=15.0, kernel=2.0, step=0.5, min_n=8)
        ok = [x for x in rows if x["peak_ratio"] >= 5]
        pos = np.array([[x["center_x"], x["center_y"]] for x in ok]); vec = np.array([[x["refined_dx"], x["refined_dy"]] for x in ok])
        dd = {"n_windows": len(rows), "n_reliable": len(ok), "windows": ok}
        if len(ok) >= 10:
            dd.update(NR.vector_autocorr(pos, vec, bins=((0, blk * 1.5), (blk * 1.5, blk * 3), (blk * 3, 600))))
            dd["mag_median"] = float(np.median(np.linalg.norm(vec - vec.mean(0), axis=1)))
            dd["mean_dx"], dd["mean_dy"] = float(vec[:, 0].mean()), float(vec[:, 1].mean())
        out["sources"][f"C_R2_w{blk:g}"] = dd
    # G: B2 matched pairs under R2 (requires real_downstream R2)
    pn = CACHE / "real_down_pernuc_R2.csv.gz"
    if pn.exists():
        import experiment_lib as lib
        f = pd.read_csv(pn)
        f = f[(f.pipeline_method == lib.METHOD_B2) & (f.status == "matched")]
        wm = MX.apply(*aff["R2"], mov[f.moving_orig_index.to_numpy(int)])
        u = fixed[f.fixed_orig_index.to_numpy(int)] - wm
        out["sources"]["G_R2_B2matched"] = NR.full_diagnostic(wm, u, origin, rng=rng, n_perm=200)
        # relations: residual magnitude vs Graph support / B2 confidence
        allb2 = pd.read_csv(pn); allb2 = allb2[allb2.pipeline_method == lib.METHOD_B2]
        z = np.load(CACHE / "real_residual_vectors_R2.npz")
        mag = pd.Series(np.linalg.norm(z["vec"], axis=1), index=z["moving_index"])
        a = allb2.set_index("moving_orig_index").join(mag.rename("res_mag"), how="inner")
        out["relations"]["graph_support_bins"] = _bin_table(a["top1_graph_support"], a["res_mag"], [0, 0.5, 0.7, 0.8, 0.9, 1.01])
        out["relations"]["b2_confidence_bins"] = _bin_table(a["calibrated_confidence"], a["res_mag"], [0, 0.5, 0.8, 0.9, 0.95, 1.01])
        out["relations"]["spearman_res_vs_graph_support"] = float(a[["res_mag", "top1_graph_support"]].corr("spearman").iloc[0, 1])
    # relations: density, ROI edge distance, HE white boundary distance
    z = np.load(CACHE / "real_residual_vectors_R2.npz")
    pos, mag = z["pos"], np.linalg.norm(z["vec"], axis=1)
    knn = cKDTree(fixed).query(pos, k=7)[0][:, 6]
    out["relations"]["knn6_bins"] = _bin_table(pd.Series(knn), pd.Series(mag), list(np.quantile(knn, [0, 0.25, 0.5, 0.75, 1.0]) + np.array([0, 0, 0, 0, 1e-6])))
    M2, t2 = aff["R2"]
    edge = np.minimum(MX.poly_edge_distance(MX.he_footprint_world(M2, t2, W, H), pos),
                      MX.poly_edge_distance(fixed[MX.ConvexHull(fixed).vertices], pos))
    out["relations"]["roi_edge_bins"] = _bin_table(pd.Series(edge), pd.Series(mag), [0, 25, 50, 100, 200, 2000])
    he = MX.HE(r)
    sd = he._sample(he.sdist, he.pix(M2, t2, fixed[z["fixed_index"]]))
    out["relations"]["he_white_signed_dist_bins"] = _bin_table(pd.Series(sd), pd.Series(mag), [-1e9, 0, 5, 15, 40, 1e9])
    save("real_residual.json", out)


def _bin_table(x, y, edges):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (x >= a) & (x < b)
        rows.append({"lo": float(a), "hi": float(b), "n": int(s.sum()), "res_mag_median": float(np.median(y[s])) if s.any() else np.nan,
                     "res_mag_p90": float(np.quantile(y[s], 0.9)) if s.any() else np.nan, "frac_gt2um": float(np.mean(y[s] > 2)) if s.any() else np.nan})
    return rows


# ----------------------------------------------------------------------------- N3 semi-synthetic calibration
SEMI_AMPS = (0.0, 0.5, 1.0, 2.0, 4.0)
SEMI_REPS = 20


def run_semisyn(k):
    amp = SEMI_AMPS[k]
    r = L.load_real()
    fixed, mov = r["fixed_all"], r["moving_px"]
    aff = real_stage_affines()
    mvC, fvC = common_sets(r, aff)
    w = MX.apply(*aff["R2"], mov)[mvC]; fx = fixed[fvC]
    i, j, u = NR.proxy_pairs(w, fx, r=4.0)
    q_pair = len(u) / len(w)
    sig = float(np.median([1.4826 * np.median(np.abs(u[:, a] - np.median(u[:, a]))) for a in (0, 1)]))
    hull = fx[MX.ConvexHull(fx).vertices]
    res = []
    for rep in range(SEMI_REPS):
        rng = G.rng_for("semisyn", f"{amp:g}", rep)
        box = (fx[:, 0].min() - 120, fx[:, 1].min() - 120, fx[:, 0].max() + 120, fx[:, 1].max() + 120)
        fld = G.nonaffine_field(rng, box, hull, amp) if amp > 0 else None
        mvs, dtrue = NR.semisynthetic(fx, hull, q_pair=q_pair, sigma_pair=sig, n_moving=len(w), amp_um=amp, rng=rng, field_fn=fld)
        ii, jj, uu = NR.proxy_pairs(mvs, fx, r=4.0)
        d = NR.full_diagnostic(mvs[ii], uu, np.zeros(2), rng=rng, n_perm=0)
        d.update(amp=amp, rep=rep, q_pair=q_pair, sigma_pair=sig, n_pairs=len(uu))
        tgt = -dtrue[ii]
        d["truth_corr_proxy_vs_field"] = float(np.corrcoef(np.r_[uu[:, 0], uu[:, 1]], np.r_[tgt[:, 0], tgt[:, 1]])[0, 1]) if amp > 0 else np.nan
        d.pop("heldout")
        res.append(d)
        log(f"semisyn amp {amp} rep {rep}: rho0-20 {d['rho_0_20']:.3f} skill100min {d['heldout_skill_b100_min']:.3f}")
    save(f"semisyn_{k}.json", res)


# ----------------------------------------------------------------------------- synthetic
def eval_stage_truth(M, t, pair):
    e = G.affine_errors_common(M, t, pair)
    e.update(L.transform_errors(M, t, pair.M_true, pair.t_true, pair.width, pair.height))
    w = L.apply_eff(M, t, pair.moving_px)
    sh = pair.true_fixed_index >= 0
    d = np.linalg.norm(w[sh] - pair.fixed[pair.true_fixed_index[sh]], axis=1)
    e["true_pair_residual_median_um"] = float(np.median(d)); e["true_pairs_within3"] = float(np.mean(d <= 3))
    e["affine_success"] = bool(e["rms_common_um"] <= D.AFFINE_SUCCESS_RMS_UM and e["chirality_correct"]) if np.isfinite(e["rms_common_um"]) else False
    nn = cKDTree(pair.fixed).query(w)[0]
    e["m2f_within3_all"] = float(np.mean(nn <= 3))
    e["icp_trimmed_objective_q07"] = float(np.mean(nn[nn <= np.quantile(nn, 0.7)]))
    return e


def true_field(pair, fam, cond, rep, overrides):
    p = dict(G.DEFAULTS); p.update(overrides or {})
    allp = np.concatenate([pair.fixed_fov, pair.moving_fov_world])
    box = (allp[:, 0].min() - 120, allp[:, 1].min() - 120, allp[:, 0].max() + 120, allp[:, 1].max() + 120)
    return G.nonaffine_field(G.rng_for("deform", fam, cond, rep), box, pair.moving_fov_world, p["deform_rms_um"])


def run_syn(kind, i):
    if kind == "test":
        fam, cond, rep = test_cases()[i]
        ov = TEST_CONDITIONS[cond]
    else:
        fam, cond, rep = D.main_cases()[i]
        ov = D.CONDITIONS[cond]
    pair = G.make_pair(fam, cond, rep, ov)
    st = S.run_a2_stages(pair.moving_px, pair.fixed)
    out = {"kind": kind, "index": i, "family": fam, "condition": cond, "rep": rep, "params": pair.params,
           "r1a_equals_r1b": st["r1a_equals_r1b"], "selected_ncc_rank": st["selected_ncc_rank"], "stages": {}}
    if kind == "dev":
        prev = next(m for m in json.loads((IV / f"cache/main_{i:03d}.json").read_text())["methods"] if m["method_key"] == "A2")
        out["R2_max_abs_diff_vs_prior"] = float(max(np.abs(np.asarray(st["stages"]["R2"]["M"]) - prev["M"]).max(),
                                                   np.abs(np.asarray(st["stages"]["R2"]["t"]) - prev["t"]).max()))
    for k, v in list(st["stages"].items()) + [("R3", {"M": pair.M_true.tolist(), "t": pair.t_true.tolist()})]:
        M, t = np.asarray(v["M"]), np.asarray(v["t"])
        rec = {"M": v["M"], "t": v["t"], **eval_stage_truth(M, t, pair), **{"param_" + a: b for a, b in MX.decompose(M, t).items()}}
        out["stages"][k] = rec
    # residual-structure diagnostic vs truth (Test, deformation conditions only)
    if kind == "test" and cond in ("IV_T1_affine_only", "IV_T2_nonrigid_mild", "IV_T3_nonrigid_moderate", "IV_T4_nonrigid_strong"):
        fld = true_field(pair, fam, cond, rep, ov)
        rng = np.random.default_rng(i)
        diag = {}
        for k in ("R3", "R2"):
            M, t = np.asarray(out["stages"][k]["M"]), np.asarray(out["stages"][k]["t"])
            w = L.apply_eff(M, t, pair.moving_px)
            inside = G.in_polygon(w, pair.fixed_fov)
            w = w[inside]
            ii, jj, uu = NR.proxy_pairs(w, pair.fixed, r=4.0)
            dd = NR.full_diagnostic(w[ii], uu, np.zeros(2), rng=rng, n_perm=50)
            if fld is not None:
                tgt = -fld(pair.moving_world_true[inside][ii])
                dd["truth_corr_proxy_vs_field"] = float(np.corrcoef(np.r_[uu[:, 0], uu[:, 1]], np.r_[tgt[:, 0], tgt[:, 1]])[0, 1])
                dd["truth_field_rms_at_pairs"] = float(np.sqrt(np.mean(np.sum(tgt**2, 1))))
            tfi = pair.true_fixed_index[inside][ii]
            dd["proxy_pair_true_id_rate"] = float(np.mean(tfi == jj))
            diag[k] = dd
        out["residual_diagnostic"] = diag
    save(f"syn_{kind}_{i:03d}.json", out)
    log(f"syn {kind} {i} {pair.name}: " + ", ".join(f"{k} {v['rms_common_um']:.2f}" for k, v in out["stages"].items()))


def run_syn_down(j):
    i, key = syn_down_jobs()[j]
    fam, cond, rep = test_cases()[i]
    rec = load(f"syn_test_{i:03d}.json")
    pair = G.make_pair(fam, cond, rep, TEST_CONDITIONS[cond])
    if key == "TRUE":
        M, t = pair.M_true, pair.t_true
    else:
        M, t = np.asarray(rec["stages"][key]["M"]), np.asarray(rec["stages"][key]["t"])
    if key == "R1b" and rec["r1a_equals_r1b"]:
        save(f"syn_down_{j:03d}.json", {"index": i, "key": key, "same_as": "R1a"})
        return
    fixed, moving, fidx, midx, truth = subset_and_truth(M, t, pair.moving_px, pair.fixed, pair.width, pair.height, pair.true_fixed_index)
    if len(fixed) < 40 or len(moving) < 40:
        save(f"syn_down_{j:03d}.json", {"index": i, "key": key, "skipped": "subset too small"})
        return
    rows, frames = downstream_frames(fixed, moving, fidx, midx, truth, f"{pair.name}:{key}")
    save(f"syn_down_{j:03d}.json", {"index": i, "key": key, "family": fam, "condition": cond, "rep": rep, "rows": rows})
    log(f"syn down {pair.name} {key} done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("--index", type=int, default=0)
    a = ap.parse_args()
    CACHE.mkdir(exist_ok=True)
    {"real_stages": lambda: run_real_stages(), "real_heldout": lambda: run_real_heldout(a.index),
     "real_metrics": lambda: run_real_metrics(), "real_downstream": lambda: run_real_downstream(a.index),
     "real_residual": lambda: run_real_residual(), "semisyn": lambda: run_semisyn(a.index),
     "syn_test": lambda: run_syn("test", a.index), "syn_dev": lambda: run_syn("dev", a.index),
     "syn_down": lambda: run_syn_down(a.index)}[a.mode]()


if __name__ == "__main__":
    main()
