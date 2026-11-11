"""Runner: python -B run_cases.py {synthetic|control|real|downstream} --index i  (writes cache/*.json only)."""
from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

import init_lib as L

L.CACHE.mkdir(parents=True, exist_ok=True)
KEYS = ("A0", "A1", "A2", "A3")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def method_record(key, res, pair=None):
    rec = {"method_key": key, "method": res.method, "M": res.M.tolist(), "t": res.t.tolist(), "objective": res.objective,
           "runtime_s": res.runtime_s, "peak_mem_mb": res.peak_mem_mb, "n_hypotheses_scored": res.n_hypotheses_scored,
           "n_refined": res.n_refined, "basins": res.refined, "diagnostics": res.diagnostics}
    det = L.detector(res.M, res.t, pair.moving_px, pair.fixed, basin_margin=res.diagnostics.get("basin_margin", np.nan))
    rec.update(det)
    if pair is not None and pair.M_true is not None:
        rec.update(L.transform_errors(res.M, res.t, pair.M_true, pair.t_true, pair.width, pair.height))
        w = L.apply_eff(res.M, res.t, pair.moving_px)
        rec["nuclei_coord_error_median_um"] = float(np.median(np.linalg.norm(w - pair.moving_world_true, axis=1)))
        sh = pair.true_fixed_index >= 0
        rec["true_pair_distance_median_um"] = float(np.median(np.linalg.norm(w[sh] - pair.fixed[pair.true_fixed_index[sh]], axis=1))) if sh.any() else np.nan
        rec["true_pairs_within3"] = float(np.mean(np.linalg.norm(w[sh] - pair.fixed[pair.true_fixed_index[sh]], axis=1) <= 3)) if sh.any() else np.nan
        e = rec["transform_rms_error_um"]
        rec["outcome"] = "success" if e <= 5 and rec["chirality_correct"] else ("imprecise" if e <= 20 and rec["chirality_correct"] else "wrong_basin")
    # truth-optimal objective (wrong-basin diagnosis: did ICP prefer a worse-than-truth local minimum?)
    return rec


def truth_objective(pair):
    if pair.M_true is None:
        return np.nan
    C = np.diag([1.0, -1.0]) if np.linalg.det(pair.M_true) < 0 else np.eye(2)
    M, t, err, _ = L.affine_icp(pair.moving_px @ C.T, pair.fixed, pair.M_true @ C, pair.t_true, max_iter=80, trim_quantile=0.7)
    return err


def run_pair(pair, tag, restrict=None, keys=KEYS):
    out = {"case": tag, "params": pair.params, "truth_basin_objective": truth_objective(pair), "methods": []}
    for k in keys:
        log(f"{tag} {k}")
        res = L.run_method(k, pair.moving_px, pair.fixed, pair.width, pair.height, restrict_chirality=restrict)
        out["methods"].append(method_record(k, res, pair))
        log(f"  obj={res.objective:.3f} t={res.runtime_s:.1f}s {out['methods'][-1].get('outcome', '')} {out['methods'][-1]['detector_status']}")
    return out


# ----------------------------------------------------------------------------- controls
def real_pair(restrict_note=""):
    r = L.load_real()
    return L.Pair("real", r["fixed_all"], r["moving_px"], r["width"], r["height"], None, None,
                  -np.ones(len(r["moving_px"]), int), None, {"note": restrict_note})


def density_only_pair(seed=301):
    """Independent points drawn from the *Real fixed* density (no shared nuclei), known affine to pixel frame."""
    rng = np.random.default_rng(seed)
    r = L.load_real()
    f = r["fixed_all"]
    lo, hi = f.min(0), f.max(0)
    ax = np.linspace(lo[0], hi[0], 200); ay = np.linspace(lo[1], hi[1], 200)
    H, _, _ = np.histogram2d(f[:, 0], f[:, 1], bins=[np.r_[ax, hi[0] + 1e-6], np.r_[ay, hi[1] + 1e-6]])
    from scipy.ndimage import gaussian_filter
    lam = gaussian_filter(H, 12 / (ax[1] - ax[0]))
    lam = lam / lam.sum() * len(f) * 0.95
    p = []
    cells = rng.poisson(lam)
    for (i, j), c in np.ndenumerate(cells):
        if c:
            p.append(np.column_stack([rng.uniform(ax[i], ax[i] + (ax[1] - ax[0]), c), rng.uniform(ay[j], ay[j] + (ay[1] - ay[0]), c)]))
    p = np.concatenate(p)
    keep = np.ones(len(p), bool)
    for a, b in cKDTree(p).query_pairs(4.5):
        if keep[a] and keep[b]:
            keep[b] = False
    p = p[keep]
    M = 1.1 * L.rot(0.6) @ np.diag([1.0, -1.0])
    c = f.mean(0) + np.array([0.0, 50.0])
    t = c - M @ np.array([450.5, 450.5])
    px = np.linalg.solve(M, (p - t).T).T
    ok = (px[:, 0] >= 0) & (px[:, 0] < 901) & (px[:, 1] >= 0) & (px[:, 1] < 901)
    return L.Pair("density_only", f, px[ok], 901.0, 901.0, M, t, -np.ones(ok.sum(), int), p[ok], {"seed": seed})


def controls():
    r = L.load_real()
    syn = L.make_pair(102, offset_um=50.0)
    rng = np.random.default_rng(401)
    return [
        ("NC1a unrelated: Real fixed vs synthetic moving", L.Pair("nc1a", r["fixed_all"], syn.moving_px, 901.0, 901.0, None, None,
                                                                   -np.ones(len(syn.moving_px), int), None, {}), None),
        ("NC1b unrelated: synthetic independent sections", L.make_pair(103, offset_um=50.0, unrelated_moving=True, name="nc1b"), None),
        ("NC2a orientation mismatch: Real, wrong chirality only", real_pair("proper only"), "proper"),
        ("NC2b orientation mismatch: synthetic, wrong chirality only", L.make_pair(104, offset_um=50.0), "proper"),
        ("NC3 tiny overlap: synthetic offset 900 um", L.make_pair(105, offset_um=900.0), None),
        ("NC4 density-only: resampled Real fixed density, no shared nuclei", density_only_pair(), None),
        ("NC5 no correspondence: uniform random moving", L.Pair("nc5", r["fixed_all"], rng.uniform(0, 901, (1842, 2)), 901.0, 901.0,
                                                               None, None, -np.ones(1842, int), None, {}), None),
    ]


# ----------------------------------------------------------------------------- real
def run_real():
    r = L.load_real()
    pair = real_pair()
    Ms, ts = L.saved_effective(r)
    ref = np.load(L.PRIOR_AUDIT / "cache/corrected_points.npz")
    Mr = ref["matrix"] @ np.diag([-1.0, 1.0]); tr = ref["translation"] + ref["matrix"] @ np.array([r["width"], 0.0])
    out = {"case": "real", "saved_effective": [Ms.tolist(), ts.tolist()], "reference_refit_effective": [Mr.tolist(), tr.tolist()], "methods": []}
    for k in KEYS:
        log(f"real {k}")
        res = L.run_method(k, pair.moving_px, pair.fixed, pair.width, pair.height)
        splits = L.split_heldout(k, pair.moving_px, pair.fixed, pair.width, pair.height, 0) + \
            L.split_heldout(k, pair.moving_px, pair.fixed, pair.width, pair.height, 1)
        hold = float(np.mean([s["heldout_inlier3"] for s in splits]))
        rec = method_record(k, res, pair)
        rec.update(L.detector(res.M, res.t, pair.moving_px, pair.fixed, basin_margin=res.diagnostics.get("basin_margin", np.nan),
                              heldout_inlier=hold))
        rec["heldout_splits"] = splits
        rec["vs_saved"] = L.transform_errors(res.M, res.t, Ms, ts, pair.width, pair.height)
        rec["vs_reference_refit"] = L.transform_errors(res.M, res.t, Mr, tr, pair.width, pair.height)
        wf, warea, nn = L.he_white_fraction(r, res.M, res.t)
        rec["he_white_fixed_fraction"], rec["he_white_area_fraction"] = wf, warea
        out["methods"].append(rec)
        log(f"  obj={res.objective:.3f} heldout={hold:.3f} vs_ref={rec['vs_reference_refit']['transform_rms_error_um']:.2f} {rec['detector_status']}")
    # reference/saved for context
    out["saved_detector"] = L.detector(Ms, ts, pair.moving_px, pair.fixed)
    out["reference_detector"] = L.detector(Mr, tr, pair.moving_px, pair.fixed)
    L.save_json(L.CACHE / "real.json", out)


# ----------------------------------------------------------------------------- downstream
DOWNSTREAM_SYNTH = ("offset000", "offset050", "offset100", "missing40_offset50", "low_shared_offset50", "partial_overlap_350")


def downstream_jobs():
    jobs = [("real", k) for k in KEYS] + [("real", "saved_subset")]
    jobs += [(c, k) for c in DOWNSTREAM_SYNTH for k in KEYS]
    return jobs


def run_downstream(i):
    import experiment_lib as lib
    import pandas as pd
    case, key = downstream_jobs()[i]
    per = pd.read_csv(L.SEP_SRC / "per_nucleus_results.csv", low_memory=False)
    mm, um, th, _ = lib.fit_separated_models(per, source_method=lib.METHOD_B1, feature_names=lib.UOT_FEATURES, output_method=lib.METHOD_B2)
    ref_partner = None
    if case == "real":
        r = L.load_real()
        if key == "saved_subset":
            fixed, moving = r["fixed_subset"], r["moving_subset"]
            fidx = cKDTree(r["fixed_all"]).query(fixed)[1]; midx = cKDTree(r["moving_saved_all"]).query(moving)[1]
            truth = -np.ones(len(moving), int)
        else:
            rec = next(m for m in json.loads((L.CACHE / "real.json").read_text())["methods"] if m["method_key"] == key)
            M, t = np.asarray(rec["M"]), np.asarray(rec["t"])
            mv, fv = L.subset_rule(M, t, r["moving_px"], r["fixed_all"], r["width"], r["height"])
            fixed, moving = r["fixed_all"][fv], L.apply_eff(M, t, r["moving_px"][mv])
            fidx, midx = np.flatnonzero(fv), np.flatnonzero(mv)
            truth = -np.ones(len(moving), int)
        # reference partner (from the prior audit's diagnostic refit), keyed by original ids -- reference only
        ref = np.load(L.PRIOR_AUDIT / "cache/corrected_points.npz")["corrected_all"]
        d, j = cKDTree(r["fixed_all"]).query(ref)
        back = cKDTree(ref).query(r["fixed_all"])[1]
        ref_partner = np.where((d <= 3) & (back[j] == np.arange(len(ref))), j, -1)
    else:
        cond = L.SYNTHETIC_CONDITIONS[case]
        pair = L.make_pair(L.SYNTHETIC_SEEDS[0], **cond)
        rec = next(m for m in json.loads((L.CACHE / f"synthetic_{case}_{L.SYNTHETIC_SEEDS[0]}.json").read_text())["methods"] if m["method_key"] == key)
        M, t = np.asarray(rec["M"]), np.asarray(rec["t"])
        mv, fv = L.subset_rule(M, t, pair.moving_px, pair.fixed, pair.width, pair.height)
        fixed, moving = pair.fixed[fv], L.apply_eff(M, t, pair.moving_px[mv])
        fidx, midx = np.flatnonzero(fv), np.flatnonzero(mv)
        remap = -np.ones(len(pair.fixed), int); remap[fidx] = np.arange(len(fidx))
        tf = pair.true_fixed_index[mv]
        truth = np.where(tf >= 0, remap[np.maximum(tf, 0)], -1)
    log(f"downstream {case} {key}: fixed {len(fixed)} moving {len(moving)}")
    ctx = lib.build_fixed_context(fixed, moving)
    runs = lib.method_runs(ctx)
    sample = SimpleNamespace(fixed_points=fixed, moving_points=moving, fixed_ids=np.array([f"f{i}" for i in fidx], dtype=object),
                             moving_ids=np.array([f"m{i}" for i in midx], dtype=object), true_fixed_index=truth,
                             common_moving_mask=truth >= 0, seed=0, difficulty=case)
    scen = {"scenario_id": f"{case}:{key}", "split": "Diag", "strength": case, "template_group": case}
    frames = {m: lib.result_frame(sample, scen, ctx, runs[m], method=m) for m in (lib.METHOD_B0, lib.METHOD_B1, lib.METHOD_B3)}
    frames[lib.METHOD_B2] = lib.apply_separated_adoption(frames[lib.METHOD_B1], mm, um, th["match"]["threshold"], th["unmatched"]["threshold"], method=lib.METHOD_B2)
    reg = ctx.regional
    hit = np.isclose(np.max(np.abs(reg.best_offsets), axis=1), 18.0)
    rows = []
    for m, fr in frames.items():
        matched = fr.status == "matched"
        dest = fr.loc[matched, "top1_fixed_index"]
        row = {"case": case, "method_key": key, "pipeline_method": m, "n_fixed": len(fixed), "n_moving": len(moving),
               "ncc18_nucleus_boundary_rate": float(hit[reg.cell_anchor_index].mean()),
               "matched": int(matched.sum()), "ambiguous": int((fr.status == "ambiguous").sum()), "unmatched": int((fr.status == "unmatched").sum()),
               "duplicate_destinations": int(dest.duplicated(keep=False).sum()),
               "candidate_distance_median_um": float(fr.spatial_distance_um.median()),
               "graph_support_median": float(fr.top1_graph_support.median()),
               "min_cost_median": float(fr.absolute_min_cost.median()),
               "row_mass_median": float(fr.row_mass_ratio.median()), "transported_total_mass": float(fr.transported_total_mass.iloc[0]),
               "confidence_median": float(fr.calibrated_confidence.median()),
               "solver_converged": bool(fr.solver_converged.iloc[0])}
        if (truth >= 0).any():
            com = fr.is_common_cell.to_numpy(bool)
            row["true_candidate_recall30"] = float(fr.true_in_candidate_set[com].mean())
            row["top1_accuracy_common"] = float(fr.top1_correct[com].mean())
            row["matched_precision"] = float(fr.top1_correct[matched].mean()) if matched.any() else np.nan
            row["matched_recall_common"] = float((fr.top1_correct & matched)[com].mean())
        if ref_partner is not None:
            top1_orig = fidx[fr.top1_fixed_index.to_numpy(int)]
            rp = ref_partner[midx]
            has = rp >= 0
            row["top1_equals_reference_partner"] = float(np.mean(top1_orig[has] == rp[has]))
            row["matched_equals_reference_partner"] = float(np.mean(top1_orig[matched.to_numpy() & has] == rp[matched.to_numpy() & has])) if (matched.to_numpy() & has).any() else np.nan
        rows.append(row)
    L.save_json(L.CACHE / f"downstream_{i:02d}.json", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("--index", type=int, default=0)
    a = ap.parse_args()
    if a.mode == "synthetic":
        cond, seed = L.synthetic_cases()[a.index]
        pair = L.make_pair(seed, name=cond, **L.SYNTHETIC_CONDITIONS[cond])
        out = run_pair(pair, f"{cond}_{seed}")
        out["condition"], out["seed"] = cond, seed
        L.save_json(L.CACHE / f"synthetic_{cond}_{seed}.json", out)
    elif a.mode == "control":
        name, pair, restrict = controls()[a.index]
        out = run_pair(pair, name, restrict=restrict)
        L.save_json(L.CACHE / f"control_{a.index}.json", out)
    elif a.mode == "real":
        run_real()
    elif a.mode == "real_checker":
        pair = real_pair()
        rows = {k: L.checker_heldout(k, pair.moving_px, pair.fixed, pair.width, pair.height) for k in KEYS}
        L.save_json(L.CACHE / "real_checker.json", rows)
    elif a.mode == "downstream":
        run_downstream(a.index)


if __name__ == "__main__":
    main()
