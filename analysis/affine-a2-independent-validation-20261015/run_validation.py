"""Runner: python -B run_validation.py {main|control|real|downstream} --index i   (writes ./cache/*.json only).

Frozen components are imported, never modified: A0/A2/A3, detector and transform errors from the 2026-10-14
`init_lib`; Graph/UOT/B2 from the 2026-10-14 `experiment_lib`. Truth (persistent ids, true affine) is used only after
the method has returned.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import design as D
import indep_gen as G

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
import sys  # noqa: E402

sys.path.insert(0, str(HERE.parent / "affine-global-initialization-20261014"))
import init_lib as L  # noqa: E402  (frozen)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def truth_objective(pair):
    """Trimmed affine-ICP objective reached when started from the true affine (wrong-basin diagnosis)."""
    C = np.diag([1.0, -1.0]) if np.linalg.det(pair.M_true) < 0 else np.eye(2)
    _, _, err, _ = L.affine_icp(pair.moving_px @ C.T, pair.fixed, pair.M_true @ C, pair.t_true, max_iter=80, trim_quantile=0.7)
    return float(err)


def evaluate(key, res, pair):
    bm = res.diagnostics.get("basin_margin", np.nan)
    rec = {"method_key": key, "method": res.method, "M": res.M.tolist(), "t": res.t.tolist(), "objective": res.objective,
           "runtime_s": res.runtime_s, "peak_mem_mb": res.peak_mem_mb, "n_hypotheses_scored": res.n_hypotheses_scored,
           "n_refined": res.n_refined, "basins": res.refined,
           "diagnostics": {k: v for k, v in res.diagnostics.items()}, "basin_margin_is_inf": bool(np.isinf(bm))}
    # ---- frozen detector, exactly as used for the 2026-10-14 synthetic evaluation (no truth)
    rec.update(L.detector(res.M, res.t, pair.moving_px, pair.fixed, basin_margin=bm))
    # ---- truth-based evaluation
    rec.update(L.transform_errors(res.M, res.t, pair.M_true, pair.t_true, pair.width, pair.height))   # full moving field (prior definition)
    rec.update(G.affine_errors_common(res.M, res.t, pair))                                            # common physical region (primary)
    w = L.apply_eff(res.M, res.t, pair.moving_px)
    sh = pair.true_fixed_index >= 0
    if sh.any():
        d = np.linalg.norm(w[sh] - pair.fixed[pair.true_fixed_index[sh]], axis=1)
        rec["true_pair_distance_median_um"] = float(np.median(d))
        rec["true_pairs_within3"] = float(np.mean(d <= 3))
    rms = rec["rms_common_um"] if np.isfinite(rec["rms_common_um"]) else rec["transform_rms_error_um"]
    rec["affine_success"] = bool(rms <= D.AFFINE_SUCCESS_RMS_UM and rec["chirality_correct"])
    # was a correct basin among the refined, distinct hypotheses? (initialisation vs selection failure)
    rec["refined_basin_rms_common_um"] = [G.affine_errors_common(np.asarray(b[1]), np.asarray(b[2]), pair)["rms_common_um"] for b in res.refined]
    return rec


def run_pair(pair, keys, restrict=None):
    out = {"name": pair.name, "family": pair.family, "params": pair.params, "M_true": pair.M_true.tolist(),
           "t_true": pair.t_true.tolist(), "truth_basin_objective": truth_objective(pair), "methods": []}
    for k in keys:
        log(f"{pair.name} {k}")
        res = L.run_method(k, pair.moving_px, pair.fixed, pair.width, pair.height, restrict_chirality=restrict)
        rec = evaluate(k, res, pair)
        out["methods"].append(rec)
        log(f"  obj={res.objective:.3f} t={res.runtime_s:.1f}s rms_common={rec['rms_common_um']:.2f} "
            f"success={rec['affine_success']} {rec['detector_status']}")
    return out


def main_pair(i):
    fam, cond, rep = D.main_cases()[i]
    return G.make_pair(fam, cond, rep, D.CONDITIONS[cond]), (fam, cond, rep)


def control_pair(j):
    name, fam, rep, ov, mode, restrict, expect = D.CONTROLS[j]
    return G.make_pair(fam, name, rep, ov, mode=mode, name=f"{fam}:{name}:{rep}"), restrict, expect


# ----------------------------------------------------------------------------- Real (A2 frozen, checkerboard held-out)
def run_real():
    r = L.load_real()
    fixed, mov, W, H = r["fixed_all"], r["moving_px"], r["width"], r["height"]
    res = L.run_method("A2", mov, fixed, W, H)
    ck = L.checker_heldout("A2", mov, fixed, W, H)
    hold = float(np.mean([s["heldout_inlier3"] for s in ck]))
    bm = res.diagnostics.get("basin_margin", np.nan)
    det = L.detector(res.M, res.t, mov, fixed, basin_margin=bm if bm is not None else np.inf, heldout_inlier=hold)
    prior = next(m for m in json.loads((L.CACHE / "real.json").read_text())["methods"] if m["method_key"] == "A2")
    vs_prior = L.transform_errors(res.M, res.t, np.asarray(prior["M"]), np.asarray(prior["t"]), W, H)
    out = {"M": res.M.tolist(), "t": res.t.tolist(), "objective": res.objective, "runtime_s": res.runtime_s,
           "diagnostics": res.diagnostics, "basins": res.refined, "checker": ck, "heldout_checker_mean": hold,
           "vs_prior_a2_rms_um": vs_prior["transform_rms_error_um"], **det}
    log(f"real A2 obj={res.objective:.3f} vs_prior={vs_prior['transform_rms_error_um']:.2e} {det['detector_status']}")
    L.save_json(CACHE / "real_a2.json", out)


# ----------------------------------------------------------------------------- downstream (frozen Graph/UOT/B2)
_B2 = None


def b2_models():
    global _B2
    if _B2 is None:
        import experiment_lib as lib
        import pandas as pd
        per = pd.read_csv(L.SEP_SRC / "per_nucleus_results.csv", low_memory=False)
        mm, um, th, _ = lib.fit_separated_models(per, source_method=lib.METHOD_B1, feature_names=lib.UOT_FEATURES, output_method=lib.METHOD_B2)
        _B2 = (mm, um, th)
    return _B2


def downstream(fixed_all, moving_px, M, t, width, height, true_fixed_index, tag):
    """Production-equivalent subset rule + frozen Graph/UOT/B2. true_fixed_index: per moving point or None (Real)."""
    import experiment_lib as lib
    mm, um, th = b2_models()
    mv, fv = L.subset_rule(M, t, moving_px, fixed_all, width, height)
    fixed, moving = fixed_all[fv], L.apply_eff(M, t, moving_px[mv])
    fidx, midx = np.flatnonzero(fv), np.flatnonzero(mv)
    if true_fixed_index is not None:
        remap = -np.ones(len(fixed_all), int); remap[fidx] = np.arange(len(fidx))
        tf = true_fixed_index[mv]
        truth = np.where(tf >= 0, remap[np.maximum(tf, 0)], -1)
    else:
        truth = -np.ones(len(moving), int)
    base = {"tag": tag, "n_fixed_subset": int(len(fixed)), "n_moving_subset": int(len(moving)),
            "common_in_subset": int((truth >= 0).sum()),
            "true_partner_outside_subset": int(((true_fixed_index[mv] >= 0) & (truth < 0)).sum()) if true_fixed_index is not None else None}
    if len(fixed) < 40 or len(moving) < 40:
        return [dict(base, pipeline_method=m, skipped="subset too small") for m in (lib.METHOD_B0, lib.METHOD_B1, lib.METHOD_B2, lib.METHOD_B3)], {}
    ctx = lib.build_fixed_context(fixed, moving)
    runs = lib.method_runs(ctx)
    sample = SimpleNamespace(fixed_points=fixed, moving_points=moving, fixed_ids=np.array([f"f{i}" for i in fidx], dtype=object),
                             moving_ids=np.array([f"m{i}" for i in midx], dtype=object), true_fixed_index=truth,
                             common_moving_mask=truth >= 0, seed=0, difficulty=tag)
    scen = {"scenario_id": tag, "split": "ExternalValidation", "strength": tag, "template_group": tag}
    frames = {m: lib.result_frame(sample, scen, ctx, runs[m], method=m) for m in (lib.METHOD_B0, lib.METHOD_B1, lib.METHOD_B3)}
    frames[lib.METHOD_B2] = lib.apply_separated_adoption(frames[lib.METHOD_B1], mm, um, th["match"]["threshold"], th["unmatched"]["threshold"], method=lib.METHOD_B2)
    hit = np.isclose(np.max(np.abs(ctx.regional.best_offsets), axis=1), 18.0)
    rows = []
    for m, fr in frames.items():
        met = lib.method_metrics(fr)
        matched = fr.status == "matched"
        met.update(base, pipeline_method=m, ncc18_boundary_rate=float(hit[ctx.regional.cell_anchor_index].mean()),
                   candidate_distance_median_um=float(fr.spatial_distance_um.median()),
                   duplicate_destinations_matched=int(fr.loc[matched, "top1_fixed_index"].duplicated(keep=False).sum()),
                   solver_converged=bool(fr.solver_converged.iloc[0]))
        rows.append(met)
    # per-nucleus distributions for the domain-gap comparison (quantiles only)
    qs = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    _, knn = lib.point_density_and_knn(fixed)
    dens, _ = lib.point_density_and_knn(fixed)
    b0, b1, b2 = frames[lib.METHOD_B0], frames[lib.METHOD_B1], frames[lib.METHOD_B2]
    dist = {"density_fixed_subset": dens,
            "knn_um_q": np.quantile(knn, qs).tolist(),
            "candidate_distance_um_q": np.quantile(b0.spatial_distance_um, qs).tolist(),
            "graph_support_q": np.quantile(b0.top1_graph_support, qs).tolist(),
            "row_mass_ratio_q": np.quantile(b1.row_mass_ratio, qs).tolist(),
            "b2_confidence_q": np.quantile(b2.calibrated_confidence, qs).tolist(),
            "quantiles": qs}
    return rows, dist


def run_downstream(i):
    kind, idx, key = D.downstream_jobs()[i]
    if kind == "main":
        pair, (fam, cond, rep) = main_pair(idx)
        rec_file = CACHE / f"main_{idx:03d}.json"
    else:
        pair, _, _ = control_pair(idx)
        rec_file = CACHE / f"control_{idx:02d}.json"
    if key == "TRUE":
        M, t = pair.M_true, pair.t_true
    else:
        d = json.loads(rec_file.read_text())
        assert d["params"]["n_fixed"] == len(pair.fixed) and d["params"]["n_moving"] == len(pair.moving_px), "generator not deterministic"
        m = next(x for x in d["methods"] if x["method_key"] == key)
        M, t = np.asarray(m["M"]), np.asarray(m["t"])
    log(f"downstream {kind} {idx} {pair.name} {key}")
    rows, dist = downstream(pair.fixed, pair.moving_px, M, t, pair.width, pair.height, pair.true_fixed_index, f"{pair.name}:{key}")
    L.save_json(CACHE / f"downstream_{i:03d}.json", {"kind": kind, "index": idx, "name": pair.name, "affine_key": key,
                                                     "rows": rows, "dist": dist})


def run_real_downstream():
    r = L.load_real()
    a2 = json.loads((CACHE / "real_a2.json").read_text())
    rows, dist = downstream(r["fixed_all"], r["moving_px"], np.asarray(a2["M"]), np.asarray(a2["t"]), r["width"], r["height"], None, "real:A2")
    L.save_json(CACHE / "downstream_real_a2.json", {"kind": "real", "name": "real", "affine_key": "A2", "rows": rows, "dist": dist})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("--index", type=int, default=0)
    a = ap.parse_args()
    CACHE.mkdir(exist_ok=True)
    if a.mode == "main":
        pair, (fam, cond, rep) = main_pair(a.index)
        out = run_pair(pair, D.methods_for_main(rep))
        out.update(kind="main", index=a.index, condition=cond, rep=rep, factor=D.FACTOR[cond])
        L.save_json(CACHE / f"main_{a.index:03d}.json", out)
    elif a.mode == "control":
        pair, restrict, expect = control_pair(a.index)
        out = run_pair(pair, D.methods_for_control(D.CONTROLS[a.index][0]), restrict=restrict)
        out.update(kind="control", index=a.index, control=D.CONTROLS[a.index][0], expect=expect, restrict_chirality=restrict)
        L.save_json(CACHE / f"control_{a.index:02d}.json", out)
    elif a.mode == "real":
        run_real()
    elif a.mode == "real_downstream":
        run_real_downstream()
    elif a.mode == "downstream":
        run_downstream(a.index)


if __name__ == "__main__":
    main()
