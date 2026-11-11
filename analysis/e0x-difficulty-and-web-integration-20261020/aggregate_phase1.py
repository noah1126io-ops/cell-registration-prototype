"""Aggregate Phase 1 (ablation, candidate diagnostics, synthetic sweep, 031-8_6 failure analysis) -> CSV / JSON / figures.

  python -B aggregate_phase1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e0xlab  # noqa: E402,F401
from oir.frames import json_safe  # noqa: E402
from oir.sample import load_sample  # noqa: E402
import run_real as prev  # noqa: E402
import synth  # noqa: E402

CACHE = HERE / "cache"
FIG = HERE / "figs"
FIG.mkdir(exist_ok=True)


def g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return default
        d = d[k]
    return d


# ----------------------------------------------------------------------------- ablation
def ablation():
    rows, cands = [], []
    recs = {}
    for p in sorted(CACHE.glob("abl_*.json")):
        r = json.loads(p.read_text())
        recs[(r["sample"], r["fixed"], r["variant"])] = r
    for (s, f, v), r in sorted(recs.items()):
        top = r["candidates"][0] if r["candidates"] else {}
        ev = top.get("eval_if_frame") or {}
        x3 = recs.get((s, f, "X3"))
        dpose = None
        if x3 and x3["candidates"] and top:
            ext = np.asarray(r["sample_extents_um"]["IF" if f == "HE" else "HE"])
            probe = np.stack(np.meshgrid(np.linspace(0, ext[0], 5), np.linspace(0, ext[1], 5)), -1).reshape(-1, 2)
            A = np.asarray(top["T_final"]["M"]); a = np.asarray(top["T_final"]["t"])
            B = np.asarray(x3["candidates"][0]["T_final"]["M"]); b = np.asarray(x3["candidates"][0]["T_final"]["t"])
            dpose = float(np.median(np.linalg.norm(probe @ A.T + a - (probe @ B.T + b), axis=1)))
        pc = g(top, "icp", "pose_change") or {}
        rows.append({"sample": s, "fixed": f, "variant": v, "canvas": r["canvas"], "icp_view": r["icp"], "verdict": r["verdict"], "margin": r["margin"],
                     "safety": r["safety"], "accepted": r["accepted"], "theta_deg": top.get("theta_deg"), "scale_search": top.get("scale"),
                     "t_search_um": json.dumps([round(x, 1) for x in top.get("t_um", [])]), "ncc_peak": top.get("ncc"), "z_peak": top.get("score_z"),
                     "nn_median_um": g(ev, "final", "nn_median_um"), "le_3um": g(ev, "final", "le_3um"), "null_le_3um": g(ev, "null", "le_3um"),
                     "inlier_over_null": ev.get("inlier_over_null"), "n_if_eval": ev.get("n_if_eval"),
                     "scale_he_to_if": g(ev, "params_he_to_if", "scale"), "rotation_he_to_if_deg": g(ev, "params_he_to_if", "rotation_deg"),
                     "vs_reference_median_um": g(ev, "vs_reference_um", "median"), "icp_scale_change": pc.get("scale_change"),
                     "icp_rot_change_deg": pc.get("rot_change_deg"), "icp_median_shift_um": pc.get("median_shift_um"),
                     "search_le_3um_fixed_frame": g(top, "search_eval", "le_3um"), "final_le_3um_fixed_frame": g(top, "final_eval", "le_3um"),
                     "overlap_area_mm2": g(top, "overlap", "overlap_area_mm2"), "pose_diff_vs_X3_um": dpose,
                     "runtime_s": r["runtime_s"], "search_runtime_s": r["search_runtime_s"], "peak_mem_mb": r["peak_mem_mb"], "n_poses": r["n_poses"]})
        if v == "X3":
            topc = r["candidates"][0] if r["candidates"] else None
            for c in r["candidates"]:
                e = c.get("eval_if_frame") or {}
                ov = c.get("overlap") or {}
                cen = np.mean(np.asarray(ov.get("overlap_polygon_fixed_um") or [[np.nan, np.nan]]), 0)
                cen0 = np.mean(np.asarray(g(topc, "overlap", "overlap_polygon_fixed_um") or [[np.nan, np.nan]]), 0)
                img = e.get("image") or {}
                cands.append({"sample": s, "fixed": f, "rank": c["rank"], "theta_deg": c["theta_deg"], "scale": c["scale"], "t_um": json.dumps([round(x, 1) for x in c["t_um"]]),
                              "z": c.get("score_z"), "ncc": c.get("ncc"), "relative_gap_to_top": c.get("relative_gap_to_top"),
                              "region_centre_fixed_um": json.dumps([round(float(x), 1) for x in cen]), "dist_to_top_region_um": float(np.linalg.norm(cen - cen0)),
                              "search_le_3um": g(c, "search_eval", "le_3um"), "final_le_3um_fixed_frame": g(c, "final_eval", "le_3um"),
                              "final_null_fixed_frame": g(c, "final_eval", "null_le_3um"), "final_inlier_over_null_fixed_frame": g(c, "final_eval", "inlier_over_null"),
                              "geometric_pass": g(c, "final_eval", "geometric_pass"), "le_3um_if": g(e, "final", "le_3um"), "null_le_3um_if": g(e, "null", "le_3um"),
                              "nn_median_um_if": g(e, "final", "nn_median_um"), "icp_median_shift_um": g(c, "icp", "pose_change", "median_shift_um"),
                              "icp_scale_change": g(c, "icp", "pose_change", "scale_change"), "icp_rot_change_deg": g(c, "icp", "pose_change", "rot_change_deg"),
                              "overlap_area_mm2": ov.get("overlap_area_mm2"), "n_fixed_in_overlap": ov.get("n_fixed_in_overlap"),
                              "n_moving_in_overlap": ov.get("n_moving_in_overlap"), "fixed_density_per_mm2": ov.get("fixed_density_per_mm2"),
                              "moving_density_per_mm2": ov.get("moving_density_per_mm2"), "img_tissue_dice": img.get("tissue_dice"),
                              "img_nuclear_corr": img.get("nuclear_corr"), "vs_reference_median_um": g(e, "vs_reference_um", "median"),
                              "safety": "PREVIEW ONLY (candidate)"})
    pd.DataFrame(rows).to_csv(HERE / "e0x_ablation.csv", index=False)
    pd.DataFrame(cands).to_csv(HERE / "candidate_region_diagnostics.csv", index=False)
    return pd.DataFrame(rows), pd.DataFrame(cands), recs


# ----------------------------------------------------------------------------- 031-8_6 vs 031-6_2 structure
def self_similarity(points, extent, res=8.0, sigma=8.0, exclude_um=120.0):
    """Repeated-structure index: best normalised autocorrelation of the nuclear density at shifts > exclude_um."""
    n = (np.ceil(np.asarray(extent) / res)).astype(int)
    img = np.zeros((n[1], n[0]))
    ij = np.clip((points / res).astype(int), 0, n - 1)
    np.add.at(img, (ij[:, 1], ij[:, 0]), 1.0)
    d = gaussian_filter(img, sigma / res)
    d = d - d.mean()
    F = np.fft.rfft2(d, s=(2 * n[1], 2 * n[0]))
    ac = np.fft.irfft2(F * np.conj(F), s=(2 * n[1], 2 * n[0]))
    ac = np.fft.fftshift(ac) / ac[0, 0]
    cy, cx = n[1], n[0]
    yy, xx = np.mgrid[0:ac.shape[0], 0:ac.shape[1]]
    far = np.hypot((yy - cy) * res, (xx - cx) * res) > exclude_um
    return float(ac[far].max())


def structure_summary():
    out = {}
    for s in ("031-6_2", "031-8_6"):
        S = load_sample(s)
        rec = {}
        for key, mod in (("HE", S.he), ("IF", S.iff)):
            fm = prev.load_feats(prev.CACHE / f"feats_{s}_{key}_2um.npz")
            res = fm.res_um
            ext = mod.frame.extent_um()
            pts = mod.nuclei_um
            tissue_mm2 = float(fm.maps["tissue_bin"].sum() * res * res / 1e6)
            env_mm2 = float(fm.maps["envelope"].sum() * res * res / 1e6)
            bb = pts.max(0) - pts.min(0)
            rec[key] = {"fov_um": list(ext), "fov_mm2": ext[0] * ext[1] / 1e6, "nuclei": int(len(pts)),
                        "nuclei_bbox_um": bb.tolist(), "nuclei_bbox_mm2": float(np.prod(bb) / 1e6),
                        "nuclei_hull_mm2": float(ConvexHull(pts).volume / 1e6), "tissue_mm2": tissue_mm2, "tissue_envelope_mm2": env_mm2,
                        "density_per_tissue_envelope_mm2": len(pts) / max(env_mm2, 1e-9),
                        "self_similarity_index": self_similarity(pts, ext)}
        rec["fov_area_ratio_IF_over_HE"] = rec["IF"]["fov_mm2"] / rec["HE"]["fov_mm2"]
        rec["tissue_envelope_ratio_IF_over_HE"] = rec["IF"]["tissue_envelope_mm2"] / rec["HE"]["tissue_envelope_mm2"]
        rec["density_ratio_HE_over_IF"] = rec["HE"]["density_per_tissue_envelope_mm2"] / rec["IF"]["density_per_tissue_envelope_mm2"]
        out[s] = rec
    return out


# ----------------------------------------------------------------------------- synthetic sweep
def sweep():
    rows = []
    for p in sorted(CACHE.glob("syn_*.json")):
        r = json.loads(p.read_text())
        for m, v in r["results"].items():
            rows.append({"case": r["case"], "cond": r["cond"], "factor": r["factor"], "seed": r["seed"], "method": m,
                         **{f"p_{k}": vv for k, vv in r["params"].items()}, "overlap_achieved": r["overlap_achieved"],
                         "n_fixed": r["n_fixed"], "n_moving": r["n_moving"], **v})
    df = pd.DataFrame(rows)
    if df.empty:
        return df, df
    df.to_csv(HERE / "difficulty_sweep.csv", index=False)
    summ = (df.groupby(["factor", "cond", "method"])
              .agg(n=("case", "count"), region_found=("region_found", "mean"), reg_success=("reg_success", "mean"),
                   accepted=("accepted", "mean"), correct_accept=("correct_accept", "mean"), false_accept=("false_accept", "mean"),
                   median_err_um=("err_um", "median"), true_id=("true_id", "median"), le_3um=("le_3um", "median"),
                   null_le_3um=("null_le_3um", "median"), overlap=("overlap_achieved", "mean"), runtime_s=("runtime_s", "median"))
              .reset_index())
    summ.to_csv(HERE / "difficulty_sweep_summary.csv", index=False)
    return df, summ


# ----------------------------------------------------------------------------- figures
def fig_ablation(ab):
    if ab.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, s in zip(axes, ("031-6_2", "031-8_6")):
        d = ab[ab["sample"] == s]
        x = np.arange(4); w = 0.38
        for k, f in enumerate(("HE", "IF")):
            dd = d[d.fixed == f].set_index("variant").reindex(["X0", "X1", "X2", "X3"])
            vals = dd["le_3um"].fillna(0).to_numpy()
            bars = ax.bar(x + (k - 0.5) * w, vals, w, label=f"{f} fixed", color=["#1f77b4", "#ff7f0e"][k])
            nulls = dd["null_le_3um"].fillna(0).to_numpy()
            ax.scatter(x + (k - 0.5) * w, nulls, color="k", marker="_", s=120, zorder=3)
            for b, (_, row) in zip(bars, dd.iterrows()):
                lab = {"UNIQUE / Accepted": "ACC"}.get(row["safety"], str(row["safety"])[:9]) if isinstance(row["safety"], str) else "—"
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, lab, ha="center", fontsize=7, rotation=90)
        ax.set_xticks(x, ["X0\nbbox / all", "X1\nunion / all", "X2\nbbox / crop", "X3 (E0x)\nunion / crop"])
        ax.set_title(f"{s}"); ax.axhline(0.25, ls=":", c="gray", lw=1)
    axes[0].set_ylabel("≤3 µm proximity (IF frame)\n(black tick = mirror null)")
    axes[0].legend(fontsize=8)
    fig.suptitle("E0x ablation: canvas extension × ICP field restriction (dotted = prespecified 0.25)")
    fig.tight_layout(); fig.savefig(HERE / "e0x_ablation.png", dpi=150); plt.close(fig)


def fig_sweep(summ):
    if summ.empty:
        return
    factors = [f for f in synth.LEVELS]
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.5))
    for ax, f in zip(axes.ravel(), factors):
        lv = synth.LEVELS[f]
        for m, c in (("X3_L", "#1f77b4"), ("X0_L", "#d62728"), ("X3_S", "#2ca02c")):
            ys, fs = [], []
            for v in lv:
                cid = "base" if v == synth.BASE[f] else f"{f}={v:g}"
                r = summ[(summ.cond == cid) & (summ.method == m)]
                ys.append(r["correct_accept"].iloc[0] if len(r) else np.nan)
                fs.append(r["false_accept"].iloc[0] if len(r) else np.nan)
            if not np.all(np.isnan(ys)):
                ax.plot(range(len(lv)), ys, "o-", c=c, label=m)
                if np.nansum(fs) > 0:
                    ax.plot(range(len(lv)), fs, "x--", c=c, alpha=0.7)
        ax.set_xticks(range(len(lv)), [f"{v:g}" for v in lv]); ax.set_ylim(-0.05, 1.05); ax.set_title(f)
    axes[0, 0].set_ylabel("correct accept rate (n=3)"); axes[1, 0].set_ylabel("correct accept rate (n=3)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Synthetic difficulty sweep (one factor at a time from the base condition; x = false accept)")
    fig.tight_layout(); fig.savefig(HERE / "difficulty_sweep.png", dpi=150); plt.close(fig)
    grid = summ[(summ.factor == "grid") & (summ.method == "X3_L")]
    if len(grid):
        R = [0.25, 0.5, 1.0]; N = [0.0, 0.5, 0.85]
        M = np.full((3, 3), np.nan)
        for i, r in enumerate(R):
            for j, n in enumerate(N):
                q = grid[grid.cond == f"grid ratio={r:g} ns={n:g}"]
                if len(q):
                    M[i, j] = q["correct_accept"].iloc[0]
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, "—" if np.isnan(M[i, j]) else f"{M[i, j]:.2f}", ha="center", va="center", color="w")
        ax.set_xticks(range(3), [f"{n:g}" for n in N]); ax.set_yticks(range(3), [f"{r:g}" for r in R])
        ax.set_xlabel("non-shared nuclei fraction"); ax.set_ylabel("FOV size ratio (moving / fixed)")
        ax.set_title("E0x correct accept: FOV mismatch × non-shared"); fig.colorbar(im, ax=ax)
        fig.tight_layout(); fig.savefig(HERE / "difficulty_fov_vs_nonshared.png", dpi=150); plt.close(fig)


def fig_candidates(recs):
    for s in ("031-8_6", "031-6_2"):
        r = recs.get((s, "HE", "X3"))
        if not r:
            continue
        fm = prev.load_feats(prev.CACHE / f"feats_{s}_HE_2um.npz")
        S = load_sample(s)
        fig, ax = plt.subplots(figsize=(7, 7))
        ext = S.he.frame.extent_um()
        ax.imshow(fm.maps["tissue"], cmap="gray_r", extent=(0, fm.shape[1] * fm.res_um, fm.shape[0] * fm.res_um, 0), alpha=0.6)
        ax.scatter(*S.he.nuclei_um.T, s=1, c="#c2185b", alpha=0.4, label="H&E nuclei")
        cols = plt.cm.tab10(np.arange(10))
        for c in r["candidates"]:
            poly = np.asarray(c["overlap"]["overlap_polygon_fixed_um"])
            pp = np.vstack([poly, poly[:1]])
            fe = c.get("final_eval") or {}
            ax.plot(pp[:, 0], pp[:, 1], c=cols[c["rank"]], lw=2.5 if c["rank"] == 0 else 1.2,
                    label=f"#{c['rank']} z={c['score_z']:.2f} gap={c['relative_gap_to_top']:.3f} ≤3µm={fe.get('le_3um', float('nan')):.3f}")
        ax.set_xlim(0, ext[0]); ax.set_ylim(ext[1], 0); ax.set_aspect("equal")
        ax.set_title(f"{s}: E0x (X3, HE fixed) top candidates — verdict {r['verdict']}\n(all PREVIEW ONLY unless accepted)")
        ax.legend(fontsize=7, loc="lower left")
        fig.tight_layout(); fig.savefig(HERE / f"candidate_regions_{s}.png", dpi=140); plt.close(fig)


def main():
    ab, cd, recs = ablation()
    st = structure_summary()
    df, summ = sweep()
    fig_ablation(ab); fig_sweep(summ); fig_candidates(recs)
    diag = {"date": "2026-10-20", "protocol": "protocol.md (fixed before evaluation runs)",
            "acceptance_rule": "UNIQUE AND <=3um >= 0.25 AND >= 3x mirror null (unchanged)",
            "ablation_rows": len(ab), "candidate_rows": len(cd), "synthetic_runs": int(len(df)),
            "synthetic_cases": int(df["case"].nunique()) if len(df) else 0,
            "structure": st,
            "ablation": ab.to_dict(orient="records") if len(ab) else [],
            "synthetic_summary": summ.to_dict(orient="records") if len(summ) else [],
            "false_accepts_total": int(df["false_accept"].sum()) if len(df) else None,
            "note": "Real pairs were inspected during development; they are descriptive, not an independent test."}
    (HERE / "diagnostic_summary.json").write_text(json.dumps(json_safe(diag), indent=1, ensure_ascii=False))
    print("ablation", len(ab), "candidates", len(cd), "synthetic runs", len(df))


if __name__ == "__main__":
    main()
