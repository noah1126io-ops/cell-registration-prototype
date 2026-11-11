"""Collect runs/*/result.json -> CSVs + figures (no engine call).  Missing runs are listed, never filled in.

  python -B collect.py
"""
from __future__ import annotations

import csv
import json
import math
import subprocess

import numpy as np

from common import CORRECT_UM, FIG, HERE, RUNS, load_cases, pose_params, probe

METHOD_LABEL = {"A": "A points (E0x)", "B": "B image tissue+nuclear", "C": "C image occ+cleft", "D": "D auto (production)"}


def load():
    out = {}
    for c in load_cases():
        for m in ("D", "B"):
            p = RUNS / f"{c['case_id']}__{m}" / "result.json"
            out[(c["case_id"], m)] = json.loads(p.read_text()) if p.exists() else None
    return out


def sacct():
    """Slurm wait / elapsed per array task (best effort)."""
    try:
        txt = subprocess.run(["sacct", "-n", "-P", "-X", "--name=arv-bench", "-S", "2026-11-11", "-o", "JobID,Submit,Start,End,Elapsed,State,NodeList"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:  # noqa: BLE001
        return {}
    from datetime import datetime
    rows = {}
    for line in txt.strip().splitlines():
        jid, sub, st, en, el, state, node = line.split("|")
        if "_" not in jid or state != "COMPLETED":
            continue
        idx = int(jid.split("_")[1])
        try:
            wait = (datetime.fromisoformat(st) - datetime.fromisoformat(sub)).total_seconds()
        except ValueError:
            wait = None
        rows[idx] = {"slurm_job": jid, "slurm_wait_s": wait, "slurm_elapsed": el, "node": node}
    return rows


def hit_rank(cands):
    for i, c in enumerate(cands):
        r = c.get("ref_search")
        if r and r["at_reference"]:
            return i
    return None


def src_cands(res, gate, src):
    return [c for c in res["gates"][gate]["candidates"] if c["source"] == src]


def loc_row(case, res, method, src):
    s = res["searches"].get(src)
    if s is None:
        return {"case_id": case["case_id"], "method": method, "executed": False, "reason": "source not run (" + "; ".join(res["image_status"].get("reasons") or ["n/a"]) + ")"}
    cands = src_cands(res, "same", src)
    top = cands[0] if cands else None
    r = top and top.get("ref_search")
    hr = hit_rank(cands)
    sc = [c["search"]["score"] for c in cands]
    ov = top and top["validation"]["overlap"]["overlap_frac_of_moving"]
    return {"case_id": case["case_id"], "expected": case["expected"], "method": method, "executed": True, "verdict": s["verdict"],
            "margin": s["margin"], "n_candidates": len(cands), "n_poses": s.get("n_poses"),
            "top1_score": sc[0] if sc else None, "top2_score": sc[1] if len(sc) > 1 else None,
            "ambiguity_top2_over_top1": (sc[1] / sc[0]) if len(sc) > 1 and sc[0] else None,
            "top1_theta_deg": top and top["pose_search"]["theta_deg"], "top1_scale": top and top["pose_search"]["scale"],
            "top1_tx": top and top["T_search"]["t"][0], "top1_ty": top and top["T_search"]["t"][1], "top1_overlap_frac": ov,
            "top1_ref_dist_um": r["probe_dist_um"] if r else None, "top1_at_reference": (r["at_reference"] if r else None),
            "topk_hit_rank": hr, "top1_landmark_median_um": (r or {}).get("landmark_median_um"), "runtime_s": s["runtime_s"]}


def val_row(case, res, method, gate_name, gate, src_of_interest):
    d = res["gates"][gate]["decision"]
    sel = next((c for c in res["gates"][gate]["candidates"] if c["id"] == d["selected_candidate"]), None)
    top_img = next((c for c in res["gates"][gate]["candidates"] if c["id"] == "image#0"), None)
    top_pts = next((c for c in res["gates"][gate]["candidates"] if c["id"] == "points#0"), None)
    rs = sel and sel.get("ref_final")
    v = (sel or {}).get("validation", {})
    q = (top_img or {}).get("validation", {}).get("quadrant_consistency") if gate == "adjacent" else None
    positive_decision = d["status"] in ("ACCEPTED", "POINT_ONLY_FALLBACK", "TISSUE_LEVEL")
    if case["expected"] == "POSITIVE":
        outcome = ("TP" if rs and rs["at_reference"] else "FP_WRONG_REGION") if positive_decision else "FN"
    else:
        outcome = "FP" if positive_decision else "TN"
    ff = v.get("fixed_frame", {})
    ref = sel and sel.get("refine", {})
    return {"case_id": case["case_id"], "expected": case["expected"], "method": method, "gate": gate_name, "gate_used": gate,
            "status": d["status"], "accepted": d["accepted"], "failure_stage": d.get("failure_stage"), "selected": d["selected_candidate"],
            "selected_ref_dist_um": rs and rs["probe_dist_um"], "selected_landmark_median_um": (rs or {}).get("landmark_median_um"),
            "outcome": outcome,
            "nuclear_le3um": ff.get("le_3um"), "nuclear_null_le3um": ff.get("null_le_3um"), "nuclear_over_null": ff.get("inlier_over_null"),
            "nuclear_pass": ff.get("geometric_pass"), "quadrant_n_agree": q and q["n_agree"], "quadrant_pass": q and q["pass"],
            "icp_ran": (ref or {}).get("ran"), "icp_shift_um": ((ref or {}).get("pose_change") or {}).get("median_shift_um"),
            "icp_rot_change_deg": ((ref or {}).get("pose_change") or {}).get("rot_change_deg"),
            "icp_scale_change": ((ref or {}).get("pose_change") or {}).get("scale_change"), "flags": "|".join((sel or {}).get("flags", [])),
            "image_top_ref_dist_um": ((top_img or {}).get("ref_search") or {}).get("probe_dist_um"),
            "points_top_ref_dist_um": ((top_pts or {}).get("ref_search") or {}).get("probe_dist_um"),
            "image_vs_points_top_um": _pd(top_img, top_pts, res), "reason": d["reason"]}


def _pd(a, b, res):
    if not a or not b:
        return None
    pr = probe(res["inputs"]["moving"]["extent"])
    A, B = a["T_search"], b["T_search"]
    return float(np.median(np.linalg.norm(pr @ np.asarray(A["M"]).T + A["t"] - (pr @ np.asarray(B["M"]).T + B["t"]), axis=1)))


def quadrant_global(R):
    rows = []
    for case_id in ("P8_6_orig", "P6_2_orig", "N8_6_IFmirror", "N6_2_IFmirror", "Ncross_HE6_2xIF8_6", "Ncross_HE8_6xIF6_2"):
        res = R.get((case_id, "D"))
        case = next(c for c in load_cases() if c["case_id"] == case_id)
        for q in range(4):
            p = RUNS / "quadrant_global" / f"{case_id}_Q{q}.json"
            if not p.exists():
                rows.append({"case_id": case_id, "quadrant": q, "executed": False}); continue
            g = json.loads(p.read_text())
            o = np.asarray(g["origin_in_parent_moving"], float); ext = np.asarray(g["extent"], float)
            cq = ext / 2
            top = (g["candidates"] or [None])[0]
            row = {"case_id": case_id, "expected": case["expected"], "quadrant": q, "executed": True, "verdict": g["verdict"], "margin": g["margin"],
                   "runtime_s": g["runtime_s"]}
            for name, T in (("full_image0", res and next((c["T_search"] for c in res["gates"]["same"]["candidates"] if c["id"] == "image#0"), None)),
                            ("reference", case["reference"])):
                if top is None or T is None:
                    row[f"dist_to_{name}_um"] = None; continue
                Mq, tq = np.asarray(top["M_search"]), np.asarray(top["t"])
                M, t = np.asarray(T["M"]), np.asarray(T["t"])
                d = float(np.linalg.norm(Mq @ cq + tq - (M @ (cq + o) + t)))
                dth = (pose_params(Mq)["theta_deg"] - pose_params(M)["theta_deg"] + 180) % 360 - 180
                row[f"dist_to_{name}_um"] = d; row[f"dtheta_to_{name}_deg"] = dth
                row[f"agree_{name}"] = bool(d <= 30 and abs(dth) <= 2)
            rows.append(row)
    return rows


def write(name, rows):
    if not rows:
        (HERE / name).write_text("# no rows: nothing executed\n"); return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(HERE / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main():
    R = load()
    cases = load_cases()
    sa = sacct()
    tasks = [f"{c['case_id']} {m}" for c in cases for m in ("D", "B")]
    loc, val, ext, qa = [], [], [], []
    missing = [f"{k[0]}__{k[1]}" for k, v in R.items() if v is None]
    for c in cases:
        rD, rB = R[(c["case_id"], "D")], R[(c["case_id"], "B")]
        for m, res, src in (("A", rD, "points"), ("B", rB, "image"), ("C", rD, "image")):
            if res is None:
                loc.append({"case_id": c["case_id"], "method": m, "executed": False, "reason": "run missing"}); continue
            loc.append(loc_row(c, res, m, src))
        for m, res in (("D", rD), ("B", rB)):
            if res is None:
                continue
            for gname in ("production", "same", "adjacent"):
                gate = res["production_gate"] if gname == "production" else gname
                val.append(val_row(c, res, m, gname, gate, None))
            t = res["timing"]
            idx = tasks.index(f"{c['case_id']} {m}")
            ext.append({**{k: v for k, v in val[-3].items()}, "engine_relation": res["engine_relation"]["value"], "metadata_relation": res["metadata_relation"],
                        "kind": c["kind"], "fixed_extent_um": "x".join(f"{e:.0f}" for e in res["inputs"]["fixed"]["extent"]),
                        "moving_extent_um": "x".join(f"{e:.0f}" for e in res["inputs"]["moving"]["extent"]),
                        "n_fixed_nuclei": res["inputs"]["fixed"]["n_points"], "n_moving_nuclei": res["inputs"]["moving"]["n_points"],
                        **{k: round(v, 1) for k, v in t.items()}, "peak_rss_mb": round(res["peak_rss_mb"]), "peak_tracemalloc_mb": round(res["peak_tracemalloc_mb"]),
                        "gpu_s": "n/a (engine is CPU-only)", **sa.get(idx, {})})
            for a in res["quadrant_audit"]:
                qa.append({"case_id": c["case_id"], "expected": c["expected"], "method": m, "candidate": a["id"], "n_agree": a["quadrant"]["n_agree"],
                           "pass": a["quadrant"]["pass"], "ref_dist_um": (a["ref"] or {}).get("probe_dist_um"),
                           "at_reference": (a["ref"] or {}).get("at_reference"),
                           "quadrant_shifts_um": "|".join(f"{q['shift_um']:.0f}" if q.get("ok") else "NA" for q in a["quadrant"]["quadrants"]),
                           "quadrant_dtheta_deg": "|".join(f"{q['d_theta_deg']:.0f}" if q.get("ok") else "NA" for q in a["quadrant"]["quadrants"]),
                           "quadrant_notes": "|".join(q.get("note", "") for q in a["quadrant"]["quadrants"] if not q.get("ok")),
                           "runtime_s": a["quadrant"]["runtime_s"]})
    write("localization_comparison.csv", loc)
    write("external_validation_results.csv", ext)
    write("validation_rows.csv", val)
    write("quadrant_audit_all_candidates.csv", qa)
    qg = quadrant_global(R)
    write("quadrant_global_audit.csv", qg)
    # confusion matrix
    cm = []
    for m in ("D", "B"):
        for g in ("production", "same", "adjacent"):
            rows = [v for v in val if v["method"] == m and v["gate"] == g]
            for exp in ("POSITIVE", "NEGATIVE", "PRESUMED_NEGATIVE"):
                rr = [v for v in rows if v["expected"] == exp]
                if not rr:
                    continue
                cnt = {o: sum(v["outcome"] == o for v in rr) for o in ("TP", "FP_WRONG_REGION", "FN", "FP", "TN")}
                cm.append({"method": m, "gate": g, "expected": exp, "n": len(rr), **cnt,
                           "cases": "; ".join(f"{v['case_id']}={v['status']}/{v['outcome']}" for v in rr)})
    write("validation_confusion_matrix.csv", cm)
    (HERE / "missing_runs.txt").write_text("\n".join(missing) + ("\n" if missing else "none\n"))
    figures(loc, ext, qa, qg)
    print("missing:", missing or "none")


# ----------------------------------------------------------------------------- figures (reference palette slots 1-3)
COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)


def figures(loc, ext, qa, qg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(exist_ok=True)
    pos = [c["case_id"] for c in load_cases() if c["expected"] == "POSITIVE"]
    # 1. top-1 distance to the reference (log) per method, positives
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for j, m in enumerate("ABC"):
        for i, cid in enumerate(pos):
            r = next((x for x in loc if x["case_id"] == cid and x["method"] == m), None)
            if not r or not r.get("executed") or r.get("top1_ref_dist_um") is None:
                ax.text(3, i + (j - 1) * 0.25, "not run", fontsize=6, color=INK2, va="center"); continue
            d = max(r["top1_ref_dist_um"], 1.0)
            ax.plot([d], [i + (j - 1) * 0.25], "o", ms=7, color=COL[m], mec="white", mew=1.5, label=METHOD_LABEL[m] if i == 0 else None)
            if r["topk_hit_rank"] not in (None, 0):
                ax.text(d * 1.15, i + (j - 1) * 0.25, f"hit @rank {r['topk_hit_rank']}", fontsize=6, color=INK2, va="center")
    ax.axvline(CORRECT_UM, color=INK2, lw=1, ls="--"); ax.text(CORRECT_UM * 1.05, -0.7, "100 µm (prespecified)", fontsize=7, color=INK2)
    ax.set_xscale("log"); ax.set_yticks(range(len(pos))); ax.set_yticklabels(pos, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("top-1 candidate: distance to reference pose (µm, log)", color=INK2, fontsize=8)
    ax.set_title("Global localisation on positive (derived) cases", fontsize=10, color=INK, loc="left")
    _ax(ax); ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "fig1_localization_top1.png", dpi=150); plt.close(fig)
    # 2. quadrant agreement of every image candidate: counts of n_agree per candidate class, one panel per feature
    if qa:
        cats = [("at reference", lambda r: r["at_reference"] is True), ("wrong region (positive case)", lambda r: r["expected"] == "POSITIVE" and not r["at_reference"]),
                ("negative case", lambda r: r["expected"] != "POSITIVE")]
        fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
        for ax, (lab, f) in zip(axs, cats):
            for j, (m, col, name) in enumerate((("D", COL["C"], "occ+cleft (C/D)"), ("B", COL["B"], "tissue+nuclear (B)"))):
                rows = [r for r in qa if r["method"] == m and f(r)]
                cnt = [sum(r["n_agree"] == k for r in rows) for k in range(5)]
                ax.bar(np.arange(5) + (j - 0.5) * 0.38, cnt, width=0.36, color=col, edgecolor="white", linewidth=2, label=name)
                for k, c in enumerate(cnt):
                    if c:
                        ax.text(k + (j - 0.5) * 0.38, c + 0.3, str(c), ha="center", fontsize=7, color=INK2)
            ax.axvline(2.5, color=INK2, ls="--", lw=1)
            ax.set_title(lab, fontsize=9, color=INK, loc="left"); ax.set_xticks(range(5)); _ax(ax); ax.grid(axis="y", color=GRID); ax.grid(axis="x", visible=False)
            ax.set_xlabel("quadrants agreeing (pass ≥ 3, dashed)", fontsize=8, color=INK2)
        axs[0].set_ylabel("image candidates", fontsize=8, color=INK2); axs[0].legend(fontsize=7, frameon=False, loc="upper left")
        fig.suptitle("Engine quadrant check (±100 µm / ±4° window) on every image candidate", fontsize=10, x=0.01, ha="left")
        fig.tight_layout(); fig.savefig(FIG / "fig2_quadrant_audit.png", dpi=150); plt.close(fig)
    # 3. runtimes per stage (D runs)
    rows = [e for e in ext if e["method"] == "D"]
    if rows:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        parts = [("C1_image_wall_s", "C1 image search", COL["C"]), ("C1_points_wall_s", "C1 point search", COL["A"]),
                 ("C2C3_same_wall_s", "C2/C3 same gate", "#eda100"), ("C2C3_adjacent_wall_s", "C2/C3 adjacent gate (+quadrant)", "#6250d6")]
        for i, e in enumerate(rows):
            left = 0
            for k, lab, col in parts:
                v = e.get(k) or 0
                ax.barh(i, v / 60, left=left / 60, color=col, height=0.6, edgecolor="white", linewidth=2, label=lab if i == 0 else None)
                left += v
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([e["case_id"] for e in rows], fontsize=7); ax.invert_yaxis()
        ax.set_xlabel("wall time (min, 4 CPU, Slurm mib-dbia)", fontsize=8, color=INK2)
        ax.set_title("Runtime by stage (method D runs)", fontsize=10, color=INK, loc="left")
        _ax(ax); ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=4); fig.tight_layout(); fig.savefig(FIG / "fig3_runtime.png", dpi=150); plt.close(fig)
    # 4. profile features per case
    try:
        P = list(csv.DictReader(open(HERE / "profiles.csv")))
    except FileNotFoundError:
        P = []
    if P:
        feats = [("moving_cleft_strength", "IF cleft strength"), ("moving_occ_nuclear_corr", "IF occ–nuclear corr"),
                 ("moving_boundary_info", "IF boundary info"), ("moving_repetitiveness", "IF repetitiveness"), ("fov_area_ratio", "FOV area ratio HE/IF")]
        fig, axs = plt.subplots(1, len(feats), figsize=(11, 4.2), sharey=True)
        for ax, (k, lab) in zip(axs, feats):
            vals = [float(p[k]) for p in P]
            ax.barh(range(len(P)), vals, color=COL["A"], height=0.6, edgecolor="white", linewidth=2)
            ax.set_title(lab, fontsize=8, color=INK); _ax(ax)
        axs[0].set_yticks(range(len(P))); axs[0].set_yticklabels([p["case_id"] for p in P], fontsize=7); axs[0].invert_yaxis()
        fig.suptitle("Pre-search data profile (data_profiler 0.1)", fontsize=10, x=0.01, ha="left"); fig.tight_layout()
        fig.savefig(FIG / "fig4_profiles.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
