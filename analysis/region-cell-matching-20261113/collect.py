"""Aggregate all runs into the deliverable CSVs and figures (no recomputation).

  candidate_comparison.csv      one row per (dataset, case, pose, candidate): C1 rank, reference label, Graph/UOT/B2, decoy
  rerank_summary.csv            one row per (dataset, case, pose): C1 top-1 vs reranked top-1 vs alternatives, rescue / harm
  correspondence_results.csv    matching metrics: Experiment A (known global) + Experiment B/C (C1 top-1, reranked, correct)
  deformation_comparison.csv    Phase 2 rows: Experiment A, Experiment B (C1 top-1 / reranked top-1), REAL-D, REAL-S
"""
from __future__ import annotations

import json
from pathlib import Path

import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from src.registration_research import candidate_adapter as CA  # noqa: E402
from src.registration_research import candidate_reranking as RR  # noqa: E402
ALT = ("excess", "c1", "c3_le_3um", "graph_support", "uot_mass", "b2_rate_raw")


def candidate_poses(ds, case, path, which):
    """{candidate: (M, t)} for the pose used, and the engine's distinct probe (for the de-duplicated 'gap')."""
    if ds == "SYN-C1":
        d = json.loads((HERE / "cache" / f"c1syn_{case}.json").read_text())
        key = "final" if which == "final" else "search"
        poses = {c["id"]: (np.asarray(c[f"M_{key}"]), np.asarray(c[f"t_{key}"])) for c in d["candidates"]}
        wm = d["truth"]["wm"]
        return poses, RR.engine_probe((wm, wm))
    res = json.loads(Path(path).read_text())
    poses = {c.id: c.T(which) for c in CA.from_benchmark_result(Path(path), "same")}
    return poses, RR.engine_probe(res["inputs"]["moving"]["extent"])


def rerank_row(base, which, pool, recs, poses, probe):
    """C1 top-1 vs reranked top-1 (and alternatives) within one candidate pool (all sources, or one source)."""
    dec = RR.decide(recs, poses=poses, probe=probe)
    by = {r["candidate"]: r for r in recs}
    row = {**base, "pose": which, "pool": pool, "n_candidates": len(recs), "any_correct_in_topk": any(r.get("correct_region") for r in recs),
           "n_correct_in_topk": sum(bool(r.get("correct_region")) for r in recs),
           "correct_rank_c1": next((i + 1 for i, r in enumerate(recs) if r.get("correct_region")), None),
           "supported": dec["supported"], "reasons": ";".join(dec.get("reasons") or []), "s_top": dec.get("s_top"),
           "s_second": dec.get("s_second"), "second_distinct": dec.get("second_distinct")}
    for k in ALT:
        top = recs[RR.order(recs, k)[0]]["candidate"]
        row[f"top1_{k}"] = top
        row[f"top1_correct_{k}"] = bool(by[top].get("correct_region"))
    # effective rule: the reranked top-1 replaces the C1 top-1 only when Graph/UOT support it
    eff = dec["top"] if dec["supported"] else recs[0]["candidate"]
    row["top1_effective"] = eff
    row["top1_correct_effective"] = bool(by[eff].get("correct_region"))
    row["rescue_effective"] = (not row["top1_correct_c1"]) and row["top1_correct_effective"]
    row["harm_effective"] = row["top1_correct_c1"] and (not row["top1_correct_effective"])
    row["rescue"] = (not row["top1_correct_c1"]) and row["top1_correct_excess"]
    row["harm"] = row["top1_correct_c1"] and (not row["top1_correct_excess"])
    row["false_support"] = dec["supported"] and not row["top1_correct_excess"]
    return row


def load_p1():
    cand, rer, corr, defo = [], [], [], []
    for p in sorted((HERE / "runs").glob("p1_*.json")):
        if p.stem.endswith("_SMOKE"):
            continue
        d = json.loads(p.read_text())
        ds, case, meta = d["task"]["dataset"], d["task"]["case"], d["meta"]
        base = {"dataset": ds, "case": case, "split": meta.get("split", "real"), "cond": meta.get("cond", meta.get("kind")),
                "expected": meta.get("expected")}
        if ds == "SYN-C1":
            base["experiment"] = "C-K30" if case.startswith("K30_") else "B/C"
        for which, pose in d["poses"].items():
            recs = pose["records"]
            poses, probe = candidate_poses(ds, case, d["task"]["path"], which)
            pose["decision"] = RR.decide(recs, poses=poses, probe=probe)       # de-duplicated gap (protocol change log)
            by = {r["candidate"]: r for r in recs}
            for i, r in enumerate(recs):
                exc = (r.get("B2_matched_frac") or 0) - (r.get("decoy_B2_matched_frac") or 0) if r.get("status") == "OK" else np.nan
                cand.append({**base, "pose": which, **{k: v for k, v in r.items() if not isinstance(v, (dict, list))}, "excess": exc,
                             "rerank_position": pose["orders"]["excess"].index(r["candidate"]) + 1, "c1_order": i + 1})
            sources = list(dict.fromkeys(r["source"] for r in recs))
            for pool in ["all"] + (sources if len(sources) > 1 else []):
                pr = recs if pool == "all" else [r for r in recs if r["source"] == pool]
                rer.append(rerank_row(base, which, pool, pr, poses, probe))
            if ds == "SYN-C1":
                picks = {"c1_top1": recs[0]["candidate"], "rerank_top1": pose["orders"]["excess"][0]}
                cc = next((r["candidate"] for r in recs if r.get("correct_region")), None)
                if cc:
                    picks["correct_candidate"] = cc
                for tag, cid in picks.items():
                    r = by[cid]
                    for mth in ("B0", "B1", "B3", "B2"):
                        corr.append({**base, "pose": which, "pick": tag, "candidate": cid,
                                     "correct_region": r.get("correct_region"), "method": mth, "status": r.get("status"),
                                     "matched_n": r.get(f"{mth}_matched_n"), "precision": r.get(f"{mth}_precision"),
                                     "recall": r.get(f"{mth}_recall"), "f1": r.get(f"{mth}_f1"),
                                     "unmatched_precision": r.get(f"{mth}_unmatched_precision"),
                                     "unmatched_recall": r.get(f"{mth}_unmatched_recall"), "false_matches": r.get(f"{mth}_false_matches"),
                                     "common_n": r.get(f"{mth}_common_n")})
            for tag, p2 in pose.get("phase2", {}).items():
                for r in p2["rows"]:
                    defo.append({**base, "experiment": ("B" if ds == "SYN-C1" else "REAL-D") if not case.startswith("K30_") else "C-K30",
                                 "pose": which, "pick": tag,
                                 "candidate": p2["candidate"], "correct_region": p2["correct_region"], **r})
    return pd.DataFrame(cand), pd.DataFrame(rer), corr, defo


def candidate_recall():
    """Experiment C-K30: rank of the first correct-region candidate with the default n_fine=6 vs n_fine=30 (C1 only)."""
    rows = []
    for p in sorted((HERE / "cache").glob("c1syn_K30_*.json")):
        k30 = json.loads(p.read_text())
        base = HERE / "cache" / p.name.replace("c1syn_K30_", "c1syn_")
        k6 = json.loads(base.read_text()) if base.is_file() else None
        row = {"case": k30["task"]["case"], "cond": k30["task"]["cond"], "seed": k30["task"]["seed"]}
        for tag, d in (("k6", k6), ("k30", k30)):
            if d is None:
                continue
            cs = d["candidates"]
            row[f"{tag}_n_candidates"] = len(cs)
            row[f"{tag}_correct_rank"] = next((i + 1 for i, c in enumerate(cs) if c.get("correct_region")), None)
            row[f"{tag}_min_grid_err_um"] = min(c["grid_err_final_um"] for c in cs)
            row[f"{tag}_verdict"] = d["searches"]["points"]["verdict"]
            row[f"{tag}_runtime_s"] = d["c1_runtime_s"]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    candidate_recall().to_csv(HERE / "candidate_recall_k30.csv", index=False)
    cand, rer, corr, defo = load_p1()
    for p in sorted((HERE / "cache").glob("p2known_*.json")):
        d = json.loads(p.read_text())
        for r in d["deformation"]:
            defo.append({"experiment": "A", "split": "known_global", "pick": "known_global", "correct_region": True, **r})
        for r in d["matching"]:
            corr.append({"experiment": "A", "pick": "known_global", "correct_region": True, **r})
    rs = HERE / "runs" / "real_s.json"
    if rs.is_file():
        d = json.loads(rs.read_text())
        for r in d["deformation"]:
            defo.append({"experiment": "REAL-S", "dataset": "REAL-S", "case": d["case"], "pick": "saved_affine", **r})
    cand.to_csv(HERE / "candidate_comparison.csv", index=False)
    rer.to_csv(HERE / "rerank_summary.csv", index=False)
    pd.DataFrame(corr).to_csv(HERE / "correspondence_results.csv", index=False)
    pd.DataFrame(defo).to_csv(HERE / "deformation_comparison.csv", index=False)
    print(len(cand), len(rer), len(corr), len(defo))


if __name__ == "__main__":
    main()
