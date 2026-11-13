"""Prespecified reranking of C1 candidates by Graph/UOT/B2 correspondence (experiment_protocol.md §3.3).

The score is the B2 adoption rate in the candidate's overlap minus the adoption rate of the mirror decoy run through
the same pipeline (chance level for that density / overlap).  Raw UOT objective values are not used for ranking because
they depend on the number of nuclei and the overlap size.  The fact that Graph/UOT prefers a candidate is a
correspondence-consistency signal, not independent proof that the region is right.
"""
from __future__ import annotations

import math

import numpy as np

DISTINCT_UM = 40.0   # registration_core SearchConfig.distinct_um: poses closer than this are the same candidate region
RULE = {"min_excess": 0.10, "min_ratio_to_decoy": 3.0, "decoy_floor": 0.01, "min_gap": 0.05}   # fixed 2026-11-13

# descriptive alternative orderings (key, larger_is_better)
ALTERNATIVES = {"c1": None, "c3_le_3um": ("c3_le_3um", True), "graph_support": ("graph_support_median", True),
                "uot_mass": ("uot_mass_ratio_to_source", True), "b2_rate_raw": ("B2_matched_frac", True)}


def excess(rec) -> float:
    if rec.get("status") != "OK":
        return -math.inf
    return float(rec["B2_matched_frac"]) - float(rec.get("decoy_B2_matched_frac") or 0.0)


def order(records, key="excess"):
    """Indices of records best-first.  Ties (and TOO_FEW) fall back to the C1 order (= input order)."""
    idx = list(range(len(records)))
    if key == "c1":
        return idx
    if key == "excess":
        val = [excess(r) for r in records]
    else:
        name, larger = ALTERNATIVES[key]
        val = [(-math.inf if records[i].get("status") != "OK" or records[i].get(name) is None else
                (float(records[i][name]) if larger else -float(records[i][name]))) for i in idx]
    return sorted(idx, key=lambda i: (-val[i], i))


def same_region(pa, pb, probe, tol=DISTINCT_UM) -> bool:
    (Ma, ta), (Mb, tb) = pa, pb
    d = np.linalg.norm(probe @ np.asarray(Ma).T + ta - (probe @ np.asarray(Mb).T + tb), axis=1)
    return bool(np.median(d) < tol)


def engine_probe(moving_extent):
    """The probe registration_core's PoseSearch.distinct uses (moving centre +- 200 um)."""
    mc = np.asarray(moving_extent, float) / 2.0
    return mc + np.array([[0, 0], [200, 0], [0, 200], [-200, 0], [0, -200]], float)


def decide(records, rule=RULE, poses=None, probe=None):
    """Top candidate by the prespecified score and whether Graph/UOT support it (all three conditions).

    poses = {candidate: (M, t)} and probe: the 'gap' is measured against the best candidate of a DIFFERENT region
    (pooled image + point candidates can contain the same region twice; protocol change log 2026-11-13)."""
    if not records:
        return {"top": None, "supported": False, "reasons": ["no candidates"]}
    o = order(records)
    top = records[o[0]]
    s1 = excess(top)
    second = None
    for i in o[1:]:
        if poses is not None and probe is not None and same_region(poses[top["candidate"]], poses[records[i]["candidate"]], probe):
            continue
        second = i
        break
    s2 = excess(records[second]) if second is not None else -math.inf
    rate, dec = float(top.get("B2_matched_frac") or 0.0), float(top.get("decoy_B2_matched_frac") or 0.0)
    checks = {"excess": s1 >= rule["min_excess"],
              "ratio_to_decoy": rate >= rule["min_ratio_to_decoy"] * max(dec, rule["decoy_floor"]),
              "gap": (s1 - s2) >= rule["min_gap"] if math.isfinite(s2) else math.isfinite(s1)}
    return {"top": top["candidate"], "top_index": o[0], "order": [records[i]["candidate"] for i in o],
            "second_distinct": records[second]["candidate"] if second is not None else None,
            "s_top": s1 if math.isfinite(s1) else None, "s_second": s2 if math.isfinite(s2) else None,
            "supported": bool(math.isfinite(s1) and all(checks.values())), "checks": checks, "rule": dict(rule),
            "dedup": poses is not None, "reasons": [k for k, v in checks.items() if not v]}
