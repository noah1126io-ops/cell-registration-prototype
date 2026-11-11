"""Strategy Planner prototype (RESEARCH MODE ONLY).  Rule-based; reads a Data Profiler profile (pre-search only) and
returns a plan that the existing engine can execute with its current options.  It never removes a source that has
usable input, never auto-accepts TISSUE_LEVEL, never treats UNKNOWN as SAME, and never relaxes the production gates.

Thresholds marked PROVISIONAL are placeholders derived from 2 development samples (031-6_2, 031-8_6) and the derived
stress cases of 2026-11-11; they are NOT calibrated and must not be used to accept anything automatically.
"""
from __future__ import annotations

import copy

PLANNER_VERSION = "adaptive_strategy 0.1 (2026-11-11, research prototype)"

# PROVISIONAL (see adaptive_strategy_rules.md for the evidence behind each value)
DEFAULT_RULES = {
    "image_min_cleft_strength": 0.15,        # moving-side band-passed cleft std; 8_6 0.28, 6_2 0.24
    "image_max_occ_nuclear_corr": 0.5,       # occ map should not be the nuclear stain
    "points_min_nuclei_per_mm2": 300.0,      # point-density search needs texture (6_2 HE: 1242 nuclei / 4.6 mm2 = 270)
    "small_field_area_ratio": 10.0,          # fixed/moving area above this -> uniqueness evidence weak (IF500 cases)
    "repetitive_autocorr": 0.30,             # cleft self-similarity at > 80 um shifts
    "min_tissue_fraction": 0.05,
}

STRATEGY_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "cellreg adaptive strategy plan (research)",
    "type": "object",
    "required": ["planner_version", "sources", "primary_source", "section_relation", "validation", "refinement", "auto_decision", "reasons"],
    "properties": {
        "planner_version": {"type": "string"},
        "sources": {"type": "array", "items": {"enum": ["image", "points"]}, "description": "every source with usable input; candidates of all are kept"},
        "primary_source": {"enum": ["image", "points", None]},
        "engine_options": {"type": "object", "description": "registration_core.WorkflowConfig overrides that realise the plan"},
        "section_relation": {"type": "object", "properties": {"value": {"enum": ["SAME", "DIFFERENT", "UNKNOWN"]},
                                                              "confidence": {"type": "string"}, "gate": {"enum": ["same", "adjacent", "both"]}}},
        "validation": {"type": "object", "properties": {
            "tissue_level": {"type": "boolean"}, "nuclear_level": {"type": "boolean"}, "transform_plausibility": {"type": "boolean"},
            "candidate_uniqueness": {"type": "boolean"}, "independent_evidence": {"type": "array", "items": {"type": "string"}}}},
        "refinement": {"type": "object", "properties": {"nuclear_icp_final": {"type": "boolean"}, "tissue_refine": {"type": "boolean"}}},
        "auto_decision": {"type": "object", "properties": {"cell_level_auto_accept_allowed": {"type": "boolean"},
                                                            "tissue_level_auto_accept_allowed": {"const": False},
                                                            "workflow_d_export_allowed_without_review": {"type": "boolean"}}},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
        "human_review": {"type": "array", "items": {"type": "string"}},
        "reasons": {"type": "array", "items": {"type": "string"}},
    },
}


def plan(profile: dict, rules: dict | None = None) -> dict:
    r = {**DEFAULT_RULES, **(rules or {})}
    f, m, rel = profile["fixed"], profile["moving"], profile["relation"]
    reasons, flags, review = [], [], []
    # 1. which sources can run at all (availability only - a low score never removes a source)
    img_ok = bool(profile["images_both"] and f["image_usable"] and m["image_usable"] and profile["coordinates_consistent"])
    pts_ok = bool(profile["nuclei_both"] and profile["coordinates_consistent"])
    sources = [s for s, ok in (("image", img_ok), ("points", pts_ok)) if ok]
    if not profile["coordinates_consistent"]:
        review.append("pixel spacing / coordinate units missing or points outside the imaged extent: fix metadata first")
    # 2. what each source is likely to carry (pre-search, one side at a time)
    img_struct = img_ok and m["cleft_strength"] >= r["image_min_cleft_strength"] and m["occ_nuclear_corr"] <= r["image_max_occ_nuclear_corr"] \
        and min(f["tissue_fraction"], m["tissue_fraction"]) >= r["min_tissue_fraction"]
    if img_ok and not img_struct:
        flags.append("WEAK_TISSUE_STRUCTURE")
        reasons.append("image source kept but structural signal weak (cleft / occupancy / occ-nuclear separation)")
    pts_texture = pts_ok and min(f["nuclei_per_mm2"], m["nuclei_per_mm2"]) >= r["points_min_nuclei_per_mm2"]
    if pts_ok and not pts_texture:
        flags.append("SPARSE_NUCLEI")
    if profile["fov_area_ratio"] >= r["small_field_area_ratio"]:
        flags.append("SMALL_MOVING_FIELD")
        review.append("moving field small relative to fixed: uniqueness of any region is weak, confirm visually")
    if m["repetitiveness"] >= r["repetitive_autocorr"]:
        flags.append("REPETITIVE_STRUCTURE")
    if m["boundary_info"] < 0.05:
        flags.append("NO_TISSUE_BOUNDARY_IN_FIELD")
    # 3. section relation: metadata decides the gate; UNKNOWN / CONFLICT never becomes SAME
    val, conf = rel["value"], rel["confidence"]
    if conf == "CONFLICT":
        gate = "both"; review.append(f"section relation records conflict (metadata {rel['metadata']} vs file name {rel['filename_hint']}): confirm provenance")
    elif val == "SAME":
        gate = "same"
    elif val == "DIFFERENT":
        gate = "adjacent"
    else:
        gate = "both"; review.append("section relation UNKNOWN: cell-level acceptance needs the relation to be confirmed")
    nuclear_level = pts_ok and gate in ("same", "both")
    cell_auto = bool(gate == "same" and conf == "CONFIRMED_METADATA" and pts_ok)
    if val == "DIFFERENT" and pts_ok:
        reasons.append("DIFFERENT sections: nuclear consistency is still computed and reported (it may hold for very close sections) "
                       "but cannot accept at cell level without review")
    # 4. primary source (ranking/display only; all sources' candidates are kept and compared)
    primary = "image" if img_struct else ("points" if pts_ok else ("image" if img_ok else None))
    # 5. refinement
    icp_final = bool(gate == "same" and pts_ok)
    p = {
        "planner_version": PLANNER_VERSION, "sources": sources, "primary_source": primary,
        "engine_options": {"localization": "auto" if "image" in sources else "points", "run_points_with_images": True,
                           "image_feature": "occ+cleft",
                           "section_relation": {"same": "same", "adjacent": "adjacent", "both": "unknown"}[gate]},
        "section_relation": {"value": val, "confidence": conf, "gate": gate},
        "validation": {"tissue_level": img_ok, "nuclear_level": nuclear_level, "transform_plausibility": True, "candidate_uniqueness": True,
                       "independent_evidence": [e for e, ok in (("cross-source agreement (image vs points)", len(sources) == 2),
                                                               ("global per-quarter localisation", img_ok),
                                                               ("landmarks / visual review", True)) if ok]},
        "refinement": {"nuclear_icp_final": icp_final, "tissue_refine": False},
        "auto_decision": {"cell_level_auto_accept_allowed": cell_auto, "tissue_level_auto_accept_allowed": False,
                          "workflow_d_export_allowed_without_review": cell_auto},
        "risk_flags": flags, "human_review": review, "reasons": reasons,
    }
    if not sources:
        p["human_review"].append("no usable source: manual landmarks required")
    return p


def validate_plan(p: dict) -> list:
    """Minimal schema + safety checks (no jsonschema dependency).  Returns a list of violations."""
    bad = [k for k in STRATEGY_SCHEMA["required"] if k not in p]
    if p.get("auto_decision", {}).get("tissue_level_auto_accept_allowed") is not False:
        bad.append("TISSUE_LEVEL must never be auto-accepted")
    if p.get("section_relation", {}).get("value") != "SAME" and p.get("auto_decision", {}).get("cell_level_auto_accept_allowed"):
        bad.append("cell-level auto acceptance requires a confirmed SAME relation")
    if p.get("auto_decision", {}).get("workflow_d_export_allowed_without_review") and not p.get("auto_decision", {}).get("cell_level_auto_accept_allowed"):
        bad.append("Workflow D export without review requires cell-level acceptance")
    return bad


def example_profile():
    """Small synthetic profile for tests/docs."""
    side = {"image_usable": True, "tissue_fraction": 0.5, "cleft_strength": 0.25, "occ_nuclear_corr": 0.0, "nuclei_per_mm2": 1500.0,
            "repetitiveness": 0.1, "boundary_info": 0.2}
    return {"fixed": copy.deepcopy(side), "moving": copy.deepcopy(side), "images_both": True, "nuclei_both": True, "coordinates_consistent": True,
            "fov_area_ratio": 4.0, "relation": {"value": "UNKNOWN", "confidence": "NONE", "metadata": "UNKNOWN", "filename_hint": "UNKNOWN"}}
