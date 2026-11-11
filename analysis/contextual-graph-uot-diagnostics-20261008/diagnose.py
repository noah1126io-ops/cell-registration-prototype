"""Diagnose the low real-data matched rate without changing the algorithm."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, cKDTree
from scipy.stats import ks_2samp, wasserstein_distance


ROOT = Path("/mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu")
SOURCE = ROOT / "analysis/contextual-graph-uot-20261008"
OUT = ROOT / "analysis/contextual-graph-uot-diagnostics-20261008"
REAL_INPUT = ROOT / "sweeps/real-grid27-20260908-v2/input"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(OUT))

from candidate_generation import build_candidate_rankings, build_candidate_set  # noqa: E402
from config import CANDIDATES, DIFFICULTIES, METHODS, REGIONAL, SEEDS, UOT  # noqa: E402
from contextual_uot import _extract, _run_plan  # noqa: E402
from graph_consistency import graph_support_scores  # noqa: E402
from local_fingerprint import compute_fingerprint_pair, pairwise_fingerprint_cost  # noqa: E402
from regional_pattern import match_regional_patterns  # noqa: E402
from synthetic_correspondence import assert_identity_alignment, generate_correspondence_sample  # noqa: E402
import diagnostic_visualization as viz  # noqa: E402


@dataclass
class PipelineState:
    fixed: np.ndarray
    moving: np.ndarray
    regional: object
    fingerprints: object
    fingerprint_cost: np.ndarray
    rankings: object
    candidates: object
    union_count: np.ndarray
    graph_support: np.ndarray
    selected_cost: np.ndarray
    plan: np.ndarray
    log: dict
    result: object


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_finite(name: str, values: np.ndarray) -> None:
    values = np.asarray(values)
    if not np.isfinite(values).all():
        invalid = int(np.count_nonzero(~np.isfinite(values)))
        raise ValueError(f"{name} contains {invalid} NaN/Inf values")


def load_real_points() -> tuple[np.ndarray, np.ndarray]:
    fixed = np.load(REAL_INPUT / "success_metric_fixed_points.npy", allow_pickle=False)
    moving = np.load(REAL_INPUT / "success_metric_moving_points.npy", allow_pickle=False)
    if fixed.shape != (1830, 2) or moving.shape != (1754, 2):
        raise RuntimeError(f"Unexpected real point shapes: fixed={fixed.shape}, moving={moving.shape}")
    validate_finite("real fixed points", fixed)
    validate_finite("real moving points", moving)
    return fixed, moving


def _union_counts(rankings) -> np.ndarray:
    source_k = int(CANDIDATES["source_k"])
    counts = []
    for row in range(len(rankings.spatial)):
        union = set(rankings.spatial[row, :source_k].tolist())
        union.update(rankings.fingerprint[row, :source_k].tolist())
        union.update(rankings.regional[row, :source_k].tolist())
        counts.append(len(union))
    return np.asarray(counts, dtype=int)


def run_frozen_m5(fixed: np.ndarray, moving: np.ndarray) -> PipelineState:
    regional = match_regional_patterns(fixed, moving)
    fingerprints = compute_fingerprint_pair(fixed, moving)
    fingerprint_cost = pairwise_fingerprint_cost(fingerprints)
    rankings = build_candidate_rankings(fixed, moving, fingerprint_cost, regional)
    candidates = build_candidate_set(rankings, fingerprint_cost, regional, include_region=True)
    union_count = _union_counts(rankings)
    graph_support = graph_support_scores(fixed, moving, candidates)
    selected_cost = candidates.base_cost + float(CANDIDATES["graph_weight"]) * (1.0 - graph_support)
    cost = np.full((len(moving), len(fixed)), float(CANDIDATES["outside_cost"]), dtype=float)
    np.put_along_axis(cost, candidates.indices, selected_cost, axis=1)
    plan, log, _ = _run_plan(cost)
    result = _extract(
        "M5 Full contextual graph-UOT",
        plan,
        log,
        moving,
        fixed,
        regional,
        allowed_indices=candidates.indices,
        candidates=candidates,
        graph_support=graph_support,
    )
    for name, values in (
        ("fingerprint cost", fingerprint_cost),
        ("candidate base cost", candidates.base_cost),
        ("graph support", graph_support),
        ("selected cost", selected_cost),
        ("UOT plan", plan),
        ("confidence", result.confidence),
    ):
        validate_finite(name, values)
    return PipelineState(
        fixed=fixed,
        moving=moving,
        regional=regional,
        fingerprints=fingerprints,
        fingerprint_cost=fingerprint_cost,
        rankings=rankings,
        candidates=candidates,
        union_count=union_count,
        graph_support=graph_support,
        selected_cost=selected_cost,
        plan=plan,
        log=log,
        result=result,
    )


def _slots_for_indices(candidate_indices: np.ndarray, selected: np.ndarray) -> np.ndarray:
    matches = candidate_indices == selected[:, None]
    if not np.all(np.sum(matches, axis=1) == 1):
        raise AssertionError("A selected fixed index is missing or duplicated in its candidate row.")
    return np.argmax(matches, axis=1)


def trace_pipeline(state: PipelineState, prefix: str = "real_moving") -> pd.DataFrame:
    n = len(state.moving)
    candidate_mass = np.take_along_axis(state.plan, state.candidates.indices, axis=1)
    local_order = np.argsort(candidate_mass, axis=1)[:, ::-1]
    ranked_indices = np.take_along_axis(state.candidates.indices, local_order, axis=1)
    ranked_mass = np.take_along_axis(candidate_mass, local_order, axis=1)
    ranked_cost = np.take_along_axis(state.selected_cost, local_order, axis=1)
    row_mass = np.sum(state.plan, axis=1)
    normalized_mass = np.divide(
        ranked_mass,
        row_mass[:, None],
        out=np.zeros_like(ranked_mass),
        where=row_mass[:, None] > 0,
    )
    source_mass = 1.0 / max(len(state.fixed), len(state.moving))
    top1 = ranked_indices[:, 0]
    top2 = ranked_indices[:, 1]
    top_slots = _slots_for_indices(state.candidates.indices, top1)
    top2_slots = _slots_for_indices(state.candidates.indices, top2)
    min_cost = np.min(state.selected_cost, axis=1)
    sorted_cost = np.sort(state.selected_cost, axis=1)
    effective_count = np.exp(state.result.normalized_row_entropy * np.log(state.candidates.indices.shape[1]))
    outside_mass = np.maximum(row_mass - np.sum(candidate_mass, axis=1), 0.0)
    row_ids = np.arange(n)
    base_min_slots = np.argmin(state.candidates.base_cost, axis=1)
    graph_min_slots = np.argmin(state.selected_cost, axis=1)
    base_min_indices = state.candidates.indices[row_ids, base_min_slots]
    graph_min_indices = state.candidates.indices[row_ids, graph_min_slots]
    spatial_rank1 = state.rankings.spatial[:, 0]
    regional_rank1 = state.rankings.regional[:, 0]
    fingerprint_rank1 = state.rankings.fingerprint[:, 0]
    source_k = int(CANDIDATES["source_k"])
    candidate_sets = [set(map(int, row)) for row in state.candidates.indices]
    retained_spatial = np.asarray(
        [sum(int(value) in candidate_sets[row] for value in state.rankings.spatial[row, :source_k]) for row in range(n)]
    )
    retained_regional = np.asarray(
        [sum(int(value) in candidate_sets[row] for value in state.rankings.regional[row, :source_k]) for row in range(n)]
    )
    retained_fingerprint = np.asarray(
        [sum(int(value) in candidate_sets[row] for value in state.rankings.fingerprint[row, :source_k]) for row in range(n)]
    )

    mass_component = 0.30 * np.clip(state.result.row_mass_ratio, 0.0, 1.0)
    concentration_component = 0.25 * np.clip(normalized_mass[:, 0], 0.0, 1.0)
    margin_component = 0.15 * np.clip(state.result.top1_top2_margin / 0.25, 0.0, 1.0)
    graph_component = 0.15 * state.graph_support[np.arange(n), top_slots]
    region_confidence = state.regional.cell_confidence
    region_component = 0.15 * region_confidence
    confidence_recomputed = (
        mass_component + concentration_component + margin_component + graph_component + region_component
    )

    status = state.result.status.astype(str)
    exclusive_reason = np.full(n, "unexpected", dtype=object)
    exclusive_reason[status == "unmatched"] = "row_mass_ratio_below_0.12"
    ambiguous = status == "ambiguous"
    exclusive_reason[ambiguous & (state.result.row_mass_ratio < 0.25)] = "row_mass_ratio_0.12_to_0.25"
    exclusive_reason[ambiguous & (state.result.row_mass_ratio >= 0.25) & (normalized_mass[:, 0] < 0.16)] = (
        "top1_concentration_below_0.16"
    )
    exclusive_reason[
        ambiguous
        & (state.result.row_mass_ratio >= 0.25)
        & (normalized_mass[:, 0] >= 0.16)
        & (state.result.top1_top2_margin < 0.01)
    ] = "top1_top2_margin_below_0.01"
    exclusive_reason[status == "matched"] = "all_matched_conditions_passed"

    data = {
        "moving_nucleus_id": [f"{prefix}_{index:06d}" for index in range(n)],
        "moving_index": np.arange(n),
        "moving_x_um": state.moving[:, 0],
        "moving_y_um": state.moving[:, 1],
        "regional_anchor_index": state.regional.cell_anchor_index,
        "regional_best_offset_x_um": state.regional.cell_offsets[:, 0],
        "regional_best_offset_y_um": state.regional.cell_offsets[:, 1],
        "regional_best_offset_norm_um": np.linalg.norm(state.regional.cell_offsets, axis=1),
        "regional_offset_hits_search_boundary": (
            (np.abs(state.regional.cell_offsets[:, 0]) >= float(REGIONAL["search_radius_um"]) - 1e-9)
            | (np.abs(state.regional.cell_offsets[:, 1]) >= float(REGIONAL["search_radius_um"]) - 1e-9)
        ),
        "regional_best_ncc": state.result.regional_ncc,
        "regional_second_ncc": state.regional.second_ncc[state.regional.cell_anchor_index],
        "regional_ncc_margin": state.result.regional_margin,
        "regional_peak_ratio": state.regional.peak_ratio[state.regional.cell_anchor_index],
        "regional_confidence_component_raw": region_confidence,
        "spatial_candidate_count": np.full(n, state.rankings.spatial.shape[1]),
        "regional_candidate_count": np.full(n, state.rankings.regional.shape[1]),
        "fingerprint_candidate_count": np.full(n, state.rankings.fingerprint.shape[1]),
        "candidate_union_count_before_prune": state.union_count,
        "uot_input_candidate_count": np.full(n, state.candidates.indices.shape[1]),
        "candidate_pruned_count": state.union_count - state.candidates.indices.shape[1],
        "retained_spatial_source20_count": retained_spatial,
        "retained_regional_source20_count": retained_regional,
        "retained_fingerprint_source20_count": retained_fingerprint,
        "spatial_rank1_fixed_index": spatial_rank1,
        "regional_rank1_fixed_index": regional_rank1,
        "fingerprint_rank1_fixed_index": fingerprint_rank1,
        "base_cost_rank1_fixed_index": base_min_indices,
        "graph_cost_rank1_fixed_index": graph_min_indices,
        "regional_rank1_differs_from_spatial": regional_rank1 != spatial_rank1,
        "fingerprint_rank1_differs_from_spatial": fingerprint_rank1 != spatial_rank1,
        "base_rank1_differs_from_spatial": base_min_indices != spatial_rank1,
        "graph_rank1_differs_from_base": graph_min_indices != base_min_indices,
        "uot_rank1_differs_from_graph_min_cost": top1 != graph_min_indices,
        "minimum_candidate_cost": min_cost,
        "second_minimum_candidate_cost": sorted_cost[:, 1],
        "minimum_cost_margin": sorted_cost[:, 1] - sorted_cost[:, 0],
        "uot_top1_cost": ranked_cost[:, 0],
        "uot_top2_cost": ranked_cost[:, 1],
        "uot_top1_top2_cost_margin": ranked_cost[:, 1] - ranked_cost[:, 0],
        "row_transport_mass": row_mass,
        "source_mass": np.full(n, source_mass),
        "row_mass_ratio": state.result.row_mass_ratio,
        "normalized_top1_mass": normalized_mass[:, 0],
        "normalized_top2_mass": normalized_mass[:, 1],
        "top1_top2_mass_margin": state.result.top1_top2_margin,
        "normalized_row_entropy": state.result.normalized_row_entropy,
        "effective_candidate_count": effective_count,
        "outside_candidate_mass_fraction": np.divide(
            outside_mass, row_mass, out=np.zeros_like(row_mass), where=row_mass > 0
        ),
        "top1_fixed_index": top1,
        "top2_fixed_index": top2,
        "reciprocal": state.result.reciprocal,
        "top1_spatial_distance_um": state.candidates.spatial_distance_um[np.arange(n), top_slots],
        "top1_region_residual_um": state.candidates.region_residual_um[np.arange(n), top_slots],
        "top1_fingerprint_cost": state.candidates.fingerprint_cost[np.arange(n), top_slots],
        "top1_fingerprint_similarity": state.candidates.fingerprint_similarity[np.arange(n), top_slots],
        "top1_region_cost": state.candidates.region_cost[np.arange(n), top_slots],
        "top1_region_similarity": state.candidates.region_similarity[np.arange(n), top_slots],
        "top1_graph_support": state.graph_support[np.arange(n), top_slots],
        "top1_base_cost_before_graph": state.candidates.base_cost[np.arange(n), top_slots],
        "top2_graph_support": state.graph_support[np.arange(n), top2_slots],
        "confidence_mass_contribution": mass_component,
        "confidence_concentration_contribution": concentration_component,
        "confidence_margin_contribution": margin_component,
        "confidence_graph_contribution": graph_component,
        "confidence_region_contribution": region_component,
        "confidence_recomputed": confidence_recomputed,
        "confidence_saved_algorithm": state.result.confidence,
        "confidence_recompute_absolute_error": np.abs(confidence_recomputed - state.result.confidence),
        "passes_unmatched_mass_floor_0.12": state.result.row_mass_ratio >= 0.12,
        "passes_matched_row_mass_0.25": state.result.row_mass_ratio >= 0.25,
        "passes_top1_concentration_0.16": normalized_mass[:, 0] >= 0.16,
        "passes_mass_margin_0.01": state.result.top1_top2_margin >= 0.01,
        "status": status,
        "exclusive_final_reason": exclusive_reason,
    }
    for rank in range(3):
        fixed_index = ranked_indices[:, rank]
        slots = _slots_for_indices(state.candidates.indices, fixed_index)
        data[f"rank{rank + 1}_fixed_index"] = fixed_index
        data[f"rank{rank + 1}_fixed_x_um"] = state.fixed[fixed_index, 0]
        data[f"rank{rank + 1}_fixed_y_um"] = state.fixed[fixed_index, 1]
        data[f"rank{rank + 1}_transport_mass"] = ranked_mass[:, rank]
        data[f"rank{rank + 1}_normalized_mass"] = normalized_mass[:, rank]
        data[f"rank{rank + 1}_total_cost"] = ranked_cost[:, rank]
        data[f"rank{rank + 1}_position_cost"] = state.candidates.position_cost[np.arange(n), slots]
        data[f"rank{rank + 1}_fingerprint_cost"] = state.candidates.fingerprint_cost[np.arange(n), slots]
        data[f"rank{rank + 1}_region_cost"] = state.candidates.region_cost[np.arange(n), slots]
        data[f"rank{rank + 1}_graph_support"] = state.graph_support[np.arange(n), slots]
    table = pd.DataFrame(data)
    if not table["moving_nucleus_id"].is_unique:
        raise AssertionError("Moving nucleus IDs are not unique.")
    return table


def verify_against_saved(trace: pd.DataFrame) -> dict:
    saved = pd.read_csv(SOURCE / "real_correspondence_candidates.csv")
    if len(saved) != len(trace):
        raise AssertionError("Saved and recomputed real row counts differ.")
    exact_columns = {
        "top1_fixed_index": "top1_fixed_index",
        "top2_fixed_index": "top2_fixed_index",
        "reciprocal": "reciprocal",
        "status": "status",
    }
    numeric_columns = {
        "top1_transport_mass": "rank1_transport_mass",
        "top2_transport_mass": "rank2_transport_mass",
        "top1_top2_margin": "top1_top2_mass_margin",
        "normalized_row_entropy": "normalized_row_entropy",
        "total_transported_mass": "row_transport_mass",
        "row_mass_ratio": "row_mass_ratio",
        "regional_ncc": "regional_best_ncc",
        "regional_ncc_margin": "regional_ncc_margin",
        "fingerprint_similarity": "top1_fingerprint_similarity",
        "graph_support_score": "top1_graph_support",
        "affine_candidate_distance_um": "top1_spatial_distance_um",
        "confidence": "confidence_saved_algorithm",
    }
    exact = {name: bool(np.array_equal(saved[left].to_numpy(), trace[right].to_numpy())) for name, (left, right) in {
        key: (key, value) for key, value in exact_columns.items()
    }.items()}
    maximum_errors = {
        name: float(np.max(np.abs(saved[left].to_numpy(dtype=float) - trace[right].to_numpy(dtype=float))))
        for name, (left, right) in {key: (key, value) for key, value in numeric_columns.items()}.items()
    }
    if not all(exact.values()) or max(maximum_errors.values()) > 1e-12:
        raise AssertionError(f"Frozen result reproduction failed: exact={exact}, max_errors={maximum_errors}")
    return {"exact_columns": exact, "numeric_max_absolute_errors": maximum_errors}


def stage_diagnostics(state: PipelineState, trace: pd.DataFrame) -> pd.DataFrame:
    n = len(trace)
    stage_values = [
        ("Affine", np.full(n, len(state.fixed)), np.ones(n), "No filtering; 1,830 fixed nuclei form the search universe"),
        (
            "Regional NCC",
            trace["regional_candidate_count"].to_numpy(),
            trace["regional_best_ncc"].to_numpy(),
            f"Ranks 30 candidates; {int(trace['regional_offset_hits_search_boundary'].sum()):,} offsets hit ±18 um boundary; no deletion",
        ),
        ("Fingerprint", trace["fingerprint_candidate_count"].to_numpy(), trace["top1_fingerprint_similarity"].to_numpy(), "Ranks candidates inside spatial top-80; no moving-nucleus rejection"),
        ("Candidate union/prune", trace["uot_input_candidate_count"].to_numpy(), trace["minimum_candidate_cost"].to_numpy(), "Union of three top-20 lists is cost-ranked and pruned to 30"),
        ("Graph", trace["uot_input_candidate_count"].to_numpy(), trace["top1_graph_support"].to_numpy(), "No deletion; adds 0.75 × (1 - support) to candidate cost"),
        ("UOT", trace["uot_input_candidate_count"].to_numpy(), trace["row_mass_ratio"].to_numpy(), "1,295 rows receive mass ratio <0.12"),
        ("Confidence classification", trace["uot_input_candidate_count"].to_numpy(), trace["confidence_saved_algorithm"].to_numpy(), "25 matched; classification uses mass/concentration/margin, not final confidence"),
    ]
    rows = []
    for stage, counts, scores, reason in stage_values:
        counts = np.asarray(counts, dtype=float)
        scores = np.asarray(scores, dtype=float)
        rows.append(
            {
                "stage": stage,
                "input_moving_nuclei": n,
                "valid_moving_nuclei": int(np.count_nonzero(np.isfinite(scores))),
                "candidate_count_p05": float(np.percentile(counts, 5)),
                "candidate_count_median": float(np.median(counts)),
                "candidate_count_p95": float(np.percentile(counts, 95)),
                "zero_candidate_nuclei": int(np.count_nonzero(counts == 0)),
                "valid_score_nuclei": int(np.count_nonzero(np.isfinite(scores))),
                "score_p05": float(np.percentile(scores[np.isfinite(scores)], 5)),
                "score_median": float(np.median(scores[np.isfinite(scores)])),
                "score_p95": float(np.percentile(scores[np.isfinite(scores)], 95)),
                "nan_count": int(np.count_nonzero(np.isnan(scores))),
                "inf_count": int(np.count_nonzero(np.isinf(scores))),
                "operation_and_main_degradation": reason,
            }
        )
    return pd.DataFrame(rows)


def point_density_and_knn(points: np.ndarray) -> tuple[float, np.ndarray]:
    hull_area = float(ConvexHull(points).volume)
    density = len(points) / hull_area
    knn = cKDTree(points).query(points, k=2)[0][:, 1]
    return density, knn


def synthetic_point_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    correspondences = pd.read_csv(SOURCE / "correspondence_results.csv")
    correspondences = correspondences.loc[
        (correspondences["method"] == "M5 Full contextual graph-UOT")
        & correspondences["scenario_id"].str.startswith("primary_")
    ].copy()
    graph = pd.read_csv(SOURCE / "graph_candidate_diagnostics.csv")
    graph = graph.loc[graph["scenario_id"].str.startswith("primary_")].copy()
    graph["candidate_total_cost"] = graph["base_cost"] + float(CANDIDATES["graph_weight"]) * (
        1.0 - graph["graph_support"]
    )
    minimum = graph.groupby(["scenario_id", "moving_index"], as_index=False).agg(
        minimum_candidate_cost=("candidate_total_cost", "min"),
        candidate_count=("fixed_index", "count"),
    )
    top_cost = correspondences[["scenario_id", "moving_index", "top1_fixed_index"]].merge(
        graph[["scenario_id", "moving_index", "fixed_index", "candidate_total_cost"]],
        left_on=["scenario_id", "moving_index", "top1_fixed_index"],
        right_on=["scenario_id", "moving_index", "fixed_index"],
        how="left",
        validate="one_to_one",
    )[["scenario_id", "moving_index", "candidate_total_cost"]].rename(
        columns={"candidate_total_cost": "uot_top1_cost"}
    )
    correspondences = correspondences.merge(minimum, on=["scenario_id", "moving_index"], validate="one_to_one")
    correspondences = correspondences.merge(top_cost, on=["scenario_id", "moving_index"], validate="one_to_one")
    density_rows = []
    for difficulty, settings in DIFFICULTIES.items():
        for seed in SEEDS:
            scenario_id = f"primary_{difficulty.lower()}_{seed}"
            sample = generate_correspondence_sample(seed=seed, difficulty=difficulty, **settings)
            assert_identity_alignment(sample)
            density, knn = point_density_and_knn(sample.moving_points)
            true_error = np.full(len(sample.moving_points), np.nan)
            common = sample.true_fixed_index >= 0
            true_error[common] = np.linalg.norm(
                sample.fixed_points[sample.true_fixed_index[common]] - sample.moving_points[common], axis=1
            )
            density_rows.append(
                pd.DataFrame(
                    {
                        "scenario_id": scenario_id,
                        "moving_index": np.arange(len(sample.moving_points)),
                        "moving_nuclear_density_per_um2": density,
                        "moving_knn_distance_um": knn,
                        "true_affine_correspondence_distance_um": true_error,
                    }
                )
            )
    geometry = pd.concat(density_rows, ignore_index=True)
    correspondences = correspondences.merge(geometry, on=["scenario_id", "moving_index"], validate="one_to_one")
    correspondences["dataset_group"] = correspondences["difficulty"]
    return correspondences, graph


def distribution_points(real_trace: pd.DataFrame, fixed: np.ndarray, moving: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    synthetic, _ = synthetic_point_table()
    real_density, real_knn = point_density_and_knn(moving)
    real = pd.DataFrame(
        {
            "dataset_group": "Real",
            "moving_nuclear_density_per_um2": real_density,
            "moving_knn_distance_um": real_knn,
            "true_affine_correspondence_distance_um": np.nan,
            "affine_candidate_distance_um": real_trace["top1_spatial_distance_um"],
            "regional_ncc": real_trace["regional_best_ncc"],
            "regional_ncc_margin": real_trace["regional_ncc_margin"],
            "fingerprint_similarity": real_trace["top1_fingerprint_similarity"],
            "graph_support_score": real_trace["top1_graph_support"],
            "row_mass_ratio": real_trace["row_mass_ratio"],
            "row_transport_mass": real_trace["row_transport_mass"],
            "normalized_row_entropy": real_trace["normalized_row_entropy"],
            "confidence": real_trace["confidence_saved_algorithm"],
            "top1_top2_margin": real_trace["top1_top2_mass_margin"],
            "minimum_candidate_cost": real_trace["minimum_candidate_cost"],
            "uot_top1_cost": real_trace["uot_top1_cost"],
        }
    )
    synthetic_points = pd.DataFrame(
        {
            "dataset_group": synthetic["dataset_group"],
            "moving_nuclear_density_per_um2": synthetic["moving_nuclear_density_per_um2"],
            "moving_knn_distance_um": synthetic["moving_knn_distance_um"],
            "true_affine_correspondence_distance_um": synthetic["true_affine_correspondence_distance_um"],
            "affine_candidate_distance_um": synthetic["affine_candidate_distance_um"],
            "regional_ncc": synthetic["regional_ncc"],
            "regional_ncc_margin": synthetic["regional_ncc_margin"],
            "fingerprint_similarity": synthetic["fingerprint_similarity"],
            "graph_support_score": synthetic["graph_support_score"],
            "row_mass_ratio": synthetic["row_mass_ratio"],
            "row_transport_mass": synthetic["total_transported_mass"],
            "normalized_row_entropy": synthetic["normalized_row_entropy"],
            "confidence": synthetic["confidence"],
            "top1_top2_margin": synthetic["top1_top2_margin"],
            "minimum_candidate_cost": synthetic["minimum_candidate_cost"],
            "uot_top1_cost": synthetic["uot_top1_cost"],
        }
    )
    points = pd.concat([synthetic_points, real], ignore_index=True)
    rows = []
    variables = [column for column in points.columns if column != "dataset_group"]
    for group, group_table in points.groupby("dataset_group", sort=False):
        for variable in variables:
            values = group_table[variable].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "dataset_group": group,
                    "variable": variable,
                    "count": len(finite),
                    "invalid_count": int(len(values) - len(finite)),
                    "mean": float(np.mean(finite)) if len(finite) else np.nan,
                    "std": float(np.std(finite)) if len(finite) else np.nan,
                    "p05": float(np.percentile(finite, 5)) if len(finite) else np.nan,
                    "median": float(np.median(finite)) if len(finite) else np.nan,
                    "p95": float(np.percentile(finite, 95)) if len(finite) else np.nan,
                }
            )
    summary = pd.DataFrame(rows)
    return points, summary


def confidence_analysis(real_trace: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_csv(SOURCE / "correspondence_results.csv")
    source = source.loc[
        (source["method"] == "M5 Full contextual graph-UOT")
        & source["scenario_id"].str.startswith("primary_")
    ]
    rows = []
    thresholds = np.linspace(0.0, 1.0, 41)
    for group_name, table in [("Synthetic all", source), *list(source.groupby("difficulty"))]:
        common = table["is_common_cell"].to_numpy(dtype=bool)
        correct = table["top1_correct"].to_numpy(dtype=bool)
        confidence = table["confidence"].to_numpy(dtype=float)
        for threshold in thresholds:
            selected = confidence >= threshold
            predicted_unmatched = ~selected
            tp_unmatched = np.count_nonzero(predicted_unmatched & ~common)
            unmatched_precision = tp_unmatched / max(np.count_nonzero(predicted_unmatched), 1)
            unmatched_recall = tp_unmatched / max(np.count_nonzero(~common), 1)
            unmatched_f1 = (
                2 * unmatched_precision * unmatched_recall / (unmatched_precision + unmatched_recall)
                if unmatched_precision + unmatched_recall > 0
                else 0.0
            )
            rows.append(
                {
                    "record_type": "synthetic_threshold_curve",
                    "dataset_group": str(group_name),
                    "status": None,
                    "component": None,
                    "threshold": threshold,
                    "selected_n": int(np.count_nonzero(selected)),
                    "coverage_all": float(np.mean(selected)),
                    "precision_selected": float(np.mean(correct[selected])) if np.any(selected) else np.nan,
                    "matched_recall": float(np.count_nonzero(selected & correct) / max(np.count_nonzero(common), 1)),
                    "unmatched_f1": unmatched_f1,
                    "value_p05": np.nan,
                    "value_median": np.nan,
                    "value_p95": np.nan,
                }
            )
    real_confidence = real_trace["confidence_saved_algorithm"].to_numpy()
    for threshold in thresholds:
        selected = real_confidence >= threshold
        rows.append(
            {
                "record_type": "real_threshold_coverage_only",
                "dataset_group": "Real",
                "status": None,
                "component": None,
                "threshold": threshold,
                "selected_n": int(np.count_nonzero(selected)),
                "coverage_all": float(np.mean(selected)),
                "precision_selected": np.nan,
                "matched_recall": np.nan,
                "unmatched_f1": np.nan,
                "value_p05": np.nan,
                "value_median": np.nan,
                "value_p95": np.nan,
            }
        )
    components = [
        "confidence_mass_contribution",
        "confidence_concentration_contribution",
        "confidence_margin_contribution",
        "confidence_graph_contribution",
        "confidence_region_contribution",
        "confidence_saved_algorithm",
    ]
    for status, table in real_trace.groupby("status"):
        for component in components:
            values = table[component].to_numpy(dtype=float)
            rows.append(
                {
                    "record_type": "real_component_distribution",
                    "dataset_group": "Real",
                    "status": status,
                    "component": component,
                    "threshold": np.nan,
                    "selected_n": len(values),
                    "coverage_all": len(values) / len(real_trace),
                    "precision_selected": np.nan,
                    "matched_recall": np.nan,
                    "unmatched_f1": np.nan,
                    "value_p05": float(np.percentile(values, 5)),
                    "value_median": float(np.median(values)),
                    "value_p95": float(np.percentile(values, 95)),
                }
            )
    bins = np.linspace(0.0, 1.0, 11)
    bin_index = np.clip(np.digitize(source["confidence"].to_numpy(dtype=float), bins) - 1, 0, 9)
    for index in range(10):
        selected = bin_index == index
        if not np.any(selected):
            continue
        rows.append(
            {
                "record_type": "synthetic_calibration_bin",
                "dataset_group": "Synthetic all",
                "status": None,
                "component": "top1_correct",
                "threshold": bins[index],
                "selected_n": int(np.count_nonzero(selected)),
                "coverage_all": float(np.mean(selected)),
                "precision_selected": float(np.mean(source.loc[selected, "top1_correct"])),
                "matched_recall": np.nan,
                "unmatched_f1": np.nan,
                "value_p05": float(bins[index]),
                "value_median": float(np.mean(source.loc[selected, "confidence"])),
                "value_p95": float(bins[index + 1]),
            }
        )
    return pd.DataFrame(rows)


def distribution_gaps(points: pd.DataFrame) -> pd.DataFrame:
    variables = [column for column in points.columns if column != "dataset_group"]
    real = points.loc[points["dataset_group"] == "Real"]
    rows = []
    for group in ("Easy", "Medium", "Hard"):
        synthetic = points.loc[points["dataset_group"] == group]
        for variable in variables:
            left = synthetic[variable].to_numpy(dtype=float)
            right = real[variable].to_numpy(dtype=float)
            left = left[np.isfinite(left)]
            right = right[np.isfinite(right)]
            if not len(left) or not len(right):
                continue
            ks = ks_2samp(left, right)
            rows.append(
                {
                    "synthetic_group": group,
                    "variable": variable,
                    "synthetic_median": float(np.median(left)),
                    "real_median": float(np.median(right)),
                    "median_difference_real_minus_synthetic": float(np.median(right) - np.median(left)),
                    "ks_statistic": float(ks.statistic),
                    "ks_pvalue": float(ks.pvalue),
                    "wasserstein_distance": float(wasserstein_distance(left, right)),
                }
            )
    return pd.DataFrame(rows)


def negative_controls(
    fixed: np.ndarray,
    moving: np.ndarray,
    normal_trace: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run fixed-setting controls; these runs are diagnostic and never tune the method."""
    rng = np.random.default_rng(20261008)
    permuted_y = rng.permutation(len(moving))
    controls = {
        "Normal real": None,
        "Translated +180,-150 um": moving + np.asarray([180.0, -150.0]),
        "Y-coordinate shuffled": np.column_stack([moving[:, 0], moving[permuted_y, 1]]),
    }
    traces = []
    for name, control_moving in controls.items():
        if control_moving is None:
            table = normal_trace.copy()
        else:
            print(f"Running negative control: {name}", flush=True)
            table = trace_pipeline(run_frozen_m5(fixed, control_moving), prefix="control_moving")
        table.insert(0, "control", name)
        traces.append(table)
    point_table = pd.concat(traces, ignore_index=True)
    variables = [
        "regional_best_ncc",
        "regional_ncc_margin",
        "top1_fingerprint_similarity",
        "top1_graph_support",
        "minimum_candidate_cost",
        "row_mass_ratio",
        "normalized_top1_mass",
        "top1_top2_mass_margin",
        "normalized_row_entropy",
        "outside_candidate_mass_fraction",
        "confidence_saved_algorithm",
    ]
    normal = point_table.loc[point_table["control"] == "Normal real"]
    rows = []
    for control, table in point_table.groupby("control", sort=False):
        for variable in variables:
            values = table[variable].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            reference = normal[variable].to_numpy(dtype=float)
            reference = reference[np.isfinite(reference)]
            ks = ks_2samp(reference, finite)
            rows.append(
                {
                    "control": control,
                    "variable": variable,
                    "count": len(finite),
                    "p05": float(np.percentile(finite, 5)),
                    "median": float(np.median(finite)),
                    "p95": float(np.percentile(finite, 95)),
                    "ks_vs_normal": float(ks.statistic),
                    "ks_pvalue_vs_normal": float(ks.pvalue),
                    "wasserstein_vs_normal": float(wasserstein_distance(reference, finite)),
                }
            )
        status_counts = table["status"].value_counts()
        for status in ("matched", "ambiguous", "unmatched"):
            rows.append(
                {
                    "control": control,
                    "variable": f"status_count_{status}",
                    "count": int(status_counts.get(status, 0)),
                    "p05": np.nan,
                    "median": np.nan,
                    "p95": np.nan,
                    "ks_vs_normal": np.nan,
                    "ks_pvalue_vs_normal": np.nan,
                    "wasserstein_vs_normal": np.nan,
                }
            )
    return point_table, pd.DataFrame(rows)


def add_diagnostic_phenotypes(
    trace: pd.DataFrame,
    synthetic_points: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float]:
    output = trace.copy()
    hard = synthetic_points.loc[synthetic_points["dataset_group"] == "Hard"]
    high_cost_threshold = float(np.percentile(hard["minimum_candidate_cost"].dropna(), 95))
    low_graph_threshold = float(np.percentile(hard["graph_support_score"].dropna(), 5))
    output["diagnostic_A_no_candidate"] = output["uot_input_candidate_count"] == 0
    output["diagnostic_B_cost_above_synthetic_hard_p95"] = output["minimum_candidate_cost"] > high_cost_threshold
    output["diagnostic_C_low_mass_despite_cost_within_hard_p95"] = (
        (output["minimum_candidate_cost"] <= high_cost_threshold) & (output["row_mass_ratio"] < 0.12)
    )
    output["diagnostic_D_mass_diffuse"] = (
        (output["row_mass_ratio"] >= 0.12)
        & ((output["normalized_top1_mass"] < 0.16) | (output["top1_top2_mass_margin"] < 0.01))
    )
    output["diagnostic_E_clear_top1_but_confidence_below_0.75"] = (
        (output["normalized_top1_mass"] >= 0.16)
        & (output["top1_top2_mass_margin"] >= 0.01)
        & (output["confidence_saved_algorithm"] < 0.75)
    )
    output["diagnostic_F_confidence_at_least_0.75_but_rejected"] = (
        (output["confidence_saved_algorithm"] >= 0.75) & (output["status"] != "matched")
    )
    output["diagnostic_graph_below_synthetic_hard_p05"] = output["top1_graph_support"] < low_graph_threshold
    return output, high_cost_threshold, low_graph_threshold


def classification_reasons(
    trace: pd.DataFrame,
    high_cost_threshold: float,
    low_graph_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unmatched = trace.loc[trace["status"] == "unmatched"]
    n = len(unmatched)
    rows = []
    actual = [
        ("Candidateなし", 0, "The implementation always supplies 30 candidates."),
        ("Cost不適合", 0, "There is no explicit final cost rejection rule."),
        ("UOT mass不足", int((unmatched["row_mass_ratio"] < 0.12).sum()), "Actual unmatched rule: row mass ratio < 0.12."),
        ("候補が曖昧", 0, "Concentration and margin do not assign the unmatched label."),
        ("Graph support不足", 0, "Graph support changes cost but is not a direct final rule."),
        ("Reciprocal条件不成立", 0, "Reciprocity is recorded only."),
        ("Confidence threshold不達", 0, "Final confidence is not used by the classifier."),
        ("その他", 0, "No other unmatched path was observed."),
    ]
    for cause, count, note in actual:
        rows.append(
            {
                "scope": "exclusive_actual_unmatched_rule",
                "cause": cause,
                "nuclei": count,
                "fraction_of_unmatched": count / max(n, 1),
                "definition_or_note": note,
            }
        )
    overlapping = [
        ("Candidateなし", unmatched["uot_input_candidate_count"] == 0, "candidate_count == 0"),
        (
            "CostがSynthetic Hard p95超",
            unmatched["minimum_candidate_cost"] > high_cost_threshold,
            f"Comparative diagnostic only: minimum cost > {high_cost_threshold:.6f}",
        ),
        ("UOT mass不足", unmatched["row_mass_ratio"] < 0.12, "row_mass_ratio < 0.12"),
        (
            "行内候補が曖昧",
            (unmatched["normalized_top1_mass"] < 0.16) | (unmatched["top1_top2_mass_margin"] < 0.01),
            "normalized top1 < 0.16 or top1-top2 margin < 0.01",
        ),
        (
            "Graph supportがSynthetic Hard p05未満",
            unmatched["top1_graph_support"] < low_graph_threshold,
            f"Comparative diagnostic only: graph support < {low_graph_threshold:.6f}",
        ),
        ("Reciprocalでない", ~unmatched["reciprocal"], "Recorded diagnostic; not a classifier condition"),
        (
            "Confidence < 0.75",
            unmatched["confidence_saved_algorithm"] < 0.75,
            "Diagnostic reference only; 0.75 is not a final classifier threshold",
        ),
    ]
    for cause, mask, note in overlapping:
        count = int(np.count_nonzero(mask))
        rows.append(
            {
                "scope": "overlapping_unmatched_diagnostic_flag",
                "cause": cause,
                "nuclei": count,
                "fraction_of_unmatched": count / max(n, 1),
                "definition_or_note": note,
            }
        )

    group_rows = []
    variables = [
        "uot_input_candidate_count",
        "top1_spatial_distance_um",
        "regional_best_ncc",
        "top1_fingerprint_similarity",
        "top1_graph_support",
        "row_transport_mass",
        "row_mass_ratio",
        "normalized_top1_mass",
        "top1_top2_mass_margin",
        "normalized_row_entropy",
        "confidence_saved_algorithm",
        "minimum_candidate_cost",
    ]
    for status, table in trace.groupby("status"):
        for variable in variables:
            values = table[variable].to_numpy(dtype=float)
            group_rows.append(
                {
                    "status": status,
                    "variable": variable,
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "p05": float(np.percentile(values, 5)),
                    "median": float(np.median(values)),
                    "p95": float(np.percentile(values, 95)),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(group_rows)


def select_individual_examples(trace: pd.DataFrame) -> pd.DataFrame:
    selected = []

    def add_group(name: str, table: pd.DataFrame) -> None:
        take = table.head(20).copy()
        take.insert(0, "example_group", name)
        selected.append(take)

    add_group("Matched", trace.loc[trace["status"] == "matched"].sort_values("confidence_saved_algorithm", ascending=False))
    add_group("Ambiguous", trace.loc[trace["status"] == "ambiguous"].sort_values("confidence_saved_algorithm", ascending=False))
    add_group(
        "Unmatched high top-1 score",
        trace.loc[trace["status"] == "unmatched"].sort_values(
            ["normalized_top1_mass", "minimum_candidate_cost"], ascending=[False, True]
        ),
    )
    add_group(
        "Unmatched no candidate",
        trace.loc[(trace["status"] == "unmatched") & (trace["uot_input_candidate_count"] == 0)],
    )
    add_group(
        "Unmatched small mass",
        trace.loc[trace["status"] == "unmatched"].sort_values("row_mass_ratio"),
    )
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def _summary_lookup(summary: pd.DataFrame, group: str, variable: str, statistic: str = "median") -> float:
    row = summary.loc[(summary["dataset_group"] == group) & (summary["variable"] == variable)]
    return float(row.iloc[0][statistic])


def _control_lookup(summary: pd.DataFrame, control: str, variable: str, statistic: str = "median") -> float:
    row = summary.loc[(summary["control"] == control) & (summary["variable"] == variable)]
    return float(row.iloc[0][statistic])


def make_report(
    trace: pd.DataFrame,
    distribution: pd.DataFrame,
    confidence: pd.DataFrame,
    controls: pd.DataFrame,
    high_cost_threshold: float,
) -> str:
    source_metrics = pd.read_csv(SOURCE / "method_metrics.csv")
    m0 = source_metrics.loc[source_metrics["short_method"] == "M0"].iloc[0]
    m5 = source_metrics.loc[source_metrics["short_method"] == "M5"].iloc[0]
    counts = trace["status"].value_counts()
    matched = int(counts.get("matched", 0))
    ambiguous = int(counts.get("ambiguous", 0))
    unmatched = int(counts.get("unmatched", 0))
    concentration_pass = int(trace["passes_top1_concentration_0.16"].sum())
    margin_pass = int(trace["passes_mass_margin_0.01"].sum())
    mass025_pass = int(trace["passes_matched_row_mass_0.25"].sum())
    max_confidence = float(trace["confidence_saved_algorithm"].max())
    union = trace["candidate_union_count_before_prune"].to_numpy()
    sharp_unmatched = int(
        (
            (trace["status"] == "unmatched")
            & (trace["normalized_top1_mass"] >= 0.16)
            & (trace["top1_top2_mass_margin"] >= 0.01)
        ).sum()
    )
    high_cost_unmatched = int(
        trace.loc[trace["status"] == "unmatched", "diagnostic_B_cost_above_synthetic_hard_p95"].sum()
    )
    reciprocal_unmatched = int((~trace.loc[trace["status"] == "unmatched", "reciprocal"]).sum())
    candidate_recall = pd.read_csv(SOURCE / "candidate_recall.csv")
    recall = candidate_recall.groupby(["candidate_strategy", "k"])["candidate_recall"].mean()
    real_075 = confidence.loc[
        (confidence["record_type"] == "real_threshold_coverage_only")
        & np.isclose(confidence["threshold"], 0.75)
    ].iloc[0]

    def d(group: str, variable: str) -> float:
        return _summary_lookup(distribution, group, variable)

    def c(control: str, variable: str) -> float:
        return _control_lookup(controls, control, variable)

    return f"""# Cell Correspondence low matched-rate diagnosis

This is a diagnosis-only audit of the frozen `contextual-graph-uot-20261008` prototype. No algorithm, weight, threshold, production workflow, safety setting, or prior artifact was changed.

## Executive stage table

| Stage | Valid moving nuclei | Candidate median | Main rejection / degradation reason |
|---|---:|---:|---|
| Affine | {len(trace):,} | 1,830 fixed search universe | No rejection |
| Regional NCC | {len(trace):,} | 30 | Ranking/prior only; no moving nucleus deleted |
| Fingerprint | {len(trace):,} | 30 | Ranking only; no moving nucleus deleted |
| Graph | {len(trace):,} | 30 | Cost modification only; no candidate deleted |
| UOT | {len(trace):,} | 30 | {unmatched:,} rows have row mass ratio below 0.12 |
| Confidence classification | {matched} matched | 30 | Status uses row mass/concentration/margin, not final confidence |

## Source-result verification

- Synthetic M0 Spatial-UOT: Top-1 {m0.top1_accuracy:.2%}, Top-3 {m0.top3_recall:.2%}, matched F1 {m0.matched_f1:.3f}, unmatched F1 {m0.unmatched_f1:.3f}.
- Synthetic M5 Full: Top-1 {m5.top1_accuracy:.2%}, Top-3 {m5.top3_recall:.2%}, matched F1 {m5.matched_f1:.3f}, unmatched F1 {m5.unmatched_f1:.3f}.
- Real: {len(trace):,} moving nuclei, {matched} matched ({matched / len(trace):.1%}), {ambiguous} ambiguous ({ambiguous / len(trace):.1%}), {unmatched} unmatched ({unmatched / len(trace):.1%}).
- Real maximum confidence is {max_confidence:.7f}; confidence >=0.75 is {int((trace['confidence_saved_algorithm'] >= 0.75).sum())}.
- The recomputed real result agrees with the saved source CSV exactly for status/indices/reciprocity and within {trace['confidence_recompute_absolute_error'].max():.2e} for confidence.

Synthetic Top-1 accuracy and real matched coverage are different metrics. Real has no cell-correspondence ground truth, so this report makes no real-data accuracy claim.

## Why only 25 matched?

The exact classifier path is:

1. initialize every row as `ambiguous`;
2. assign `unmatched` when `row_mass_ratio < 0.12`;
3. override as `matched` only when row mass ratio >=0.25, normalized top-1 mass >=0.16, and normalized top1-top2 mass margin >=0.01.

On real data, {concentration_pass:,}/{len(trace):,} pass concentration and {margin_pass:,}/{len(trace):,} pass the margin, but only {mass025_pass:,}/{len(trace):,} pass the matched row-mass gate. The intersection is exactly {matched}. The direct bottleneck is therefore **absolute UOT row mass**.

Final confidence is additive, not multiplicative:

`0.30*clip(row_mass_ratio) + 0.25*clip(normalized_top1) + 0.15*clip(margin/0.25) + 0.15*graph_support + 0.15*regional_confidence`.

It is never read by the status code. Thus maximum confidence {max_confidence:.7f} and 25 matched are fully compatible: the 25 assignments pass the three independent status conditions even though none reaches 0.75. Reciprocity is also recorded only; fingerprint, graph, and region influence upstream candidate cost, but are not direct status gates.

### Exact frozen scoring and thresholds

- Candidate generation forms spatial top-30, regional top-30, and fingerprint top-30 (fingerprint is evaluated inside spatial top-80), unions the first 20 from each, and prunes to 30 by base cost. It is top-K rather than radius rejection, so every real row receives candidates.
- Base cost is `0.45*position_cost + 0.85*fingerprint_cost + 0.70*region_cost`, where position and regional residual use 10 um scales. Regional cost also includes `0.25*(1-regional_confidence)`.
- Regional confidence is `clip(0.5*(best_NCC+1) + 2*max(best-second margin,0), 0, 1)`. It affects regional cost and contributes 0.15 to final confidence.
- Graph adds `0.75*(1-support)` to base cost. It changes scores/ranks and therefore UOT mass; it has no hard support threshold.
- Candidate pairs populate the dense UOT matrix; all non-candidate pairs receive cost 25. UOT uses entropy 0.1, KL mass relaxation 0.5, and translation-invariant Sinkhorn.
- Each real source has nominal mass `1/max(1754,1830) = {1 / 1830:.8f}`. `row_mass_ratio` is absolute transported row mass divided by that nominal source mass. Top-1/top-2 normalized masses and entropy describe allocation *within* the row and are not correspondence probabilities.
- Reciprocal status is `moving i == argmax_moving(plan[:, top1_fixed_i])`; it is output only. Fingerprint similarity is output and enters cost through fingerprint cost, but is not separately thresholded.
- In this real run all 1,295 unmatched have ratio <0.12; all 434 ambiguous have ratio in [0.12,0.25); and all 25 matched pass ratio >=0.25 plus concentration >=0.16 and margin >=0.01.

## Stage audit

- All {len(trace):,} moving nuclei have 30 spatial, 30 regional, and 30 fingerprint-ranked candidates.
- Their three top-20 lists have union p05/median/p95 {np.percentile(union, 5):.0f}/{np.median(union):.0f}/{np.percentile(union, 95):.0f}; base-cost pruning retains exactly 30 for every row.
- Regional best-offset norm p05/median/p95 is {np.percentile(trace['regional_best_offset_norm_um'], 5):.2f}/{np.median(trace['regional_best_offset_norm_um']):.2f}/{np.percentile(trace['regional_best_offset_norm_um'], 95):.2f} um. {int(trace['regional_offset_hits_search_boundary'].sum()):,}/{len(trace):,} ({trace['regional_offset_hits_search_boundary'].mean():.1%}) nuclei inherit an offset with x or y at the ±18 um search boundary, a direct search-window saturation warning. Regional rank-1 differs from spatial rank-1 for {int(trace['regional_rank1_differs_from_spatial'].sum()):,}/{len(trace):,} rows; fingerprint rank-1 differs for {int(trace['fingerprint_rank1_differs_from_spatial'].sum()):,}/{len(trace):,}.
- Contextual base cost changes spatial rank-1 for {int(trace['base_rank1_differs_from_spatial'].sum()):,} rows; graph cost changes the base-cost minimum for {int(trace['graph_rank1_differs_from_base'].sum()):,}; global UOT top-1 differs from the minimum graph-augmented local cost for {int(trace['uot_rank1_differs_from_graph_min_cost'].sum()):,}. These are re-ranking counts, not evidence of correctness.
- Candidate-zero, NaN, and Inf counts are all zero.
- Graph calculates soft support for all retained candidates and adds `0.75*(1-support)` to cost. It does not prune.
- Real row-mass-ratio p05/median/p95 is {np.percentile(trace['row_mass_ratio'], 5):.4f}/{np.median(trace['row_mass_ratio']):.4f}/{np.percentile(trace['row_mass_ratio'], 95):.4f}.
- {sharp_unmatched:,}/{unmatched:,} unmatched rows nevertheless pass both within-row concentration and margin conditions. A sharp row-normalized peak is not the same as large absolute mass or a calibrated correspondence probability.

## Is candidate discovery working?

**PARTIALLY.** Mechanically it works: every real nucleus reaches UOT with 30 candidates and all index mappings are valid. However, {int(trace['regional_offset_hits_search_boundary'].sum()):,} regional offsets hit the search boundary, so the regional prior is frequently search-window-limited. Real scientific candidate recall is still **unknown**, because no true IDs exist. Synthetic mean candidate recall@10/20/30 is:

| Source | @10 | @20 | @30 |
|---|---:|---:|---:|
| Affine spatial | {recall.loc[('Affine spatial', 10)]:.2%} | {recall.loc[('Affine spatial', 20)]:.2%} | {recall.loc[('Affine spatial', 30)]:.2%} |
| Regional NCC | {recall.loc[('Regional NCC', 10)]:.2%} | {recall.loc[('Regional NCC', 20)]:.2%} | {recall.loc[('Regional NCC', 30)]:.2%} |
| Fingerprint | {recall.loc[('Fingerprint', 10)]:.2%} | {recall.loc[('Fingerprint', 20)]:.2%} | {recall.loc[('Fingerprint', 30)]:.2%} |
| Combined | {recall.loc[('Combined', 10)]:.2%} | {recall.loc[('Combined', 20)]:.2%} | {recall.loc[('Combined', 30)]:.2%} |

Near-perfect synthetic affine recall shows that the synthetic candidate problem is easier than the unknown real one.

## Is UOT discarding plausible matches?

**UNKNOWN scientifically; YES as the direct status gate.** All {unmatched:,} unmatched rows are labeled so because row mass ratio is below 0.12. Without real truth or expert labels, sharp rejected candidates cannot be called plausible true matches. {high_cost_unmatched:,}/{unmatched:,} unmatched rows also have minimum contextual cost above the Synthetic Hard p95 ({high_cost_threshold:.4f}); this comparison is diagnostic only.

The requested A–F transport phenotypes are separated as follows (flags can overlap):

| Phenotype | Real count | Interpretation |
|---|---:|---|
| A. No candidate | {int(trace['diagnostic_A_no_candidate'].sum())} | Not observed |
| B. Cost above Synthetic Hard p95 | {int(trace['diagnostic_B_cost_above_synthetic_hard_p95'].sum())} | Comparative high-cost flag, not a formal rejection |
| C. Low mass although cost is within Hard p95 | {int(trace['diagnostic_C_low_mass_despite_cost_within_hard_p95'].sum())} | Low absolute mass without the comparative high-cost flag |
| D. Mass present but diffuse | {int(((trace['normalized_top1_mass'] < 0.16) | (trace['top1_top2_mass_margin'] < 0.01)).sum())} | Within-row concentration/margin failure |
| E. Clear top-1 but confidence <0.75 | {int(trace['diagnostic_E_clear_top1_but_confidence_below_0.75'].sum())} | Diagnostic overlap; 0.75 is not a status threshold |
| F. Confidence >=0.75 but rejected | {int(trace['diagnostic_F_confidence_at_least_0.75_but_rejected'].sum())} | Not observed |

## Is confidence overly conservative?

**NO as an explanation for 25 matched.** Confidence is not a classifier input. A hypothetical confidence >=0.75 filter covers {int(real_075.selected_n)} real nuclei ({real_075.coverage_all:.2%}), but real precision cannot be inferred. Synthetic calibration/precision-coverage results are in `confidence_analysis.csv`; source M5 ECE is {m5.ece:.4f}.

## Synthetic-Real gap

| Median feature | Easy | Medium | Hard | Real |
|---|---:|---:|---:|---:|
| Nuclear density (/um2) | {d('Easy', 'moving_nuclear_density_per_um2'):.4f} | {d('Medium', 'moving_nuclear_density_per_um2'):.4f} | {d('Hard', 'moving_nuclear_density_per_um2'):.4f} | {d('Real', 'moving_nuclear_density_per_um2'):.4f} |
| kNN distance (um) | {d('Easy', 'moving_knn_distance_um'):.3f} | {d('Medium', 'moving_knn_distance_um'):.3f} | {d('Hard', 'moving_knn_distance_um'):.3f} | {d('Real', 'moving_knn_distance_um'):.3f} |
| Top-1 spatial distance (um) | {d('Easy', 'affine_candidate_distance_um'):.3f} | {d('Medium', 'affine_candidate_distance_um'):.3f} | {d('Hard', 'affine_candidate_distance_um'):.3f} | {d('Real', 'affine_candidate_distance_um'):.3f} |
| Regional NCC | {d('Easy', 'regional_ncc'):.3f} | {d('Medium', 'regional_ncc'):.3f} | {d('Hard', 'regional_ncc'):.3f} | {d('Real', 'regional_ncc'):.3f} |
| Fingerprint similarity | {d('Easy', 'fingerprint_similarity'):.3f} | {d('Medium', 'fingerprint_similarity'):.3f} | {d('Hard', 'fingerprint_similarity'):.3f} | {d('Real', 'fingerprint_similarity'):.3f} |
| Graph support | {d('Easy', 'graph_support_score'):.3f} | {d('Medium', 'graph_support_score'):.3f} | {d('Hard', 'graph_support_score'):.3f} | {d('Real', 'graph_support_score'):.3f} |
| Row mass ratio | {d('Easy', 'row_mass_ratio'):.3f} | {d('Medium', 'row_mass_ratio'):.3f} | {d('Hard', 'row_mass_ratio'):.3f} | {d('Real', 'row_mass_ratio'):.3f} |
| Row entropy | {d('Easy', 'normalized_row_entropy'):.3f} | {d('Medium', 'normalized_row_entropy'):.3f} | {d('Hard', 'normalized_row_entropy'):.3f} | {d('Real', 'normalized_row_entropy'):.3f} |
| Confidence | {d('Easy', 'confidence'):.3f} | {d('Medium', 'confidence'):.3f} | {d('Hard', 'confidence'):.3f} | {d('Real', 'confidence'):.3f} |

The largest practical shifts are much larger real top-candidate distance, lower NCC and graph support, and lower absolute UOT row mass. Real entropy is lower, not higher: the characteristic failure is often small total mass concentrated on one candidate, not broad mass diffusion. Distribution shift does not prove real correspondence error.

## Negative controls

The translation control destroys cross-cloud spatial/regional alignment while preserving internal point-cloud geometry. The deterministic Y-shuffle preserves coordinate marginals/cardinality but destroys 2-D neighborhoods and regional arrangements.

| Median score | Normal | Translated | Y-shuffled |
|---|---:|---:|---:|
| NCC | {c('Normal real', 'regional_best_ncc'):.3f} | {c('Translated +180,-150 um', 'regional_best_ncc'):.3f} | {c('Y-coordinate shuffled', 'regional_best_ncc'):.3f} |
| Fingerprint similarity | {c('Normal real', 'top1_fingerprint_similarity'):.3f} | {c('Translated +180,-150 um', 'top1_fingerprint_similarity'):.3f} | {c('Y-coordinate shuffled', 'top1_fingerprint_similarity'):.3f} |
| Graph support | {c('Normal real', 'top1_graph_support'):.3f} | {c('Translated +180,-150 um', 'top1_graph_support'):.3f} | {c('Y-coordinate shuffled', 'top1_graph_support'):.3f} |
| UOT confidence | {c('Normal real', 'confidence_saved_algorithm'):.3f} | {c('Translated +180,-150 um', 'confidence_saved_algorithm'):.3f} | {c('Y-coordinate shuffled', 'confidence_saved_algorithm'):.3f} |

The large translation lowers NCC, graph support, row mass, and confidence, so the pipeline does respond when gross cross-cloud alignment is destroyed. The Y-shuffle is more concerning: median NCC increases from {c('Normal real', 'regional_best_ncc'):.3f} to {c('Y-coordinate shuffled', 'regional_best_ncc'):.3f}, graph support changes only {c('Normal real', 'top1_graph_support'):.3f} to {c('Y-coordinate shuffled', 'top1_graph_support'):.3f}, and confidence remains {c('Y-coordinate shuffled', 'confidence_saved_algorithm'):.3f}. Under this density-preserving control, the current scores provide weak discrimination. This supports a descriptor/domain-gap concern, but it still cannot establish correctness without real labels. Full KS/Wasserstein comparisons are in `negative_control_summary.csv`; controls were not used for tuning.

## Actual unmatched reasons

| Cause | Nuclei | Fraction |
|---|---:|---:|
| Candidateなし | 0 | 0.0% |
| Cost不適合 (explicit rule) | 0 | 0.0% |
| UOT mass不足 | {unmatched} | 100.0% |
| 候補が曖昧 (explicit unmatched rule) | 0 | 0.0% |
| Graph support不足 (explicit rule) | 0 | 0.0% |
| Reciprocal条件不成立 (explicit rule) | 0 | 0.0% |
| Confidence threshold不達 (explicit rule) | 0 | 0.0% |
| その他 | 0 | 0.0% |

This table follows the actual exclusive classifier. Overlapping upstream/comparative flags are separate in `classification_reasons.csv`. Non-reciprocity occurs in {reciprocal_unmatched:,}/{unmatched:,} unmatched rows but cannot be called the reject reason.

## Root-cause ranking

1. **C. UOT MASS / COST PROBLEM — direct observed bottleneck.** Absolute row-mass gates explain all unmatched labels and leave only 25 matched; concentration/margin almost always pass.
2. **B. DESCRIPTOR DOMAIN GAP — strong distribution and control evidence.** Real density, NCC, graph, distance, cost, and mass differ materially from synthetic; the Y-shuffle barely changes graph/confidence and even raises NCC. These findings raise concern about real-domain discrimination, but do not establish individual errors.
3. **A. CANDIDATE DISCOVERY / REGIONAL PRIOR — partially degraded, correctness unresolved.** Numerical coverage is complete, but {int(trace['regional_offset_hits_search_boundary'].sum()):,} offsets hit the regional search boundary. True real candidate recall remains unidentifiable.
4. **D. CONFIDENCE / CLASSIFICATION PROBLEM — semantic mismatch.** Classification is mass-gated; the reported composite confidence does not classify. It therefore cannot explain or contradict the count.
5. **F. INTRINSIC AMBIGUITY — possible, unconfirmed.** Point clouds may be insufficient, but low entropy/high within-row peaks do not support simple candidate diffusion as the main observed mechanism.

**E. IMPLEMENTATION / DATA INTEGRITY PROBLEM is not supported:** counts, stable IDs, indices, finite values, confidence reconstruction, saved-result reproduction, and synthetic IDs pass checks.

## Facts, inference, unknowns

- **Observed:** every row has candidates; graph never prunes; all unmatched rows fail the absolute-mass floor; confidence/reciprocity are not status inputs; real score distributions differ from synthetic.
- **Inference:** lower regional/graph evidence and larger distances raise cost and likely contribute to low UOT mass. This follows the cost path but is not evidence of real accuracy.
- **Unknown:** whether any real top-1 candidate is the same cell, whether a rejected sharp candidate is biologically plausible, and real candidate recall/precision.

## Recommended minimal fix (proposal only)

The operational cause is confirmed, but biological validity is not. Before changing a threshold or cost, label a small blinded real correspondence set and measure current candidate recall and precision across row mass/cost/confidence. Only after that evidence should row-mass classification be recalibrated for the real cost distribution. No fix was implemented here.
"""


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    fixed, moving = load_real_points()
    print("Recomputing frozen real M5 pipeline", flush=True)
    state = run_frozen_m5(fixed, moving)
    trace = trace_pipeline(state)
    reproduction = verify_against_saved(trace)
    stage = stage_diagnostics(state, trace)
    points, distribution = distribution_points(trace, fixed, moving)
    gaps = distribution_gaps(points)
    trace, high_cost_threshold, low_graph_threshold = add_diagnostic_phenotypes(trace, points)
    reasons, group_distribution = classification_reasons(trace, high_cost_threshold, low_graph_threshold)
    confidence = confidence_analysis(trace)
    control_points, control_summary = negative_controls(fixed, moving, trace)
    examples = select_individual_examples(trace)

    for column in (
        "regional_best_ncc",
        "top1_fingerprint_similarity",
        "top1_graph_support",
        "minimum_candidate_cost",
        "row_mass_ratio",
        "normalized_top1_mass",
        "normalized_row_entropy",
        "confidence_saved_algorithm",
    ):
        validate_finite(f"trace.{column}", trace[column].to_numpy(dtype=float))
    if len(trace) != len(moving) or not trace["moving_nucleus_id"].is_unique:
        raise AssertionError("Moving ID preservation failed.")
    if not trace["status"].isin(["matched", "ambiguous", "unmatched"]).all():
        raise AssertionError("Final classifications are not exhaustive and exclusive.")

    stage.to_csv(OUT / "stage_diagnostics.csv", index=False)
    trace.to_csv(OUT / "per_nucleus_trace.csv", index=False)
    reasons.to_csv(OUT / "classification_reasons.csv", index=False)
    group_distribution.to_csv(OUT / "classification_group_distributions.csv", index=False)
    distribution.to_csv(OUT / "synthetic_real_distribution.csv", index=False)
    points.to_csv(OUT / "synthetic_real_distribution_points.csv", index=False)
    gaps.to_csv(OUT / "distribution_gap_stats.csv", index=False)
    confidence.to_csv(OUT / "confidence_analysis.csv", index=False)
    control_summary.to_csv(OUT / "negative_control_summary.csv", index=False)
    control_points.to_csv(OUT / "negative_control_per_nucleus.csv", index=False)
    examples.to_csv(OUT / "individual_nucleus_examples.csv", index=False)

    viz.stage_attrition(stage, trace, OUT / "stage_attrition.png")
    viz.synthetic_vs_real_scores(points, OUT / "synthetic_vs_real_scores.png")
    viz.classification_reason_breakdown(reasons, OUT / "classification_reason_breakdown.png")
    viz.uot_mass_vs_confidence(trace, OUT / "uot_mass_vs_confidence.png")
    viz.confidence_coverage(confidence, OUT / "confidence_coverage.png")
    viz.individual_match_examples(fixed, moving, examples, OUT / "individual_match_examples.png")

    source_hashes = {
        filename: hashlib.sha256((SOURCE / filename).read_bytes()).hexdigest()
        for filename in ("real_correspondence_candidates.csv", "correspondence_results.csv", "method_metrics.csv")
    }
    summary = {
        "purpose": "diagnosis_only_no_parameter_or_algorithm_change",
        "source_analysis": str(SOURCE),
        "branch": subprocess.run(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True, capture_output=True, check=True
        ).stdout.strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "elapsed_seconds": time.time() - started,
        "input_counts": {"moving": len(moving), "fixed": len(fixed)},
        "status_counts": {key: int(value) for key, value in trace["status"].value_counts().items()},
        "maximum_confidence": float(trace["confidence_saved_algorithm"].max()),
        "confidence_at_least_0_75": int((trace["confidence_saved_algorithm"] >= 0.75).sum()),
        "candidate_zero_count": int((trace["uot_input_candidate_count"] == 0).sum()),
        "regional_candidate_audit": {
            "search_boundary_hit_count": int(trace["regional_offset_hits_search_boundary"].sum()),
            "search_boundary_hit_fraction": float(trace["regional_offset_hits_search_boundary"].mean()),
            "best_offset_norm_um_p05_median_p95": [
                float(np.percentile(trace["regional_best_offset_norm_um"], 5)),
                float(np.median(trace["regional_best_offset_norm_um"])),
                float(np.percentile(trace["regional_best_offset_norm_um"], 95)),
            ],
        },
        "candidate_rank_change_counts": {
            key: int(trace[key].sum())
            for key in (
                "regional_rank1_differs_from_spatial",
                "fingerprint_rank1_differs_from_spatial",
                "base_rank1_differs_from_spatial",
                "graph_rank1_differs_from_base",
                "uot_rank1_differs_from_graph_min_cost",
            )
        },
        "matched_condition_pass_counts": {
            "row_mass_ratio_at_least_0_25": int(trace["passes_matched_row_mass_0.25"].sum()),
            "normalized_top1_at_least_0_16": int(trace["passes_top1_concentration_0.16"].sum()),
            "margin_at_least_0_01": int(trace["passes_mass_margin_0.01"].sum()),
            "all_three": int((trace["status"] == "matched").sum()),
        },
        "comparative_thresholds_not_used_for_classification": {
            "synthetic_hard_minimum_cost_p95": high_cost_threshold,
            "synthetic_hard_graph_support_p05": low_graph_threshold,
        },
        "uot": {
            "solver_final_error": state.result.solver_final_error,
            "solver_converged": state.result.solver_converged,
            "transported_total_mass": state.result.transported_total_mass,
            "settings": UOT,
        },
        "reproduction": reproduction,
        "source_sha256_after_diagnostics": source_hashes,
        "deterministic_trace_sha256": hashlib.sha256(trace.to_csv(index=False).encode()).hexdigest(),
        "no_nan_or_inf_in_required_trace_columns": True,
        "conclusion": {
            "candidate_discovery_working": "PARTIALLY",
            "uot_discarding_plausible_matches": "UNKNOWN scientifically; direct mass gate YES",
            "confidence_overly_conservative": "NO as explanation of matched count",
            "dominant_category": "C. UOT MASS / COST PROBLEM, with B. DESCRIPTOR DOMAIN GAP upstream",
        },
    }
    save_json(OUT / "diagnostic_summary.json", summary)
    (OUT / "REPORT.md").write_text(
        make_report(trace, distribution, confidence, control_summary, high_cost_threshold),
        encoding="utf-8",
    )
    print(f"Diagnostics complete in {time.time() - started:.1f}s: {OUT}", flush=True)


if __name__ == "__main__":
    main()
