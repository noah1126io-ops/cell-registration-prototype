"""Candidate-fixed ground-cost ablation for contextual graph-UOT."""
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
from scipy.stats import spearmanr


ROOT = Path("/mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu")
SOURCE = ROOT / "analysis/contextual-graph-uot-20261008"
DIAGNOSTICS = ROOT / "analysis/contextual-graph-uot-diagnostics-20261008"
OUT = ROOT / "analysis/contextual-uot-cost-ablation-20261008"
REAL_INPUT = ROOT / "sweeps/real-grid27-20260908-v2/input"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(OUT))

from candidate_generation import build_candidate_rankings, build_candidate_set  # noqa: E402
from config import CANDIDATES, DIFFICULTIES, SEEDS, UOT  # noqa: E402
from contextual_uot import _extract, _run_plan  # noqa: E402
from graph_consistency import graph_support_scores  # noqa: E402
from local_fingerprint import compute_fingerprint_pair, pairwise_fingerprint_cost  # noqa: E402
from regional_pattern import match_regional_patterns  # noqa: E402
from synthetic_correspondence import assert_identity_alignment, generate_correspondence_sample  # noqa: E402
import cost_visualization as viz  # noqa: E402


COSTS = {
    "A0 Position": ("position",),
    "A1 Position+FP": ("position", "fingerprint"),
    "A2 Position+Region": ("position", "region"),
    "A3 Position+Graph": ("position", "graph"),
    "A4 Position+FP+Graph": ("position", "fingerprint", "graph"),
    "A5 Position+FP+Region": ("position", "fingerprint", "region"),
    "A6 Full": ("position", "fingerprint", "region", "graph"),
}


@dataclass
class FixedCandidateContext:
    fixed: np.ndarray
    moving: np.ndarray
    regional: object
    candidates: object
    graph_support: np.ndarray
    components: dict[str, np.ndarray]


@dataclass
class UOTRun:
    condition: str
    selected_cost: np.ndarray
    dense_cost: np.ndarray
    plan: np.ndarray
    log: dict
    result: object
    per_nucleus: pd.DataFrame
    metrics: dict


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
        raise ValueError(f"{name} contains {np.count_nonzero(~np.isfinite(values))} NaN/Inf values")


def load_real_points() -> tuple[np.ndarray, np.ndarray]:
    fixed = np.load(REAL_INPUT / "success_metric_fixed_points.npy", allow_pickle=False)
    moving = np.load(REAL_INPUT / "success_metric_moving_points.npy", allow_pickle=False)
    if fixed.shape != (1830, 2) or moving.shape != (1754, 2):
        raise RuntimeError(f"Unexpected real point shapes: fixed={fixed.shape}, moving={moving.shape}")
    return fixed, moving


def build_fixed_context(fixed: np.ndarray, moving: np.ndarray) -> FixedCandidateContext:
    regional = match_regional_patterns(fixed, moving)
    fingerprints = compute_fingerprint_pair(fixed, moving)
    fingerprint_cost = pairwise_fingerprint_cost(fingerprints)
    rankings = build_candidate_rankings(fixed, moving, fingerprint_cost, regional)
    candidates = build_candidate_set(rankings, fingerprint_cost, regional, include_region=True)
    graph_support = graph_support_scores(fixed, moving, candidates)
    weights = CANDIDATES["weights_full"]
    components = {
        "position": float(weights["position"]) * candidates.position_cost,
        "fingerprint": float(weights["fingerprint"]) * candidates.fingerprint_cost,
        "region": float(weights["region"]) * candidates.region_cost,
        "graph": float(CANDIDATES["graph_weight"]) * (1.0 - graph_support),
    }
    for name, values in components.items():
        validate_finite(name, values)
        if np.any(values < -1e-12):
            raise ValueError(f"{name} has negative cost")
    reconstructed = components["position"] + components["fingerprint"] + components["region"]
    np.testing.assert_allclose(reconstructed, candidates.base_cost, atol=1e-12, rtol=1e-12)
    return FixedCandidateContext(
        fixed=fixed,
        moving=moving,
        regional=regional,
        candidates=candidates,
        graph_support=graph_support,
        components=components,
    )


def candidate_hash(indices: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(indices).view(np.uint8)).hexdigest()


def selected_cost(context: FixedCandidateContext, terms: tuple[str, ...]) -> np.ndarray:
    return np.sum([context.components[name] for name in terms], axis=0)


def dense_candidate_cost(context: FixedCandidateContext, values: np.ndarray) -> np.ndarray:
    dense = np.full(
        (len(context.moving), len(context.fixed)),
        float(CANDIDATES["outside_cost"]),
        dtype=np.float64,
    )
    np.put_along_axis(dense, context.candidates.indices, values, axis=1)
    return dense


def generalized_kl(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if np.any(first < 0) or np.any(second <= 0):
        raise ValueError("Generalized KL requires first >=0 and second >0")
    positive = first > 0
    result = float(np.sum(second - first))
    result += float(np.sum(first[positive] * np.log(first[positive] / second[positive])))
    return result


def pot_logged_kl(first: np.ndarray, second: np.ndarray, eps: float = 1e-16) -> float:
    """Reproduce this vendored POT backend's kl_div(..., mass=False) log term."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    return float(np.sum(first * np.log(first / second + eps)))


def objective_terms(plan: np.ndarray, cost: np.ndarray) -> dict:
    n_moving, n_fixed = plan.shape
    denominator = float(max(n_moving, n_fixed))
    source = np.ones(n_moving, dtype=np.float64) / denominator
    target = np.ones(n_fixed, dtype=np.float64) / denominator
    reference = source[:, None] * target[None, :]
    transport = float(np.sum(plan * cost))
    entropy_weight = float(UOT["entropic_regularization"])
    mass_weight = float(UOT["mass_relaxation_kl"])
    entropic = entropy_weight * generalized_kl(plan, reference)
    source_penalty = mass_weight * generalized_kl(np.sum(plan, axis=1), source)
    target_penalty = mass_weight * generalized_kl(np.sum(plan, axis=0), target)
    logged_entropic = entropy_weight * pot_logged_kl(plan, reference)
    logged_source = mass_weight * pot_logged_kl(np.sum(plan, axis=1), source)
    logged_target = mass_weight * pot_logged_kl(np.sum(plan, axis=0), target)
    return {
        "transport_cost_term": transport,
        "entropic_kl_term": entropic,
        "source_mass_kl_penalty": source_penalty,
        "target_mass_kl_penalty": target_penalty,
        "mass_relaxation_penalty": source_penalty + target_penalty,
        "generalized_kl_objective_diagnostic": transport + entropic + source_penalty + target_penalty,
        "pot_logged_entropic_relative_term": logged_entropic,
        "pot_logged_source_relative_term": logged_source,
        "pot_logged_target_relative_term": logged_target,
        "final_objective_recomputed": transport + logged_entropic + logged_source + logged_target,
        "source_total_mass": float(np.sum(source)),
        "target_total_mass": float(np.sum(target)),
    }


def _slots(candidate_indices: np.ndarray, fixed_indices: np.ndarray) -> np.ndarray:
    matches = candidate_indices == fixed_indices[:, None]
    if not np.all(np.sum(matches, axis=1) == 1):
        raise AssertionError("Fixed index is missing or duplicated in candidate row")
    return np.argmax(matches, axis=1)


def run_uot(
    context: FixedCandidateContext,
    condition: str,
    values: np.ndarray,
    *,
    dense_override: np.ndarray | None = None,
    graph_in_condition: bool,
) -> UOTRun:
    validate_finite(f"{condition} selected cost", values)
    if np.any(values < -1e-12):
        raise ValueError(f"{condition} has negative selected cost")
    dense = dense_candidate_cost(context, values) if dense_override is None else np.asarray(dense_override, dtype=float)
    validate_finite(f"{condition} dense cost", dense)
    if np.any(dense < -1e-12):
        raise ValueError(f"{condition} has negative dense cost")
    plan, log, warnings = _run_plan(dense)
    result = _extract(
        condition,
        plan,
        log,
        context.moving,
        context.fixed,
        context.regional,
        allowed_indices=context.candidates.indices,
        candidates=context.candidates,
        graph_support=context.graph_support if graph_in_condition else None,
    )
    candidate_mass = np.take_along_axis(plan, context.candidates.indices, axis=1)
    local_order = np.argsort(candidate_mass, axis=1)[:, ::-1]
    ranked_indices = np.take_along_axis(context.candidates.indices, local_order, axis=1)
    ranked_mass = np.take_along_axis(candidate_mass, local_order, axis=1)
    ranked_cost = np.take_along_axis(values, local_order, axis=1)
    row_mass = np.sum(plan, axis=1)
    normalized = np.divide(ranked_mass, row_mass[:, None], out=np.zeros_like(ranked_mass), where=row_mass[:, None] > 0)
    cost_order = np.argsort(values, axis=1)
    cost_ranked_indices = np.take_along_axis(context.candidates.indices, cost_order, axis=1)
    sorted_cost = np.take_along_axis(values, cost_order, axis=1)
    outside_mass = np.maximum(row_mass - np.sum(candidate_mass, axis=1), 0.0)
    effective_count = np.exp(result.normalized_row_entropy * np.log(context.candidates.indices.shape[1]))
    per_nucleus = pd.DataFrame(
        {
            "condition": condition,
            "moving_index": np.arange(len(context.moving)),
            "moving_nucleus_id": [f"real_moving_{index:06d}" for index in range(len(context.moving))],
            "candidate_hash": candidate_hash(context.candidates.indices),
            "candidate_count": context.candidates.indices.shape[1],
            "minimum_candidate_cost": sorted_cost[:, 0],
            "second_minimum_candidate_cost": sorted_cost[:, 1],
            "cost_margin": sorted_cost[:, 1] - sorted_cost[:, 0],
            "raw_cost_top1_fixed_index": cost_ranked_indices[:, 0],
            "uot_top1_fixed_index": ranked_indices[:, 0],
            "uot_top2_fixed_index": ranked_indices[:, 1],
            "uot_top1_cost": ranked_cost[:, 0],
            "uot_top2_cost": ranked_cost[:, 1],
            "row_transport_mass": row_mass,
            "row_mass_ratio": result.row_mass_ratio,
            "normalized_top1_mass": normalized[:, 0],
            "normalized_top2_mass": normalized[:, 1],
            "top1_top2_mass_margin": result.top1_top2_margin,
            "normalized_row_entropy": result.normalized_row_entropy,
            "effective_candidate_count": effective_count,
            "outside_candidate_mass_fraction": np.divide(
                outside_mass, row_mass, out=np.zeros_like(row_mass), where=row_mass > 0
            ),
            "reciprocal": result.reciprocal,
            "confidence": result.confidence,
            "status": result.status,
        }
    )
    terms = objective_terms(plan, dense)
    total_mass = float(np.sum(plan))
    errors = np.asarray(log.get("err", []), dtype=float)
    solver_total = float(log.get("total_cost", np.nan))
    objective_error = abs(terms["final_objective_recomputed"] - solver_total)
    if objective_error > 1e-10:
        raise AssertionError(f"Objective decomposition mismatch for {condition}: {objective_error}")
    if per_nucleus["minimum_candidate_cost"].nunique() > 1:
        correlation = spearmanr(per_nucleus["minimum_candidate_cost"], per_nucleus["row_mass_ratio"])
        correlation_statistic = float(correlation.statistic)
        correlation_pvalue = float(correlation.pvalue)
    else:
        correlation_statistic = np.nan
        correlation_pvalue = np.nan
    status_counts = per_nucleus["status"].value_counts()
    metrics = {
        "condition": condition,
        "candidate_hash": candidate_hash(context.candidates.indices),
        "candidate_count_per_row": context.candidates.indices.shape[1],
        "total_transported_mass": total_mass,
        "total_mass_ratio_to_source": total_mass / terms["source_total_mass"],
        "total_mass_ratio_to_target": total_mass / terms["target_total_mass"],
        "solver_converged": bool(len(errors) and errors[-1] <= float(UOT["stop_threshold"])),
        "solver_iterations": len(errors),
        "solver_final_error": float(errors[-1]) if len(errors) else np.nan,
        "solver_final_objective": solver_total,
        "objective_decomposition_error": objective_error,
        **terms,
        "solver_warning_count": len(warnings),
        "row_mass_ratio_p05": float(np.percentile(result.row_mass_ratio, 5)),
        "row_mass_ratio_median": float(np.median(result.row_mass_ratio)),
        "row_mass_ratio_p95": float(np.percentile(result.row_mass_ratio, 95)),
        "minimum_candidate_cost_p05": float(np.percentile(per_nucleus["minimum_candidate_cost"], 5)),
        "minimum_candidate_cost_median": float(np.median(per_nucleus["minimum_candidate_cost"])),
        "minimum_candidate_cost_p95": float(np.percentile(per_nucleus["minimum_candidate_cost"], 95)),
        "cost_margin_p05": float(np.percentile(per_nucleus["cost_margin"], 5)),
        "cost_margin_median": float(np.median(per_nucleus["cost_margin"])),
        "cost_margin_p95": float(np.percentile(per_nucleus["cost_margin"], 95)),
        "row_mass_at_least_0_12_n": int(np.count_nonzero(result.row_mass_ratio >= 0.12)),
        "row_mass_at_least_0_25_n": int(np.count_nonzero(result.row_mass_ratio >= 0.25)),
        "matched_n": int(status_counts.get("matched", 0)),
        "ambiguous_n": int(status_counts.get("ambiguous", 0)),
        "unmatched_n": int(status_counts.get("unmatched", 0)),
        "normalized_top1_mass_p05": float(np.percentile(normalized[:, 0], 5)),
        "normalized_top1_mass_median": float(np.median(normalized[:, 0])),
        "normalized_top1_mass_p95": float(np.percentile(normalized[:, 0], 95)),
        "normalized_row_entropy_p05": float(np.percentile(result.normalized_row_entropy, 5)),
        "normalized_row_entropy_median": float(np.median(result.normalized_row_entropy)),
        "normalized_row_entropy_p95": float(np.percentile(result.normalized_row_entropy, 95)),
        "effective_candidate_count_median": float(np.median(effective_count)),
        "minimum_cost_row_mass_spearman": correlation_statistic,
        "minimum_cost_row_mass_spearman_pvalue": correlation_pvalue,
        "outside_candidate_mass_fraction_median": float(np.median(per_nucleus["outside_candidate_mass_fraction"])),
        "outside_candidate_mass_fraction_p95": float(np.percentile(per_nucleus["outside_candidate_mass_fraction"], 95)),
    }
    return UOTRun(condition, values, dense, plan, log, result, per_nucleus, metrics)


def cost_component_tables(context: FixedCandidateContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, summaries = [], []
    n, k = context.candidates.indices.shape
    total = np.sum(list(context.components.values()), axis=0)
    for moving_index in range(n):
        for slot in range(k):
            rows.append(
                {
                    "moving_index": moving_index,
                    "candidate_slot": slot,
                    "fixed_index": int(context.candidates.indices[moving_index, slot]),
                    "position_raw": context.candidates.position_cost[moving_index, slot],
                    "position_contribution": context.components["position"][moving_index, slot],
                    "fingerprint_raw": context.candidates.fingerprint_cost[moving_index, slot],
                    "fingerprint_contribution": context.components["fingerprint"][moving_index, slot],
                    "region_raw": context.candidates.region_cost[moving_index, slot],
                    "region_contribution": context.components["region"][moving_index, slot],
                    "graph_support": context.graph_support[moving_index, slot],
                    "graph_contribution": context.components["graph"][moving_index, slot],
                    "full_cost": total[moving_index, slot],
                }
            )
    table = pd.DataFrame(rows)
    values_by_name = {**context.components, "full": total}
    for name, values in values_by_name.items():
        row_means = np.mean(values, axis=1)
        row_variances = np.var(values, axis=1)
        summaries.append(
            {
                "selection_scope": "all_candidates",
                "component": name,
                "weight": {
                    "position": CANDIDATES["weights_full"]["position"],
                    "fingerprint": CANDIDATES["weights_full"]["fingerprint"],
                    "region": CANDIDATES["weights_full"]["region"],
                    "graph": CANDIDATES["graph_weight"],
                    "full": np.nan,
                }[name],
                "formula": {
                    "position": "0.45 * distance^2 / (2 * 10um^2)",
                    "fingerprint": "0.85 * robust fingerprint distance",
                    "region": "0.70 * (regional residual^2/(2*10um^2) + 0.25*(1-regional confidence))",
                    "graph": "0.75 * (1 - graph support)",
                    "full": "position + fingerprint + region + graph",
                }[name],
                "minimum": float(np.min(values)),
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "maximum": float(np.max(values)),
                "mean": float(np.mean(values)),
                "overall_variance": float(np.var(values)),
                "within_row_variance_mean": float(np.mean(row_variances)),
                "within_row_variance_median": float(np.median(row_variances)),
                "between_moving_row_mean_variance": float(np.var(row_means)),
                "row_mean_median": float(np.median(row_means)),
                "row_std_median": float(np.median(np.std(values, axis=1))),
                "negative_count": int(np.count_nonzero(values < 0)),
                "nan_count": int(np.count_nonzero(np.isnan(values))),
                "inf_count": int(np.count_nonzero(np.isinf(values))),
            }
        )
    return table, pd.DataFrame(summaries)


def selected_component_summary(
    candidate_values: pd.DataFrame,
    full_per_nucleus: pd.DataFrame,
    all_candidate_summary: pd.DataFrame,
) -> pd.DataFrame:
    contribution_columns = {
        "position": "position_contribution",
        "fingerprint": "fingerprint_contribution",
        "region": "region_contribution",
        "graph": "graph_contribution",
        "full": "full_cost",
    }
    rows = []
    scopes = {
        "full_min_candidate_per_row": "raw_cost_top1_fixed_index",
        "full_uot_top1_candidate_per_row": "uot_top1_fixed_index",
    }
    metadata = all_candidate_summary.set_index("component")
    for scope, index_column in scopes.items():
        selected = full_per_nucleus[["moving_index", index_column]].rename(
            columns={index_column: "fixed_index"}
        ).merge(candidate_values, on=["moving_index", "fixed_index"], validate="one_to_one")
        for component, column in contribution_columns.items():
            values = selected[column].to_numpy(dtype=float)
            rows.append(
                {
                    "selection_scope": scope,
                    "component": component,
                    "weight": metadata.loc[component, "weight"],
                    "formula": metadata.loc[component, "formula"],
                    "minimum": float(np.min(values)),
                    "median": float(np.median(values)),
                    "p95": float(np.percentile(values, 95)),
                    "maximum": float(np.max(values)),
                    "mean": float(np.mean(values)),
                    "overall_variance": float(np.var(values)),
                    "within_row_variance_mean": np.nan,
                    "within_row_variance_median": np.nan,
                    "between_moving_row_mean_variance": float(np.var(values)),
                    "row_mean_median": float(np.median(values)),
                    "row_std_median": 0.0,
                    "negative_count": int(np.count_nonzero(values < 0)),
                    "nan_count": int(np.count_nonzero(np.isnan(values))),
                    "inf_count": int(np.count_nonzero(np.isinf(values))),
                }
            )
    return pd.DataFrame(rows)


def graph_paired_table(
    first: UOTRun,
    second: UOTRun,
    *,
    dataset: str,
    scenario_id: str,
    true_fixed_index: np.ndarray | None = None,
) -> pd.DataFrame:
    left = first.per_nucleus
    right = second.per_nucleus
    rank_changed = left["uot_top1_fixed_index"].to_numpy() != right["uot_top1_fixed_index"].to_numpy()
    raw_rank_changed = left["raw_cost_top1_fixed_index"].to_numpy() != right["raw_cost_top1_fixed_index"].to_numpy()
    mass_delta = right["row_mass_ratio"].to_numpy() - left["row_mass_ratio"].to_numpy()
    patterns = np.where(
        rank_changed,
        np.where(mass_delta >= 0, "rank_changed_mass_increased", "rank_changed_mass_decreased"),
        np.where(mass_delta >= 0, "rank_unchanged_mass_increased", "rank_unchanged_mass_decreased"),
    )
    output = pd.DataFrame(
        {
            "dataset": dataset,
            "scenario_id": scenario_id,
            "moving_index": left["moving_index"],
            "a1_uot_top1_fixed_index": left["uot_top1_fixed_index"],
            "a4_uot_top1_fixed_index": right["uot_top1_fixed_index"],
            "uot_top1_changed": rank_changed,
            "raw_cost_top1_changed": raw_rank_changed,
            "minimum_cost_delta_a4_minus_a1": right["minimum_candidate_cost"] - left["minimum_candidate_cost"],
            "cost_margin_delta_a4_minus_a1": right["cost_margin"] - left["cost_margin"],
            "row_mass_ratio_a1": left["row_mass_ratio"],
            "row_mass_ratio_a4": right["row_mass_ratio"],
            "row_mass_delta_a4_minus_a1": mass_delta,
            "mass_increased": mass_delta >= 0,
            "a1_status": left["status"],
            "a4_status": right["status"],
            "status_changed": left["status"].to_numpy() != right["status"].to_numpy(),
            "neutral_real_pattern": patterns,
        }
    )
    if true_fixed_index is not None:
        truth = np.asarray(true_fixed_index, dtype=int)
        common = truth >= 0
        before = common & (left["uot_top1_fixed_index"].to_numpy() == truth)
        after = common & (right["uot_top1_fixed_index"].to_numpy() == truth)
        raw_before = common & (left["raw_cost_top1_fixed_index"].to_numpy() == truth)
        raw_after = common & (right["raw_cost_top1_fixed_index"].to_numpy() == truth)
        output["is_common_cell"] = common
        output["a1_uot_top1_correct"] = before
        output["a4_uot_top1_correct"] = after
        output["uot_identity_effect"] = np.select(
            [~common, ~before & after, before & ~after, before & after],
            ["unmatched_truth", "improved", "worsened", "remained_correct"],
            default="remained_incorrect",
        )
        output["raw_cost_identity_effect"] = np.select(
            [~common, ~raw_before & raw_after, raw_before & ~raw_after, raw_before & raw_after],
            ["unmatched_truth", "improved", "worsened", "remained_correct"],
            default="remained_incorrect",
        )
    return output


def binary_metrics(common: np.ndarray, correct: np.ndarray, status: np.ndarray) -> dict:
    predicted_matched = status == "matched"
    predicted_unmatched = status == "unmatched"
    match_tp = predicted_matched & correct
    match_precision = np.count_nonzero(match_tp) / max(np.count_nonzero(predicted_matched), 1)
    match_recall = np.count_nonzero(match_tp) / max(np.count_nonzero(common), 1)
    match_f1 = 2 * match_precision * match_recall / max(match_precision + match_recall, 1e-15)
    unmatched_truth = ~common
    unmatched_tp = predicted_unmatched & unmatched_truth
    unmatched_precision = np.count_nonzero(unmatched_tp) / max(np.count_nonzero(predicted_unmatched), 1)
    unmatched_recall = np.count_nonzero(unmatched_tp) / max(np.count_nonzero(unmatched_truth), 1)
    unmatched_f1 = 2 * unmatched_precision * unmatched_recall / max(unmatched_precision + unmatched_recall, 1e-15)
    return {
        "matched_precision": float(match_precision),
        "matched_recall": float(match_recall),
        "matched_f1": float(match_f1),
        "unmatched_precision": float(unmatched_precision),
        "unmatched_recall": float(unmatched_recall),
        "unmatched_f1": float(unmatched_f1),
    }


def aggregate_synthetic_nuclei(table: pd.DataFrame, group: str) -> dict:
    common = table["is_common_cell"].to_numpy(dtype=bool)
    uot_correct = table["uot_top1_correct"].to_numpy(dtype=bool)
    raw_correct = table["raw_cost_top1_correct"].to_numpy(dtype=bool)
    status = table["status"].to_numpy(dtype=str)
    metrics = binary_metrics(common, uot_correct, status)
    candidate_true = table.loc[common, "true_in_fixed_candidate_set"].to_numpy(dtype=bool)
    return {
        "record_type": "aggregate",
        "difficulty": group,
        "condition": table["condition"].iloc[0],
        "n_rows": len(table),
        "common_n": int(np.count_nonzero(common)),
        "unmatched_n": int(np.count_nonzero(~common)),
        "true_match_candidate_recall": float(np.mean(candidate_true)),
        "raw_cost_top1_accuracy": float(np.mean(raw_correct[common])),
        "uot_top1_accuracy": float(np.mean(uot_correct[common])),
        **metrics,
        "matched_n": int(np.count_nonzero(status == "matched")),
        "ambiguous_n": int(np.count_nonzero(status == "ambiguous")),
        "predicted_unmatched_n": int(np.count_nonzero(status == "unmatched")),
        "matched_coverage_all": float(np.mean(status == "matched")),
        "wrong_matched_count": int(np.count_nonzero((status == "matched") & ~uot_correct)),
        "correct_top1_rejected_by_mass_status_count": int(np.count_nonzero(uot_correct & (status != "matched"))),
        "total_transported_mass_sum_cases": float(table.groupby("scenario_id")["case_total_transported_mass"].first().sum()),
        "case_total_transported_mass_mean": float(table.groupby("scenario_id")["case_total_transported_mass"].first().mean()),
        "row_mass_ratio_p05": float(np.percentile(table["row_mass_ratio"], 5)),
        "row_mass_ratio_median": float(np.median(table["row_mass_ratio"])),
        "row_mass_ratio_p95": float(np.percentile(table["row_mass_ratio"], 95)),
    }


def synthetic_nucleus_table(
    sample,
    scenario_id: str,
    run: UOTRun,
    context: FixedCandidateContext,
) -> pd.DataFrame:
    table = run.per_nucleus.copy()
    table["moving_nucleus_id"] = [f"{scenario_id}_moving_{index:06d}" for index in range(len(table))]
    table["scenario_id"] = scenario_id
    table["difficulty"] = sample.difficulty
    table["seed"] = sample.seed
    table["moving_cell_id"] = sample.moving_ids
    table["true_fixed_index"] = sample.true_fixed_index
    table["is_common_cell"] = sample.true_fixed_index >= 0
    table["true_in_fixed_candidate_set"] = np.any(
        context.candidates.indices == sample.true_fixed_index[:, None], axis=1
    ) & table["is_common_cell"]
    table["raw_cost_top1_correct"] = table["is_common_cell"] & (
        table["raw_cost_top1_fixed_index"] == table["true_fixed_index"]
    )
    table["uot_top1_correct"] = table["is_common_cell"] & (
        table["uot_top1_fixed_index"] == table["true_fixed_index"]
    )
    table["case_total_transported_mass"] = run.metrics["total_transported_mass"]
    table["case_total_mass_ratio_to_source"] = run.metrics["total_mass_ratio_to_source"]
    table["solver_converged"] = run.metrics["solver_converged"]
    return table


def run_synthetic() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    nucleus_frames, paired_frames, case_rows = [], [], []
    candidate_hashes = {}
    for difficulty, settings in DIFFICULTIES.items():
        for seed in SEEDS:
            scenario_id = f"primary_{difficulty.lower()}_{seed}"
            print(f"Synthetic {scenario_id}", flush=True)
            sample = generate_correspondence_sample(seed=seed, difficulty=difficulty, **settings)
            assert_identity_alignment(sample)
            context = build_fixed_context(sample.fixed_points, sample.moving_points)
            expected_hash = candidate_hash(context.candidates.indices)
            candidate_hashes[scenario_id] = expected_hash
            runs = {}
            for condition, terms in COSTS.items():
                values = selected_cost(context, terms)
                run = run_uot(
                    context,
                    condition,
                    values,
                    graph_in_condition="graph" in terms,
                )
                if run.metrics["candidate_hash"] != expected_hash:
                    raise AssertionError("Synthetic candidate set changed between ablations")
                runs[condition] = run
                nucleus = synthetic_nucleus_table(sample, scenario_id, run, context)
                nucleus_frames.append(nucleus)
                common = nucleus["is_common_cell"].to_numpy(dtype=bool)
                row = {
                    "record_type": "case",
                    "scenario_id": scenario_id,
                    "difficulty": difficulty,
                    "seed": seed,
                    "condition": condition,
                    "n_rows": len(nucleus),
                    "common_n": int(np.count_nonzero(common)),
                    "unmatched_n": int(np.count_nonzero(~common)),
                    "true_match_candidate_recall": float(nucleus.loc[common, "true_in_fixed_candidate_set"].mean()),
                    "raw_cost_top1_accuracy": float(nucleus.loc[common, "raw_cost_top1_correct"].mean()),
                    "uot_top1_accuracy": float(nucleus.loc[common, "uot_top1_correct"].mean()),
                    **binary_metrics(
                        common,
                        nucleus["uot_top1_correct"].to_numpy(dtype=bool),
                        nucleus["status"].to_numpy(dtype=str),
                    ),
                    "matched_n": int(np.count_nonzero(nucleus["status"] == "matched")),
                    "ambiguous_n": int(np.count_nonzero(nucleus["status"] == "ambiguous")),
                    "predicted_unmatched_n": int(np.count_nonzero(nucleus["status"] == "unmatched")),
                    "matched_coverage_all": float(np.mean(nucleus["status"] == "matched")),
                    "wrong_matched_count": int(np.count_nonzero((nucleus["status"] == "matched") & ~nucleus["uot_top1_correct"])),
                    "correct_top1_rejected_by_mass_status_count": int(np.count_nonzero(nucleus["uot_top1_correct"] & (nucleus["status"] != "matched"))),
                    "total_transported_mass_sum_cases": run.metrics["total_transported_mass"],
                    "case_total_transported_mass_mean": run.metrics["total_transported_mass"],
                    "row_mass_ratio_p05": run.metrics["row_mass_ratio_p05"],
                    "row_mass_ratio_median": run.metrics["row_mass_ratio_median"],
                    "row_mass_ratio_p95": run.metrics["row_mass_ratio_p95"],
                }
                case_rows.append(row)
            paired_frames.append(
                graph_paired_table(
                    runs["A1 Position+FP"],
                    runs["A4 Position+FP+Graph"],
                    dataset="Synthetic",
                    scenario_id=scenario_id,
                    true_fixed_index=sample.true_fixed_index,
                )
            )
    nuclei = pd.concat(nucleus_frames, ignore_index=True)
    cases = pd.DataFrame(case_rows)
    aggregate_rows = []
    for condition, table in nuclei.groupby("condition", sort=False):
        aggregate_rows.append(aggregate_synthetic_nuclei(table, "All"))
        for difficulty, group in table.groupby("difficulty", sort=False):
            aggregate_rows.append(aggregate_synthetic_nuclei(group, difficulty))
    metrics = pd.concat([cases, pd.DataFrame(aggregate_rows)], ignore_index=True, sort=False)
    return metrics, nuclei, pd.concat(paired_frames, ignore_index=True), candidate_hashes


def precision_coverage_table(nuclei: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition, difficulty), table in nuclei.groupby(["condition", "difficulty"], sort=False):
        correct = table["uot_top1_correct"].to_numpy(dtype=bool)
        common = table["is_common_cell"].to_numpy(dtype=bool)
        for score_name in ("confidence", "row_mass_ratio"):
            score = table[score_name].to_numpy(dtype=float)
            for quantile in np.linspace(0, 1, 21):
                threshold = float(np.quantile(score, quantile))
                selected = score >= threshold
                rows.append(
                    {
                        "condition": condition,
                        "difficulty": difficulty,
                        "selection_score": score_name,
                        "threshold_quantile": quantile,
                        "threshold": threshold,
                        "selected_n": int(np.count_nonzero(selected)),
                        "coverage_all": float(np.mean(selected)),
                        "precision": float(np.mean(correct[selected])) if np.any(selected) else np.nan,
                        "common_correct_recall": float(np.count_nonzero(selected & correct) / max(np.count_nonzero(common), 1)),
                    }
                )
    return pd.DataFrame(rows)


def real_cost_scale_diagnostics(context: FixedCandidateContext, full: UOTRun) -> dict[str, UOTRun]:
    full_dense = full.dense_cost
    full_selected = full.selected_cost
    position = context.components["position"]
    global_offset = 1.0
    b1_dense = full_dense + global_offset
    b1_selected = full_selected + global_offset
    row_offsets = np.min(full_selected, axis=1)
    b2_dense = full_dense - row_offsets[:, None]
    b2_selected = full_selected - row_offsets[:, None]
    if np.min(b2_dense) < -1e-12:
        raise AssertionError("Per-row offset produced negative dense cost")
    scale = float(np.median(position) / np.median(full_selected))
    b3_dense = full_dense * scale
    b3_selected = full_selected * scale
    definitions = {
        "B0 Original Full": (full_selected, full_dense),
        "B1 Full + global constant 1.0": (b1_selected, b1_dense),
        "B2 Full - per-row candidate minimum": (b2_selected, b2_dense),
        "B3 Full scaled to Position median": (b3_selected, b3_dense),
    }
    runs = {"B0 Original Full": full}
    for condition, (values, dense) in definitions.items():
        if condition == "B0 Original Full":
            continue
        runs[condition] = run_uot(
            context,
            condition,
            values,
            dense_override=dense,
            graph_in_condition=True,
        )
    for run in runs.values():
        if not np.array_equal(np.argsort(run.selected_cost, axis=1), np.argsort(full_selected, axis=1)):
            raise AssertionError(f"Raw candidate ranking changed in {run.condition}")
    return runs


def cost_scale_table(runs: dict[str, UOTRun], context: FixedCandidateContext) -> pd.DataFrame:
    reference = runs["B0 Original Full"]
    rows = []
    for name, run in runs.items():
        row = dict(run.metrics)
        row["condition"] = name
        row["raw_candidate_top1_preserved_fraction"] = float(
            np.mean(
                run.per_nucleus["raw_cost_top1_fixed_index"].to_numpy()
                == reference.per_nucleus["raw_cost_top1_fixed_index"].to_numpy()
            )
        )
        row["uot_top1_preserved_fraction"] = float(
            np.mean(
                run.per_nucleus["uot_top1_fixed_index"].to_numpy()
                == reference.per_nucleus["uot_top1_fixed_index"].to_numpy()
            )
        )
        row["candidate_cost_median"] = float(np.median(run.selected_cost))
        row["candidate_cost_minimum"] = float(np.min(run.selected_cost))
        row["candidate_cost_maximum"] = float(np.max(run.selected_cost))
        row["transformation"] = {
            "B0 Original Full": "unchanged",
            "B1 Full + global constant 1.0": "add +1.0 to every dense cost; all rankings/gaps preserved",
            "B2 Full - per-row candidate minimum": "subtract each row's candidate minimum from its entire dense row; within-row rankings/gaps preserved, inter-row level changed",
            "B3 Full scaled to Position median": "multiply entire dense matrix by median(Position candidate cost)/median(Full candidate cost); rankings preserved, gaps scaled",
        }[name]
        rows.append(row)
    return pd.DataFrame(rows)


def verify_known_results(full: UOTRun) -> dict:
    source = pd.read_csv(SOURCE / "real_correspondence_candidates.csv")
    trace = full.per_nucleus
    exact = {
        "top1_fixed_index": bool(np.array_equal(source["top1_fixed_index"], trace["uot_top1_fixed_index"])),
        "status": bool(np.array_equal(source["status"], trace["status"])),
    }
    errors = {
        "row_mass_ratio": float(np.max(np.abs(source["row_mass_ratio"] - trace["row_mass_ratio"]))),
        "top1_mass": float(np.max(np.abs(source["top1_transport_mass"] - np.take_along_axis(
            full.plan, trace["uot_top1_fixed_index"].to_numpy(dtype=int)[:, None], axis=1
        )[:, 0]))),
        "confidence": float(np.max(np.abs(source["confidence"] - trace["confidence"]))),
    }
    if not all(exact.values()) or max(errors.values()) > 1e-12:
        raise AssertionError(f"Full result reproduction failed: exact={exact}, errors={errors}")
    known = json.loads((DIAGNOSTICS / "diagnostic_summary.json").read_text())
    observed_counts = {key: int(value) for key, value in trace["status"].value_counts().items()}
    if observed_counts != known["status_counts"]:
        raise AssertionError("Known diagnostic status counts were not reproduced")
    return {"exact": exact, "maximum_absolute_errors": errors, "status_counts": observed_counts}


def make_report(
    real_metrics: pd.DataFrame,
    components: pd.DataFrame,
    scale: pd.DataFrame,
    synthetic: pd.DataFrame,
    paired: pd.DataFrame,
    verification: dict,
) -> str:
    r = real_metrics.set_index("condition")
    s = synthetic.loc[(synthetic["record_type"] == "aggregate") & (synthetic["difficulty"] == "All")].set_index("condition")
    graph_real = paired.loc[paired["dataset"] == "Real"]
    graph_synth = paired.loc[(paired["dataset"] == "Synthetic") & paired["is_common_cell"].astype(bool)]
    identity = graph_synth["uot_identity_effect"].value_counts()
    raw_identity = graph_synth["raw_cost_identity_effect"].value_counts()
    patterns = graph_real["neutral_real_pattern"].value_counts()
    c = components.loc[components["selection_scope"] == "all_candidates"].set_index("component")
    c_min = components.loc[components["selection_scope"] == "full_min_candidate_per_row"].set_index("component")
    b = scale.set_index("condition")
    a1, a4 = s.loc["A1 Position+FP"], s.loc["A4 Position+FP+Graph"]
    return f"""# Experiment A — Graph-UOT cost decomposition and mass suppression

This is a candidate-fixed controlled ablation. Every A0--A6 run uses the exact Full-method 30-candidate indices. It is therefore not the original unrestricted Spatial-UOT baseline.

## Main result

| Cost | Total mass | Row mass median | Matched | Ambiguous | Unmatched | Synthetic Top-1 | Synthetic Matched F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Position | {r.loc['A0 Position','total_transported_mass']:.4f} | {r.loc['A0 Position','row_mass_ratio_median']:.4f} | {int(r.loc['A0 Position','matched_n'])} | {int(r.loc['A0 Position','ambiguous_n'])} | {int(r.loc['A0 Position','unmatched_n'])} | {s.loc['A0 Position','uot_top1_accuracy']:.4f} | {s.loc['A0 Position','matched_f1']:.4f} |
| Position+FP | {r.loc['A1 Position+FP','total_transported_mass']:.4f} | {r.loc['A1 Position+FP','row_mass_ratio_median']:.4f} | {int(r.loc['A1 Position+FP','matched_n'])} | {int(r.loc['A1 Position+FP','ambiguous_n'])} | {int(r.loc['A1 Position+FP','unmatched_n'])} | {s.loc['A1 Position+FP','uot_top1_accuracy']:.4f} | {s.loc['A1 Position+FP','matched_f1']:.4f} |
| Position+Region | {r.loc['A2 Position+Region','total_transported_mass']:.4f} | {r.loc['A2 Position+Region','row_mass_ratio_median']:.4f} | {int(r.loc['A2 Position+Region','matched_n'])} | {int(r.loc['A2 Position+Region','ambiguous_n'])} | {int(r.loc['A2 Position+Region','unmatched_n'])} | {s.loc['A2 Position+Region','uot_top1_accuracy']:.4f} | {s.loc['A2 Position+Region','matched_f1']:.4f} |
| Position+Graph | {r.loc['A3 Position+Graph','total_transported_mass']:.4f} | {r.loc['A3 Position+Graph','row_mass_ratio_median']:.4f} | {int(r.loc['A3 Position+Graph','matched_n'])} | {int(r.loc['A3 Position+Graph','ambiguous_n'])} | {int(r.loc['A3 Position+Graph','unmatched_n'])} | {s.loc['A3 Position+Graph','uot_top1_accuracy']:.4f} | {s.loc['A3 Position+Graph','matched_f1']:.4f} |
| Position+FP+Graph | {r.loc['A4 Position+FP+Graph','total_transported_mass']:.4f} | {r.loc['A4 Position+FP+Graph','row_mass_ratio_median']:.4f} | {int(r.loc['A4 Position+FP+Graph','matched_n'])} | {int(r.loc['A4 Position+FP+Graph','ambiguous_n'])} | {int(r.loc['A4 Position+FP+Graph','unmatched_n'])} | {s.loc['A4 Position+FP+Graph','uot_top1_accuracy']:.4f} | {s.loc['A4 Position+FP+Graph','matched_f1']:.4f} |
| Position+FP+Region | {r.loc['A5 Position+FP+Region','total_transported_mass']:.4f} | {r.loc['A5 Position+FP+Region','row_mass_ratio_median']:.4f} | {int(r.loc['A5 Position+FP+Region','matched_n'])} | {int(r.loc['A5 Position+FP+Region','ambiguous_n'])} | {int(r.loc['A5 Position+FP+Region','unmatched_n'])} | {s.loc['A5 Position+FP+Region','uot_top1_accuracy']:.4f} | {s.loc['A5 Position+FP+Region','matched_f1']:.4f} |
| Full | {r.loc['A6 Full','total_transported_mass']:.4f} | {r.loc['A6 Full','row_mass_ratio_median']:.4f} | {int(r.loc['A6 Full','matched_n'])} | {int(r.loc['A6 Full','ambiguous_n'])} | {int(r.loc['A6 Full','unmatched_n'])} | {s.loc['A6 Full','uot_top1_accuracy']:.4f} | {s.loc['A6 Full','matched_f1']:.4f} |

## Frozen implementation

- Position contribution: `0.45 * d^2/(2*10um^2)`.
- Fingerprint contribution: `0.85 * fingerprint_cost`.
- Region contribution: `0.70 * (regional_residual^2/(2*10um^2) + 0.25*(1-regional_confidence))`.
- Graph contribution: `0.75*(1-graph_support)`.
- Non-candidate dense entries retain cost 25 in A0--A6. UOT uses entropy 0.1, marginal KL 0.5, translation-invariant Sinkhorn, tolerance 1e-7, and unchanged status thresholds.
- A6 reproduces the saved real Full result: {verification['status_counts']}; maximum audited numeric error {max(verification['maximum_absolute_errors'].values()):.3e}.

All component values are nonnegative and finite. Across all 30 Real candidates, contribution medians (p95) are position {c.loc['position','median']:.3f} ({c.loc['position','p95']:.3f}), fingerprint {c.loc['fingerprint','median']:.3f} ({c.loc['fingerprint','p95']:.3f}), region {c.loc['region','median']:.3f} ({c.loc['region','p95']:.3f}), and graph {c.loc['graph','median']:.3f} ({c.loc['graph','p95']:.3f}). For each row's minimum-Full-cost candidate, the corresponding medians are {c_min.loc['position','median']:.3f}, {c_min.loc['fingerprint','median']:.3f}, {c_min.loc['region','median']:.3f}, and {c_min.loc['graph','median']:.3f}; their summed Full minimum median is {c_min.loc['full','median']:.3f}.

The graph term is particularly uplift-like on Real: its all-candidate row-mean median is {c.loc['graph','row_mean_median']:.3f}, while its within-row SD median is only {c.loc['graph','row_std_median']:.3f}. It still changes relative ranking, but most candidates receive a similar positive addition.

## Q1. Why is transported mass so low?

**Absolute contextual cost suppresses UOT mass.** A0 Position has total mass {r.loc['A0 Position','total_transported_mass']:.4f}; A6 Full has {r.loc['A6 Full','total_transported_mass']:.4f}. Its median minimum candidate cost changes {r.loc['A0 Position','minimum_candidate_cost_median']:.3f}→{r.loc['A6 Full','minimum_candidate_cost_median']:.3f}, while the median top1-top2 cost gap changes {r.loc['A0 Position','cost_margin_median']:.3f}→{r.loc['A6 Full','cost_margin_median']:.3f}. Thus relative separation grows, but the absolute minimum grows much more. Median row-mass ratio changes {r.loc['A0 Position','row_mass_ratio_median']:.3f}→{r.loc['A6 Full','row_mass_ratio_median']:.3f}; Full minimum-cost/row-mass Spearman is {r.loc['A6 Full','minimum_cost_row_mass_spearman']:.3f}.

Objective decomposition confirms that the solver trades transport cost against KL destruction/creation of marginal mass; details are in `cost_ablation_metrics.csv`. This vendored POT version logs `total_cost` with backend `kl_div(..., mass=False)`, which is near zero at the solution; the report therefore also records a separately named nonnegative generalized-KL objective diagnostic. Row-normalized concentration remains separate from absolute row mass.

## Q2. Does Graph improve identity discrimination?

**Yes on the existing synthetic benchmark.** With fixed candidates, A1→A4 changes UOT Top-1 {a1.uot_top1_accuracy:.4f}→{a4.uot_top1_accuracy:.4f} and raw-cost Top-1 {a1.raw_cost_top1_accuracy:.4f}→{a4.raw_cost_top1_accuracy:.4f}. Across common synthetic cells, UOT identity outcomes are improved={int(identity.get('improved', 0))}, worsened={int(identity.get('worsened', 0))}, remained correct={int(identity.get('remained_correct', 0))}; raw-cost improved={int(raw_identity.get('improved', 0))}, worsened={int(raw_identity.get('worsened', 0))}. This supports discrimination on synthetic truth, not on unlabeled real data.

## Q3. Does Graph suppress UOT mass?

**Yes.** Real A1→A4 total mass changes {r.loc['A1 Position+FP','total_transported_mass']:.4f}→{r.loc['A4 Position+FP+Graph','total_transported_mass']:.4f}; median row mass {r.loc['A1 Position+FP','row_mass_ratio_median']:.3f}→{r.loc['A4 Position+FP+Graph','row_mass_ratio_median']:.3f}. Paired real patterns are rank-changed/mass-increased={int(patterns.get('rank_changed_mass_increased', 0))}, rank-changed/mass-decreased={int(patterns.get('rank_changed_mass_decreased', 0))}, rank-unchanged/mass-increased={int(patterns.get('rank_unchanged_mass_increased', 0))}, rank-unchanged/mass-decreased={int(patterns.get('rank_unchanged_mass_decreased', 0))}. “Rank changed” is deliberately not called an improvement on real data.

## Q4. Are useful matches being rejected by the mass gate?

**Yes on synthetic, to the extent quantified by known IDs.** A4 has Top-1 accuracy {a4.uot_top1_accuracy:.4f} but matched recall {a4.matched_recall:.4f}; {int(a4.correct_top1_rejected_by_mass_status_count)} pooled correct Top-1 assignments are not labeled matched. A1 has corresponding values {a1.uot_top1_accuracy:.4f}, {a1.matched_recall:.4f}, and {int(a1.correct_top1_rejected_by_mass_status_count)}. Importantly, Graph does **not** reduce aggregate matched recall here: A1→A4 recall improves because its discrimination gain outweighs its mass penalty. The Full condition rejects {int(s.loc['A6 Full'].correct_top1_rejected_by_mass_status_count)} correct Top-1 rows and has matched recall {s.loc['A6 Full'].matched_recall:.4f}. Precision, recall, F1, wrong-match counts, and difficulty strata are retained in `synthetic_correspondence_metrics.csv`.

## Q5. Can mass and discrimination be separated?

**Mathematically yes; operational adoption still needs calibration.** B1 adds +1 to every dense cost and preserves every raw ranking/gap, yet total mass changes {b.loc['B0 Original Full','total_transported_mass']:.4f}→{b.loc['B1 Full + global constant 1.0','total_transported_mass']:.4f}. B2 subtracts each row minimum while preserving within-row ranks/gaps and yields {b.loc['B2 Full - per-row candidate minimum','total_transported_mass']:.4f}. B3 rescales all costs while preserving ranks and yields {b.loc['B3 Full scaled to Position median','total_transported_mass']:.4f}. UOT top-1 preservation versus B0 is {b.loc['B1 Full + global constant 1.0','uot_top1_preserved_fraction']:.3f}, {b.loc['B2 Full - per-row candidate minimum','uot_top1_preserved_fraction']:.3f}, and {b.loc['B3 Full scaled to Position median','uot_top1_preserved_fraction']:.3f}; global/row competition can alter the plan even when raw ranks are exact.

## Interpretation

**B. USEFUL DISCRIMINATION WITH MASS PENALTY** is closest. A1→A4 improves synthetic identity and lowers mass, although its net matched recall increases rather than decreases. Therefore the penalty is mathematically real but does not erase Graph's aggregate synthetic benefit. The larger A4→A6 contextual addition shows the clearer adoption loss: Top-1 rises {a4.uot_top1_accuracy:.4f}→{s.loc['A6 Full'].uot_top1_accuracy:.4f}, while correct-but-not-matched rows rise {int(a4.correct_top1_rejected_by_mass_status_count)}→{int(s.loc['A6 Full'].correct_top1_rejected_by_mass_status_count)}.

Real data have no truth, so no real correspondence-accuracy claim is made. Increased mass is not increased correctness, and row-normalized mass is not a match probability.

## Q6. What should change next?

The minimal research change to evaluate next is to decouple **relative candidate discrimination** from **absolute match adoption**: retain contextual relative costs, then calibrate the absolute cost baseline/mass decision on held-out persistent-ID synthetic data plus a small blinded real correspondence set. The B diagnostics are sensitivity analyses, not proposed winning algorithms; no offset/rescaling or threshold is adopted here.

No deformation field was generated. This experiment stops at Candidate → Cost → UOT plan → Mass → correspondence decision. It does not explain prior small warp displacement, which also involved confidence weighting, anchor filtering, smoothing, and safety constraints.
"""


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    fixed, moving = load_real_points()
    print("Building fixed real candidate context", flush=True)
    real_context = build_fixed_context(fixed, moving)
    component_values, component_summary = cost_component_tables(real_context)
    real_runs = {}
    for condition, terms in COSTS.items():
        print(f"Real {condition}", flush=True)
        real_runs[condition] = run_uot(
            real_context,
            condition,
            selected_cost(real_context, terms),
            graph_in_condition="graph" in terms,
        )
    verification = verify_known_results(real_runs["A6 Full"])
    component_summary = pd.concat(
        [
            component_summary,
            selected_component_summary(
                component_values,
                real_runs["A6 Full"].per_nucleus,
                component_summary,
            ),
        ],
        ignore_index=True,
    )
    real_metrics = pd.DataFrame([run.metrics for run in real_runs.values()])
    real_per_nucleus = pd.concat([run.per_nucleus for run in real_runs.values()], ignore_index=True)
    real_pair = graph_paired_table(
        real_runs["A1 Position+FP"],
        real_runs["A4 Position+FP+Graph"],
        dataset="Real",
        scenario_id="serial_section_pair_20260904",
    )

    print("Real absolute-cost sensitivity", flush=True)
    scale_runs = real_cost_scale_diagnostics(real_context, real_runs["A6 Full"])
    scale_metrics = cost_scale_table(scale_runs, real_context)
    scale_per_nucleus = pd.concat(
        [run.per_nucleus.assign(experiment_group="absolute_cost_diagnostic") for run in scale_runs.values()],
        ignore_index=True,
    )

    synthetic_metrics, synthetic_nuclei, synthetic_pair, synthetic_hashes = run_synthetic()
    paired = pd.concat([real_pair, synthetic_pair], ignore_index=True, sort=False)
    precision_coverage = precision_coverage_table(synthetic_nuclei)
    all_per_nucleus = pd.concat(
        [
            real_per_nucleus.assign(experiment_group="cost_ablation"),
            scale_per_nucleus.loc[scale_per_nucleus["condition"] != "B0 Original Full"],
        ],
        ignore_index=True,
        sort=False,
    )

    component_values.to_csv(OUT / "cost_components.csv", index=False)
    component_summary.to_csv(OUT / "cost_component_summary.csv", index=False)
    real_metrics.to_csv(OUT / "cost_ablation_metrics.csv", index=False)
    all_per_nucleus.to_csv(OUT / "per_nucleus_cost_mass.csv", index=False)
    paired.to_csv(OUT / "paired_graph_effect.csv", index=False)
    synthetic_metrics.to_csv(OUT / "synthetic_correspondence_metrics.csv", index=False)
    synthetic_nuclei.to_csv(OUT / "synthetic_per_nucleus.csv", index=False)
    precision_coverage.to_csv(OUT / "synthetic_precision_coverage.csv", index=False)
    scale_metrics.to_csv(OUT / "cost_scale_diagnostics.csv", index=False)

    viz.cost_component_distributions(component_values, OUT / "cost_component_distributions.png")
    viz.transported_mass_by_cost(real_metrics, scale_metrics, OUT / "transported_mass_by_cost.png")
    viz.row_mass_distribution(real_per_nucleus, OUT / "row_mass_distribution.png")
    viz.cost_vs_mass(real_per_nucleus, OUT / "cost_vs_mass.png")
    viz.graph_effect_paired(paired, OUT / "graph_effect_paired.png")
    viz.synthetic_accuracy_vs_mass(synthetic_metrics, OUT / "synthetic_accuracy_vs_mass.png")
    viz.matched_coverage_by_method(real_metrics, synthetic_metrics, OUT / "matched_coverage_by_method.png")

    aggregate = synthetic_metrics.loc[
        (synthetic_metrics["record_type"] == "aggregate") & (synthetic_metrics["difficulty"] == "All")
    ].set_index("condition")
    a1 = aggregate.loc["A1 Position+FP"]
    a4 = aggregate.loc["A4 Position+FP+Graph"]
    graph_improves = bool(a4["uot_top1_accuracy"] > a1["uot_top1_accuracy"])
    graph_reduces_mass = bool(
        real_metrics.set_index("condition").loc["A4 Position+FP+Graph", "total_transported_mass"]
        < real_metrics.set_index("condition").loc["A1 Position+FP", "total_transported_mass"]
    )
    graph_reduces_recall = bool(a4["matched_recall"] < a1["matched_recall"])
    if graph_improves and graph_reduces_mass:
        conclusion = "B. USEFUL DISCRIMINATION WITH MASS PENALTY"
    elif (not graph_improves) and graph_reduces_mass:
        conclusion = "C. DESCRIPTOR FAILURE"
    elif graph_reduces_mass:
        conclusion = "A. COST SCALE SUPPRESSION"
    else:
        conclusion = "E. MIXED / INCONCLUSIVE"
    source_hashes = {
        filename: hashlib.sha256((SOURCE / filename).read_bytes()).hexdigest()
        for filename in ("contextual_uot.py", "candidate_generation.py", "real_correspondence_candidates.csv")
    }
    summary = {
        "purpose": "candidate_fixed_cost_ablation_no_algorithm_change",
        "branch": subprocess.run(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True, capture_output=True, check=True
        ).stdout.strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "elapsed_seconds": time.time() - started,
        "candidate_fixed": True,
        "real_candidate_hash": candidate_hash(real_context.candidates.indices),
        "synthetic_candidate_hash_by_case": synthetic_hashes,
        "frozen_uot": UOT,
        "cost_weights": {
            "position": CANDIDATES["weights_full"]["position"],
            "fingerprint": CANDIDATES["weights_full"]["fingerprint"],
            "region": CANDIDATES["weights_full"]["region"],
            "graph": CANDIDATES["graph_weight"],
            "outside": CANDIDATES["outside_cost"],
        },
        "known_result_verification": verification,
        "graph_hypothesis": {
            "synthetic_uot_top1_a1": float(a1["uot_top1_accuracy"]),
            "synthetic_uot_top1_a4": float(a4["uot_top1_accuracy"]),
            "synthetic_matched_recall_a1": float(a1["matched_recall"]),
            "synthetic_matched_recall_a4": float(a4["matched_recall"]),
            "real_total_mass_a1": float(real_metrics.set_index("condition").loc["A1 Position+FP", "total_transported_mass"]),
            "real_total_mass_a4": float(real_metrics.set_index("condition").loc["A4 Position+FP+Graph", "total_transported_mass"]),
            "graph_improves_synthetic_identity": graph_improves,
            "graph_reduces_real_mass": graph_reduces_mass,
            "graph_reduces_synthetic_matched_recall": graph_reduces_recall,
        },
        "interpretation": conclusion,
        "real_has_ground_truth": False,
        "warping_performed": False,
        "source_sha256_after_experiment": source_hashes,
        "deterministic_real_trace_sha256": hashlib.sha256(real_per_nucleus.to_csv(index=False).encode()).hexdigest(),
    }
    save_json(OUT / "diagnostic_summary.json", summary)
    (OUT / "REPORT.md").write_text(
        make_report(real_metrics, component_summary, scale_metrics, synthetic_metrics, paired, verification),
        encoding="utf-8",
    )
    print(f"Complete in {time.time() - started:.1f}s: {conclusion}", flush=True)


if __name__ == "__main__":
    main()
