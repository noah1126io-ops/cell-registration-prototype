"""Cell-correspondence metrics, candidate recall, calibration, and hypotheses."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from candidate_generation import CandidateRankings, candidate_recall
from contextual_uot import MatchResult
from regional_pattern import RegionalMatch
from synthetic_correspondence import CorrespondenceSample


def _divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _f1(precision: float, recall: float) -> float:
    return float(2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0


def correspondence_table(
    sample: CorrespondenceSample,
    result: MatchResult,
    *,
    scenario_id: str,
) -> pd.DataFrame:
    n = len(sample.moving_points)
    common = sample.true_fixed_index >= 0
    top1_correct = common & (result.top1_index == sample.true_fixed_index)
    top3 = np.any(result.ranked_indices[:, :3] == sample.true_fixed_index[:, None], axis=1) & common
    top5 = np.any(result.ranked_indices[:, :5] == sample.true_fixed_index[:, None], axis=1) & common
    true_ids = np.full(n, None, dtype=object)
    true_ids[common] = sample.fixed_ids[sample.true_fixed_index[common]]
    return pd.DataFrame(
        {
            "scenario_id": scenario_id,
            "difficulty": sample.difficulty,
            "seed": sample.seed,
            "method": result.method,
            "moving_index": np.arange(n),
            "moving_cell_id": sample.moving_ids,
            "true_fixed_index": sample.true_fixed_index,
            "true_fixed_cell_id": true_ids,
            "is_common_cell": common,
            "top1_fixed_index": result.top1_index,
            "top1_fixed_cell_id": sample.fixed_ids[result.top1_index],
            "top2_fixed_index": result.top2_index,
            "top2_fixed_cell_id": sample.fixed_ids[result.top2_index],
            "top1_correct": top1_correct,
            "top3_contains_true": top3,
            "top5_contains_true": top5,
            "top1_transport_mass": result.top1_mass,
            "top2_transport_mass": result.top2_mass,
            "top1_top2_margin": result.top1_top2_margin,
            "normalized_row_entropy": result.normalized_row_entropy,
            "total_transported_mass": result.row_transport_mass,
            "row_mass_ratio": result.row_mass_ratio,
            "reciprocal": result.reciprocal,
            "regional_ncc": result.regional_ncc,
            "regional_ncc_margin": result.regional_margin,
            "fingerprint_similarity": result.fingerprint_similarity,
            "graph_support_score": result.graph_support,
            "affine_candidate_distance_um": result.spatial_distance_um,
            "confidence": result.confidence,
            "status": result.status,
        }
    )


def method_metrics(
    sample: CorrespondenceSample,
    result: MatchResult,
    *,
    scenario_id: str,
) -> dict:
    common = sample.true_fixed_index >= 0
    unmatched = ~common
    top1_correct = common & (result.top1_index == sample.true_fixed_index)
    predicted_matched = result.status == "matched"
    predicted_unmatched = result.status == "unmatched"
    true_positive_match = predicted_matched & top1_correct
    matched_precision = _divide(np.count_nonzero(true_positive_match), np.count_nonzero(predicted_matched))
    matched_recall = _divide(np.count_nonzero(true_positive_match), np.count_nonzero(common))
    unmatched_precision = _divide(np.count_nonzero(predicted_unmatched & unmatched), np.count_nonzero(predicted_unmatched))
    unmatched_recall = _divide(np.count_nonzero(predicted_unmatched & unmatched), np.count_nonzero(unmatched))

    ranks = result.ranked_indices
    top3 = common & np.any(ranks[:, :3] == sample.true_fixed_index[:, None], axis=1)
    top5 = common & np.any(ranks[:, :5] == sample.true_fixed_index[:, None], axis=1)
    correct_target_error = np.full(len(ranks), np.nan)
    if np.any(common):
        correct_target_error[common] = np.linalg.norm(
            sample.fixed_points[result.top1_index[common]]
            - sample.fixed_points[sample.true_fixed_index[common]],
            axis=1,
        )
    high = predicted_matched & (result.confidence >= 0.75)
    return {
        "scenario_id": scenario_id,
        "difficulty": sample.difficulty,
        "seed": sample.seed,
        "method": result.method,
        "moving_n": len(sample.moving_points),
        "fixed_n": len(sample.fixed_points),
        "common_n": int(np.count_nonzero(common)),
        "unmatched_n": int(np.count_nonzero(unmatched)),
        "top1_accuracy": float(np.mean(top1_correct[common])) if np.any(common) else np.nan,
        "top3_recall": float(np.mean(top3[common])) if np.any(common) else np.nan,
        "top5_recall": float(np.mean(top5[common])) if np.any(common) else np.nan,
        "matched_precision": matched_precision,
        "matched_recall": matched_recall,
        "matched_f1": _f1(matched_precision, matched_recall),
        "unmatched_precision": unmatched_precision,
        "unmatched_recall": unmatched_recall,
        "unmatched_f1": _f1(unmatched_precision, unmatched_recall),
        "matched_fraction": float(np.mean(predicted_matched)),
        "ambiguous_fraction": float(np.mean(result.status == "ambiguous")),
        "predicted_unmatched_fraction": float(np.mean(predicted_unmatched)),
        "high_confidence_precision": _divide(np.count_nonzero(high & top1_correct), np.count_nonzero(high)),
        "high_confidence_coverage": _divide(np.count_nonzero(high), np.count_nonzero(common)),
        "correspondence_target_error_um_median": float(np.nanmedian(correct_target_error)),
        "affine_candidate_distance_um_median": float(np.median(result.spatial_distance_um)),
        "solver_final_error": result.solver_final_error,
        "solver_converged": result.solver_converged,
        "transported_total_mass": result.transported_total_mass,
        "finite": bool(
            np.isfinite(result.confidence).all()
            and np.isfinite(result.top1_top2_margin).all()
            and np.isfinite(result.spatial_distance_um).all()
        ),
    }


def candidate_recall_rows(
    sample: CorrespondenceSample,
    rankings: CandidateRankings,
    *,
    scenario_id: str,
) -> list[dict]:
    rows = []
    strategies = {
        "Affine spatial": rankings.spatial,
        "Regional NCC": rankings.regional,
        "Fingerprint": rankings.fingerprint,
        "Combined": rankings.combined,
    }
    for name, ranking in strategies.items():
        for k in (10, 20, 30):
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "difficulty": sample.difficulty,
                    "seed": sample.seed,
                    "candidate_strategy": name,
                    "k": k,
                    "candidate_recall": candidate_recall(ranking, sample.true_fixed_index, k),
                    "common_n": int(np.count_nonzero(sample.common_moving_mask)),
                }
            )
    return rows


def regional_hypothesis_rows(
    sample: CorrespondenceSample,
    regional: RegionalMatch,
    *,
    scenario_id: str,
) -> list[dict]:
    rows = []
    common_indices = np.flatnonzero(sample.common_moving_mask)
    common_tree = cKDTree(sample.moving_points[common_indices])
    tolerance = 7.5
    for anchor_index, anchor in enumerate(regional.anchor_centers):
        local_positions = common_tree.query_ball_point(anchor, r=38.0)
        local_moving = common_indices[np.asarray(local_positions, dtype=int)] if local_positions else np.empty(0, dtype=int)
        if len(local_moving) < 3:
            continue
        true_offsets = (
            sample.fixed_points[sample.true_fixed_index[local_moving]] - sample.moving_points[local_moving]
        )
        true_offset = np.median(true_offsets, axis=0)
        errors = np.linalg.norm(regional.top_offsets[anchor_index] - true_offset, axis=1)
        rows.append(
            {
                "scenario_id": scenario_id,
                "difficulty": sample.difficulty,
                "seed": sample.seed,
                "missing_fraction": sample.missing_fraction,
                "anchor_index": anchor_index,
                "cells_supporting_truth": len(local_moving),
                "best_ncc": regional.best_ncc[anchor_index],
                "second_ncc": regional.second_ncc[anchor_index],
                "best_second_margin": regional.margin[anchor_index],
                "peak_ratio": regional.peak_ratio[anchor_index],
                "offset_error_um": errors[0],
                "correct_region_top1": bool(errors[0] <= tolerance),
                "correct_region_top3": bool(np.any(errors[:3] <= tolerance)),
                "correct_region_top5": bool(np.any(errors[:5] <= tolerance)),
            }
        )
    return rows


def calibration_rows(table: pd.DataFrame, bins: int = 10) -> tuple[list[dict], float]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (table["confidence"] >= lower) & (
            (table["confidence"] <= upper) if index == bins - 1 else (table["confidence"] < upper)
        )
        subset = table.loc[mask]
        if subset.empty:
            continue
        accuracy = float(subset["top1_correct"].mean())
        mean_confidence = float(subset["confidence"].mean())
        fraction = len(subset) / len(table)
        ece += fraction * abs(accuracy - mean_confidence)
        rows.append(
            {
                "method": table["method"].iloc[0],
                "confidence_bin_lower": lower,
                "confidence_bin_upper": upper,
                "count": len(subset),
                "mean_confidence": mean_confidence,
                "empirical_accuracy": accuracy,
                "absolute_calibration_error": abs(accuracy - mean_confidence),
            }
        )
    return rows, float(ece)


def rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    positives, negatives = np.count_nonzero(labels), np.count_nonzero(~labels)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average tied ranks.
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for group in np.flatnonzero(counts > 1):
            members = inverse == group
            ranks[members] = np.mean(ranks[members])
    rank_sum = float(np.sum(ranks[labels]))
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))

