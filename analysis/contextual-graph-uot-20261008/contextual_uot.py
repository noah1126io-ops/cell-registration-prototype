"""Conservative UOT used as a partial-assignment layer over fixed candidates."""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

from candidate_generation import CandidateSet
from config import CANDIDATES, POT_VENDOR, UOT
from regional_pattern import RegionalMatch

sys.path.insert(0, str(POT_VENDOR))
import ot  # noqa: E402


@dataclass(frozen=True)
class MatchResult:
    method: str
    ranked_indices: np.ndarray
    top1_index: np.ndarray
    top2_index: np.ndarray
    top1_mass: np.ndarray
    top2_mass: np.ndarray
    top1_top2_margin: np.ndarray
    normalized_row_entropy: np.ndarray
    row_transport_mass: np.ndarray
    row_mass_ratio: np.ndarray
    reciprocal: np.ndarray
    status: np.ndarray
    confidence: np.ndarray
    regional_ncc: np.ndarray
    regional_margin: np.ndarray
    fingerprint_similarity: np.ndarray
    graph_support: np.ndarray
    spatial_distance_um: np.ndarray
    solver_final_error: float | None
    solver_converged: bool
    transported_total_mass: float | None


def _run_plan(cost: np.ndarray) -> tuple[np.ndarray, dict, list[str]]:
    n_moving, n_fixed = cost.shape
    denominator = float(max(n_moving, n_fixed))
    source_mass = np.ones(n_moving, dtype=np.float64) / denominator
    target_mass = np.ones(n_fixed, dtype=np.float64) / denominator
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan, log = ot.unbalanced.sinkhorn_unbalanced(
            source_mass,
            target_mass,
            np.asarray(cost, dtype=np.float64),
            reg=float(UOT["entropic_regularization"]),
            reg_m=float(UOT["mass_relaxation_kl"]),
            method=UOT["solver"],
            reg_type=UOT["reg_type"],
            numItermax=int(UOT["max_iterations"]),
            stopThr=float(UOT["stop_threshold"]),
            log=True,
        )
    if not np.isfinite(plan).all() or np.any(plan < 0):
        raise RuntimeError("UOT returned a non-finite or negative transport plan.")
    return plan, log, [str(item.message) for item in caught]


def _extract(
    method: str,
    plan: np.ndarray,
    log: dict,
    moving: np.ndarray,
    fixed: np.ndarray,
    regional: RegionalMatch,
    *,
    allowed_indices: np.ndarray | None,
    candidates: CandidateSet | None,
    graph_support: np.ndarray | None,
) -> MatchResult:
    n_moving, n_fixed = plan.shape
    denominator = float(max(n_moving, n_fixed))
    source_mass = 1.0 / denominator
    row_mass = np.sum(plan, axis=1)
    row_ratio = row_mass / source_mass

    if allowed_indices is None:
        ranked = np.argsort(plan, axis=1)[:, ::-1]
    else:
        allowed_mass = np.take_along_axis(plan, allowed_indices, axis=1)
        local_order = np.argsort(allowed_mass, axis=1)[:, ::-1]
        ranked = np.take_along_axis(allowed_indices, local_order, axis=1)
    top1 = ranked[:, 0]
    top2 = ranked[:, 1]
    top1_mass = plan[np.arange(n_moving), top1]
    top2_mass = plan[np.arange(n_moving), top2]
    normalized_top1 = np.divide(top1_mass, row_mass, out=np.zeros_like(top1_mass), where=row_mass > 0)
    normalized_top2 = np.divide(top2_mass, row_mass, out=np.zeros_like(top2_mass), where=row_mass > 0)
    margin = normalized_top1 - normalized_top2

    if allowed_indices is None:
        entropy_values = plan
    else:
        entropy_values = np.take_along_axis(plan, allowed_indices, axis=1)
    probabilities = np.divide(
        entropy_values,
        row_mass[:, None],
        out=np.zeros_like(entropy_values),
        where=row_mass[:, None] > 0,
    )
    entropy = -np.sum(np.where(probabilities > 0, probabilities * np.log(np.maximum(probabilities, 1e-300)), 0), axis=1)
    entropy /= max(float(np.log(probabilities.shape[1])), 1.0)

    column_top = np.argmax(plan, axis=0)
    reciprocal = column_top[top1] == np.arange(n_moving)
    status = np.full(n_moving, "ambiguous", dtype=object)
    status[row_ratio < 0.12] = "unmatched"
    confident = (
        (row_ratio >= float(UOT["minimum_row_mass_ratio"]))
        & (normalized_top1 >= 0.16)
        & (margin >= 0.01)
    )
    status[confident] = "matched"

    if candidates is None:
        fp_similarity = np.full(n_moving, np.nan)
        graph_top = np.full(n_moving, np.nan)
        spatial = np.linalg.norm(fixed[top1] - moving, axis=1)
    else:
        slot_lookup = [
            {int(value): slot for slot, value in enumerate(row)} for row in candidates.indices
        ]
        slots = np.asarray([slot_lookup[row][int(top1[row])] for row in range(n_moving)], dtype=int)
        fp_similarity = candidates.fingerprint_similarity[np.arange(n_moving), slots]
        spatial = candidates.spatial_distance_um[np.arange(n_moving), slots]
        graph_top = (
            graph_support[np.arange(n_moving), slots]
            if graph_support is not None
            else np.full(n_moving, np.nan)
        )

    confidence_components = [
        (0.30, np.clip(row_ratio, 0.0, 1.0)),
        (0.25, np.clip(normalized_top1, 0.0, 1.0)),
        (0.15, np.clip(margin / 0.25, 0.0, 1.0)),
    ]
    if np.isfinite(graph_top).any():
        confidence_components.append((0.15, np.nan_to_num(graph_top)))
    confidence_components.append((0.15, regional.cell_confidence))
    total_weight = sum(weight for weight, _ in confidence_components)
    confidence = sum(weight * values for weight, values in confidence_components) / total_weight

    errors = [float(value) for value in log.get("err", [])]
    final_error = errors[-1] if errors else None
    return MatchResult(
        method=method,
        ranked_indices=ranked,
        top1_index=top1,
        top2_index=top2,
        top1_mass=top1_mass,
        top2_mass=top2_mass,
        top1_top2_margin=margin,
        normalized_row_entropy=entropy,
        row_transport_mass=row_mass,
        row_mass_ratio=row_ratio,
        reciprocal=reciprocal,
        status=status,
        confidence=np.clip(confidence, 0.0, 1.0),
        regional_ncc=regional.best_ncc[regional.cell_anchor_index],
        regional_margin=regional.margin[regional.cell_anchor_index],
        fingerprint_similarity=fp_similarity,
        graph_support=graph_top,
        spatial_distance_um=spatial,
        solver_final_error=final_error,
        solver_converged=bool(final_error is not None and final_error <= float(UOT["stop_threshold"])),
        transported_total_mass=float(np.sum(plan)),
    )


def spatial_only_uot(
    fixed: np.ndarray,
    moving: np.ndarray,
    regional: RegionalMatch,
    method: str = "M0 Spatial-only UOT",
) -> MatchResult:
    cost = cdist(moving, fixed, metric="sqeuclidean") / (2.0 * float(UOT["cost_scale_um"]) ** 2)
    plan, log, _ = _run_plan(cost)
    return _extract(
        method, plan, log, moving, fixed, regional,
        allowed_indices=None, candidates=None, graph_support=None,
    )


def candidate_restricted_uot(
    fixed: np.ndarray,
    moving: np.ndarray,
    regional: RegionalMatch,
    candidates: CandidateSet,
    *,
    method: str,
    graph_support: np.ndarray | None = None,
) -> MatchResult:
    cost = np.full((len(moving), len(fixed)), float(CANDIDATES["outside_cost"]), dtype=np.float64)
    selected_cost = candidates.base_cost.copy()
    if graph_support is not None:
        selected_cost += float(CANDIDATES["graph_weight"]) * (1.0 - graph_support)
    np.put_along_axis(cost, candidates.indices, selected_cost, axis=1)
    plan, log, _ = _run_plan(cost)
    return _extract(
        method, plan, log, moving, fixed, regional,
        allowed_indices=candidates.indices,
        candidates=candidates,
        graph_support=graph_support,
    )


def direct_match(
    fixed: np.ndarray,
    moving: np.ndarray,
    regional: RegionalMatch,
    candidates: CandidateSet,
    *,
    method: str,
    ranking: np.ndarray,
    use_region_distance: bool,
) -> MatchResult:
    ranked = np.asarray(ranking, dtype=int)
    top1, top2 = ranked[:, 0], ranked[:, 1]
    if use_region_distance:
        full_score = cdist(moving + regional.cell_offsets, fixed)
        threshold_good, threshold_possible = 10.0, 18.0
    else:
        full_score = np.full((len(moving), len(fixed)), float(CANDIDATES["outside_cost"]))
        np.put_along_axis(full_score, candidates.indices, candidates.base_cost, axis=1)
        threshold_good, threshold_possible = 1.8, 3.5
    score1 = full_score[np.arange(len(moving)), top1]
    score2 = full_score[np.arange(len(moving)), top2]
    similarity1 = np.exp(-np.clip(score1, 0.0, 25.0))
    similarity2 = np.exp(-np.clip(score2, 0.0, 25.0))
    normalized_margin = (similarity1 - similarity2) / np.maximum(similarity1 + similarity2, 1e-12)
    column_top = np.argmin(full_score, axis=0)
    reciprocal = column_top[top1] == np.arange(len(moving))
    status = np.full(len(moving), "ambiguous", dtype=object)
    status[score1 > threshold_possible] = "unmatched"
    status[(score1 <= threshold_good) & reciprocal] = "matched"
    confidence = np.clip(0.55 * similarity1 + 0.25 * np.clip(normalized_margin, 0, 1) + 0.20 * reciprocal, 0, 1)

    slot_lookup = [{int(value): slot for slot, value in enumerate(row)} for row in candidates.indices]
    slots = np.asarray([slot_lookup[row].get(int(top1[row]), 0) for row in range(len(moving))])
    return MatchResult(
        method=method,
        ranked_indices=ranked,
        top1_index=top1,
        top2_index=top2,
        top1_mass=np.full(len(moving), np.nan),
        top2_mass=np.full(len(moving), np.nan),
        top1_top2_margin=normalized_margin,
        normalized_row_entropy=np.full(len(moving), np.nan),
        row_transport_mass=np.full(len(moving), np.nan),
        row_mass_ratio=np.full(len(moving), np.nan),
        reciprocal=reciprocal,
        status=status,
        confidence=confidence,
        regional_ncc=regional.best_ncc[regional.cell_anchor_index],
        regional_margin=regional.margin[regional.cell_anchor_index],
        fingerprint_similarity=candidates.fingerprint_similarity[np.arange(len(moving)), slots],
        graph_support=np.full(len(moving), np.nan),
        spatial_distance_um=np.linalg.norm(fixed[top1] - moving, axis=1),
        solver_final_error=None,
        solver_converged=True,
        transported_total_mass=None,
    )

