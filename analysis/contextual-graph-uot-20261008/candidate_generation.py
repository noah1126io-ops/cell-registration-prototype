"""Candidate ranking and interpretable contextual cost construction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

from config import CANDIDATES
from regional_pattern import RegionalMatch


@dataclass(frozen=True)
class CandidateSet:
    indices: np.ndarray
    position_cost: np.ndarray
    fingerprint_cost: np.ndarray
    region_cost: np.ndarray
    base_cost: np.ndarray
    spatial_distance_um: np.ndarray
    region_residual_um: np.ndarray
    fingerprint_similarity: np.ndarray
    region_similarity: np.ndarray


@dataclass(frozen=True)
class CandidateRankings:
    spatial: np.ndarray
    regional: np.ndarray
    fingerprint: np.ndarray
    combined: np.ndarray
    spatial_distance_um: np.ndarray
    region_residual_um: np.ndarray


def _take(matrix: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.take_along_axis(matrix, indices, axis=1)


def _top_order(values: np.ndarray, k: int) -> np.ndarray:
    k = min(int(k), values.shape[1])
    if k == values.shape[1]:
        return np.argsort(values, axis=1)
    partial = np.argpartition(values, kth=k - 1, axis=1)[:, :k]
    local = np.argsort(_take(values, partial), axis=1)
    return _take(partial, local)


def build_candidate_rankings(
    fixed: np.ndarray,
    moving: np.ndarray,
    fingerprint_cost: np.ndarray,
    regional: RegionalMatch,
) -> CandidateRankings:
    top_k = int(CANDIDATES["top_k"])
    spatial_distance = cdist(moving, fixed)
    predicted = moving + regional.cell_offsets
    region_residual = cdist(predicted, fixed)
    spatial = _top_order(spatial_distance, top_k)
    regional_order = _top_order(region_residual, top_k)

    pool_k = min(int(CANDIDATES["fingerprint_spatial_pool"]), len(fixed))
    pool = _top_order(spatial_distance, pool_k)
    pool_fp = _take(fingerprint_cost, pool)
    fp_local_order = np.argsort(pool_fp, axis=1)[:, :top_k]
    fingerprint = _take(pool, fp_local_order)

    combined_rows = []
    source_k = int(CANDIDATES["source_k"])
    position_scale = float(CANDIDATES["position_scale_um"])
    region_scale = float(CANDIDATES["region_residual_scale_um"])
    weights = CANDIDATES["weights_full"]
    for row in range(len(moving)):
        union = list(
            dict.fromkeys(
                np.concatenate(
                    [spatial[row, :source_k], regional_order[row, :source_k], fingerprint[row, :source_k]]
                ).tolist()
            )
        )
        if len(union) < top_k:
            union.extend(index for index in spatial[row] if index not in set(union))
        candidates = np.asarray(union, dtype=int)
        position_component = spatial_distance[row, candidates] ** 2 / (2.0 * position_scale**2)
        region_component = region_residual[row, candidates] ** 2 / (2.0 * region_scale**2)
        region_component += 0.25 * (1.0 - regional.cell_confidence[row])
        cost = (
            weights["position"] * position_component
            + weights["fingerprint"] * fingerprint_cost[row, candidates]
            + weights["region"] * region_component
        )
        combined_rows.append(candidates[np.argsort(cost)[:top_k]])
    return CandidateRankings(
        spatial=spatial,
        regional=regional_order,
        fingerprint=fingerprint,
        combined=np.asarray(combined_rows, dtype=int),
        spatial_distance_um=spatial_distance,
        region_residual_um=region_residual,
    )


def build_candidate_set(
    rankings: CandidateRankings,
    fingerprint_cost: np.ndarray,
    regional: RegionalMatch,
    *,
    include_region: bool,
) -> CandidateSet:
    top_k = int(CANDIDATES["top_k"])
    source_k = int(CANDIDATES["source_k"])
    rows = []
    for row in range(rankings.spatial.shape[0]):
        sources = [rankings.spatial[row, :source_k], rankings.fingerprint[row, :source_k]]
        if include_region:
            sources.append(rankings.regional[row, :source_k])
        union = list(dict.fromkeys(np.concatenate(sources).tolist()))
        fill = rankings.combined[row] if include_region else rankings.spatial[row]
        used = set(union)
        for index in fill:
            if len(union) >= top_k:
                break
            if int(index) not in used:
                union.append(int(index))
                used.add(int(index))
        candidates = np.asarray(union, dtype=int)
        position_scale = float(CANDIDATES["position_scale_um"])
        region_scale = float(CANDIDATES["region_residual_scale_um"])
        position = rankings.spatial_distance_um[row, candidates] ** 2 / (2.0 * position_scale**2)
        fp = fingerprint_cost[row, candidates]
        region = rankings.region_residual_um[row, candidates] ** 2 / (2.0 * region_scale**2)
        region += 0.25 * (1.0 - regional.cell_confidence[row])
        weights = CANDIDATES["weights_full"] if include_region else CANDIDATES["weights_without_region"]
        base = weights["position"] * position + weights["fingerprint"] * fp
        if include_region:
            base = base + weights["region"] * region
        order = np.argsort(base)[:top_k]
        rows.append(candidates[order])

    indices = np.asarray(rows, dtype=int)
    position_distance = _take(rankings.spatial_distance_um, indices)
    region_residual = _take(rankings.region_residual_um, indices)
    position_cost = position_distance**2 / (2.0 * float(CANDIDATES["position_scale_um"]) ** 2)
    selected_fp = _take(fingerprint_cost, indices)
    region_cost = region_residual**2 / (2.0 * float(CANDIDATES["region_residual_scale_um"]) ** 2)
    region_cost += 0.25 * (1.0 - regional.cell_confidence[:, None])
    weights = CANDIDATES["weights_full"] if include_region else CANDIDATES["weights_without_region"]
    base_cost = weights["position"] * position_cost + weights["fingerprint"] * selected_fp
    if include_region:
        base_cost += weights["region"] * region_cost
    return CandidateSet(
        indices=indices,
        position_cost=position_cost,
        fingerprint_cost=selected_fp,
        region_cost=region_cost,
        base_cost=base_cost,
        spatial_distance_um=position_distance,
        region_residual_um=region_residual,
        fingerprint_similarity=np.exp(-0.5 * np.clip(selected_fp, 0.0, 20.0)),
        region_similarity=np.exp(-0.5 * np.clip(region_cost, 0.0, 20.0)),
    )


def candidate_recall(ranking: np.ndarray, true_fixed_index: np.ndarray, k: int) -> float:
    common = true_fixed_index >= 0
    if not np.any(common):
        return float("nan")
    retained = np.any(ranking[common, : min(k, ranking.shape[1])] == true_fixed_index[common, None], axis=1)
    return float(np.mean(retained))

