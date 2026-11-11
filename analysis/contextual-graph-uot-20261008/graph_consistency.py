"""Local soft graph support without global graph-isomorphism optimization."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from candidate_generation import CandidateSet
from config import GRAPH


def mutual_knn_graph(points: np.ndarray, k: int | None = None) -> list[np.ndarray]:
    points = np.asarray(points, dtype=float)
    k = min(int(k or GRAPH["mutual_knn"]), len(points) - 1)
    neighbors = cKDTree(points).query(points, k=k + 1)[1][:, 1:]
    neighbor_sets = [set(map(int, row)) for row in neighbors]
    return [
        np.asarray([j for j in row if i in neighbor_sets[int(j)]], dtype=int)
        for i, row in enumerate(neighbors)
    ]


def graph_support_scores(
    fixed: np.ndarray,
    moving: np.ndarray,
    candidates: CandidateSet,
) -> np.ndarray:
    """Score whether multiple neighboring edges have compatible candidates.

    The aggregation is deliberately soft: missing neighbors reduce the number
    of opportunities for support but do not require complete neighborhood
    preservation.
    """
    moving_graph = mutual_knn_graph(moving)
    fixed_graph = mutual_knn_graph(fixed)
    fixed_sets = [set(map(int, values)) for values in fixed_graph]
    output = np.zeros(candidates.indices.shape, dtype=float)
    length_tolerance = float(GRAPH["length_tolerance_um"])
    angle_tolerance = float(GRAPH["angle_tolerance_radians"])

    for moving_index, moving_neighbors in enumerate(moving_graph):
        if len(moving_neighbors) == 0:
            continue
        for slot, fixed_index in enumerate(candidates.indices[moving_index]):
            neighbor_scores = []
            fixed_neighbor_set = fixed_sets[int(fixed_index)]
            if not fixed_neighbor_set:
                continue
            for moving_neighbor in moving_neighbors:
                moving_vector = moving[int(moving_neighbor)] - moving[moving_index]
                moving_length = max(float(np.linalg.norm(moving_vector)), 1e-6)
                best = 0.0
                for neighbor_slot, fixed_neighbor in enumerate(candidates.indices[int(moving_neighbor)]):
                    if int(fixed_neighbor) not in fixed_neighbor_set:
                        continue
                    fixed_vector = fixed[int(fixed_neighbor)] - fixed[int(fixed_index)]
                    fixed_length = max(float(np.linalg.norm(fixed_vector)), 1e-6)
                    length_delta = abs(fixed_length - moving_length)
                    cosine = np.clip(
                        float(np.dot(moving_vector, fixed_vector) / (moving_length * fixed_length)), -1.0, 1.0
                    )
                    angle_delta = float(np.arccos(cosine))
                    geometry = np.exp(-0.5 * (length_delta / length_tolerance) ** 2)
                    geometry *= np.exp(-0.5 * (angle_delta / angle_tolerance) ** 2)
                    compatibility = np.exp(-0.35 * min(float(candidates.base_cost[int(moving_neighbor), neighbor_slot]), 10.0))
                    best = max(best, float(geometry * compatibility))
                neighbor_scores.append(best)
            values = np.asarray(neighbor_scores, dtype=float)
            if values.size:
                strong_fraction = float(np.mean(values >= 0.30))
                robust_mean = float(np.mean(np.sort(values)[-max(2, int(np.ceil(0.6 * len(values)))) :]))
                output[moving_index, slot] = 0.5 * strong_fraction + 0.5 * robust_mean
    return np.clip(output, 0.0, 1.0)

