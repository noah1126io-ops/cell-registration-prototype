"""Post-hoc descriptor-group diagnostics without correspondence-cost tuning."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from config import DIFFICULTIES, OUT, SEEDS
from evaluation import rank_auc
from local_fingerprint import compute_fingerprint_pair
from synthetic_correspondence import generate_correspondence_sample


def _groups(names: tuple[str, ...]) -> dict[str, np.ndarray]:
    predicates = {
        "absolute kNN distances": lambda name: name.startswith("log_knn_abs"),
        "locally normalized kNN distances": lambda name: name.startswith("knn_local_norm"),
        "radial density/count": lambda name: name.startswith("gaussian_density") or name.startswith("radial_count"),
        "angular Fourier": lambda name: name.startswith("angular_fourier"),
        "covariance anisotropy/scale": lambda name: name.startswith("covariance"),
        "neighbor-pair geometry": lambda name: name.startswith("neighbor_pair") or name.startswith("angular_gap"),
        "all fingerprint components": lambda name: True,
    }
    return {
        label: np.asarray([index for index, name in enumerate(names) if predicate(name)], dtype=int)
        for label, predicate in predicates.items()
    }


def _squared_cost(moving: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    dimensions = max(float(moving.shape[1]), 1.0)
    values = (
        np.sum(moving * moving, axis=1)[:, None]
        + np.sum(fixed * fixed, axis=1)[None, :]
        - 2.0 * moving @ fixed.T
    )
    return np.maximum(values, 0.0) / dimensions


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for difficulty, settings in DIFFICULTIES.items():
        for seed in SEEDS:
            sample = generate_correspondence_sample(seed=seed, difficulty=difficulty, **settings)
            pair = compute_fingerprint_pair(sample.fixed_points, sample.moving_points)
            spatial = cdist(sample.moving_points, sample.fixed_points)
            pool = np.argsort(spatial, axis=1)[:, : min(80, len(sample.fixed_points))]
            common = sample.true_fixed_index >= 0
            for label, columns in _groups(pair.feature_names).items():
                full_cost = _squared_cost(pair.moving[:, columns], pair.fixed[:, columns])
                pool_cost = np.take_along_axis(full_cost, pool, axis=1)
                order = np.argsort(pool_cost, axis=1)
                ranked = np.take_along_axis(pool, order, axis=1)
                labels = pool == sample.true_fixed_index[:, None]
                evaluable = common & np.any(labels, axis=1)
                rows.append(
                    {
                        "difficulty": difficulty,
                        "seed": seed,
                        "descriptor_group": label,
                        "feature_count": len(columns),
                        "evaluable_common_n": int(np.count_nonzero(evaluable)),
                        "candidate_recall_at_10": float(
                            np.mean(np.any(ranked[evaluable, :10] == sample.true_fixed_index[evaluable, None], axis=1))
                        ),
                        "candidate_recall_at_20": float(
                            np.mean(np.any(ranked[evaluable, :20] == sample.true_fixed_index[evaluable, None], axis=1))
                        ),
                        "true_candidate_mean_rank": float(
                            np.mean(np.argmax(ranked[evaluable] == sample.true_fixed_index[evaluable, None], axis=1) + 1)
                        ),
                        "true_vs_hard_negative_auc": rank_auc(
                            labels[evaluable].ravel(), (-pool_cost[evaluable]).ravel()
                        ),
                    }
                )
    per_case = pd.DataFrame(rows)
    summary = (
        per_case.groupby("descriptor_group", sort=False)
        .agg(
            feature_count=("feature_count", "first"),
            candidate_recall_at_10=("candidate_recall_at_10", "mean"),
            candidate_recall_at_20=("candidate_recall_at_20", "mean"),
            true_candidate_mean_rank=("true_candidate_mean_rank", "mean"),
            true_vs_hard_negative_auc=("true_vs_hard_negative_auc", "mean"),
        )
        .reset_index()
        .sort_values(["candidate_recall_at_10", "true_vs_hard_negative_auc"], ascending=False)
    )
    per_case.to_csv(OUT / "fingerprint_component_analysis.csv", index=False)
    summary.to_csv(OUT / "fingerprint_component_summary.csv", index=False)
    return per_case, summary


if __name__ == "__main__":
    _, summary = run()
    print(summary.to_string(index=False))

