from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


MATCH_COLUMNS = [
    "fixed_cell_id",
    "moving_cell_id",
    "fixed_centroid_x",
    "fixed_centroid_y",
    "moving_centroid_x",
    "moving_centroid_y",
    "distance",
    "fixed_area",
    "moving_area",
    "area_ratio",
    "score",
    "confidence",
    "matched_status",
]


def _empty_matches() -> pd.DataFrame:
    return pd.DataFrame(columns=MATCH_COLUMNS)


def _unmatched_row(fixed_cell: pd.Series) -> dict:
    return {
        "fixed_cell_id": fixed_cell["cell_id"],
        "moving_cell_id": pd.NA,
        "fixed_centroid_x": fixed_cell["centroid_x"],
        "fixed_centroid_y": fixed_cell["centroid_y"],
        "moving_centroid_x": np.nan,
        "moving_centroid_y": np.nan,
        "distance": np.nan,
        "fixed_area": fixed_cell.get("area", np.nan),
        "moving_area": np.nan,
        "area_ratio": np.nan,
        "score": np.nan,
        "confidence": 0.0,
        "matched_status": "unmatched_fixed",
    }


def _match_row(fixed_cell: pd.Series, moving_cell: pd.Series, distance: float, score: float, status: str) -> dict:
    fixed_area = fixed_cell.get("area", np.nan)
    moving_area = moving_cell.get("area", np.nan)
    if np.isfinite(fixed_area) and np.isfinite(moving_area) and fixed_area > 0:
        area_ratio = moving_area / fixed_area
    else:
        area_ratio = np.nan

    return {
        "fixed_cell_id": fixed_cell["cell_id"],
        "moving_cell_id": moving_cell["cell_id"],
        "fixed_centroid_x": fixed_cell["centroid_x"],
        "fixed_centroid_y": fixed_cell["centroid_y"],
        "moving_centroid_x": moving_cell["centroid_x"],
        "moving_centroid_y": moving_cell["centroid_y"],
        "distance": distance,
        "fixed_area": fixed_area,
        "moving_area": moving_area,
        "area_ratio": area_ratio,
        "score": score,
        "confidence": float(np.exp(-score)),
        "matched_status": status,
    }


def match_cells(
    fixed_features: pd.DataFrame,
    moving_features: pd.DataFrame,
    *,
    max_distance: float,
    min_area_ratio: float,
    max_area_ratio: float,
    max_score: float,
    w_pos: float = 1.0,
    w_area: float = 1.0,
    w_shape: float = 1.0,
) -> pd.DataFrame:
    """Match fixed and moving cells one-to-one under an identity transform."""
    if fixed_features.empty:
        return _empty_matches()

    if moving_features.empty:
        return pd.DataFrame([_unmatched_row(row) for _, row in fixed_features.iterrows()], columns=MATCH_COLUMNS)

    required_columns = {"cell_id", "centroid_x", "centroid_y"}
    for name, features in (("fixed_features", fixed_features), ("moving_features", moving_features)):
        missing_columns = required_columns - set(features.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{name} is missing required columns: {missing}")

    if max_distance <= 0:
        raise ValueError("max_distance must be positive.")
    if min_area_ratio <= 0 or max_area_ratio <= 0:
        raise ValueError("Area ratio thresholds must be positive.")
    if min_area_ratio > max_area_ratio:
        raise ValueError("min_area_ratio must be less than or equal to max_area_ratio.")

    fixed = fixed_features.reset_index(drop=True)
    moving = moving_features.reset_index(drop=True)
    cost = np.full((len(fixed), len(moving)), 1.0e12, dtype=np.float64)
    pair_metrics: dict[tuple[int, int], tuple[float, float]] = {}

    for fixed_idx, fixed_cell in fixed.iterrows():
        for moving_idx, moving_cell in moving.iterrows():
            distance = float(
                np.hypot(
                    fixed_cell["centroid_x"] - moving_cell["centroid_x"],
                    fixed_cell["centroid_y"] - moving_cell["centroid_y"],
                )
            )
            if distance > max_distance:
                continue

            normalized_distance = distance / max_distance
            score = float(w_pos * normalized_distance)

            fixed_area = fixed_cell.get("area", np.nan)
            moving_area = moving_cell.get("area", np.nan)
            has_area = (
                np.isfinite(fixed_area)
                and np.isfinite(moving_area)
                and fixed_area > 0
                and moving_area > 0
            )
            if has_area:
                area_ratio = float(moving_area / fixed_area)
                if not (min_area_ratio <= area_ratio <= max_area_ratio):
                    continue
                score += float(w_area * abs(np.log(area_ratio)))

            fixed_ecc = fixed_cell.get("eccentricity", np.nan)
            moving_ecc = moving_cell.get("eccentricity", np.nan)
            if np.isfinite(fixed_ecc) and np.isfinite(moving_ecc):
                score += float(w_shape * abs(fixed_ecc - moving_ecc))

            cost[fixed_idx, moving_idx] = score
            pair_metrics[(fixed_idx, moving_idx)] = (distance, score)

    if not pair_metrics:
        return pd.DataFrame([_unmatched_row(row) for _, row in fixed.iterrows()], columns=MATCH_COLUMNS)

    row_indices, col_indices = linear_sum_assignment(cost)
    assigned_pairs = {
        (int(row_idx), int(col_idx))
        for row_idx, col_idx in zip(row_indices, col_indices)
        if (int(row_idx), int(col_idx)) in pair_metrics
    }

    rows = []
    for fixed_idx, fixed_cell in fixed.iterrows():
        pair = next((candidate for candidate in assigned_pairs if candidate[0] == fixed_idx), None)
        if pair is None:
            rows.append(_unmatched_row(fixed_cell))
            continue

        moving_idx = pair[1]
        distance, score = pair_metrics[pair]
        status = "matched" if score <= max_score else "low_confidence"
        rows.append(_match_row(fixed_cell, moving.loc[moving_idx], distance, score, status))

    return pd.DataFrame(rows, columns=MATCH_COLUMNS)


def initialize_empty_matches() -> pd.DataFrame:
    """Return the table schema planned for future cell correspondence results."""
    return _empty_matches()


# TODO: Add unmatched_moving support for moving-side cells not assigned to any fixed cell.
# TODO: Replace identity-transform matching with registration-aware candidate generation.
