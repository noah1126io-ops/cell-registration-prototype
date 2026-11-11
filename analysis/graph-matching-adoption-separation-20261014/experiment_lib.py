"""Analysis-only Graph matching/adoption separation experiment utilities."""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import cdist
from scipy.special import expit


ROOT = Path("/mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu")
SOURCE = ROOT / "analysis/contextual-graph-uot-20261008"
COST_SOURCE = ROOT / "analysis/contextual-uot-cost-ablation-20261008"
OUT = ROOT / "analysis/graph-matching-adoption-separation-20261014"
REAL_INPUT = ROOT / "sweeps/real-grid27-20260908-v2/input"

sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(COST_SOURCE))

from config import CANDIDATES  # noqa: E402
from candidate_generation import CandidateSet  # noqa: E402
from graph_consistency import graph_support_scores  # noqa: E402
from synthetic_correspondence import (  # noqa: E402
    CorrespondenceSample,
    _spatial_missing_probability,
    assert_identity_alignment,
)
from validation_lib import _inverse_points, _sample_ellipse_points, _true_field  # noqa: E402
from run_cost_ablation import (  # noqa: E402
    FixedCandidateContext,
    UOTRun,
    build_fixed_context,
    candidate_hash,
    dense_candidate_cost,
    run_uot,
    selected_cost,
)


METHOD_POSITION = "Position only"
METHOD_B0 = "Position + Graph"
METHOD_B1 = "Relative-cost Graph-UOT"
METHOD_B2 = "Separated Matching / Adoption"
METHOD_B3 = "Full contextual UOT"
METHOD_GRAPH_RAW = "Graph ranking only + separated adoption"
MAIN_METHODS = (METHOD_B0, METHOD_B1, METHOD_B2, METHOD_B3)

UOT_FEATURES = (
    "log1p_absolute_min_cost",
    "absolute_cost_margin",
    "top1_graph_support",
    "graph_support_margin",
    "normalized_top1_mass",
    "top1_top2_mass_margin",
    "selection_certainty",
    "reciprocal_numeric",
    "log1p_row_mass_ratio",
    "log1p_spatial_distance_um",
)

RAW_FEATURES = (
    "log1p_absolute_min_cost",
    "absolute_cost_margin",
    "top1_graph_support",
    "graph_support_margin",
    "normalized_top1_mass",
    "top1_top2_mass_margin",
    "selection_certainty",
    "reciprocal_numeric",
    "log1p_spatial_distance_um",
)

STRENGTHS = {
    "Weak": {
        "amplitude_um": 6.0,
        "fixed_missing_fraction": 0.25,
        "moving_missing_fraction": 0.28,
        "overlap_fraction": 0.95,
        "noise_um": 0.60,
        "local_missing_strength": 0.45,
    },
    "Moderate": {
        "amplitude_um": 10.0,
        "fixed_missing_fraction": 0.28,
        "moving_missing_fraction": 0.32,
        "overlap_fraction": 0.88,
        "noise_um": 1.00,
        "local_missing_strength": 0.60,
    },
    "Strong": {
        "amplitude_um": 14.0,
        "fixed_missing_fraction": 0.32,
        "moving_missing_fraction": 0.38,
        "overlap_fraction": 0.80,
        "noise_um": 1.40,
        "local_missing_strength": 0.75,
    },
}

SPLIT_SEEDS = {
    "Train": (2101, 2102, 2103, 2104, 2105),
    "Validation": (3101, 3102, 3103, 3104, 3105),
    "Test": (4101, 4102, 4103, 4104, 4105),
}


def validate_finite(name: str, values: np.ndarray) -> None:
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains {np.count_nonzero(~np.isfinite(values))} NaN/Inf values")


def _f1(precision: float, recall: float) -> float:
    if not np.isfinite(precision) or not np.isfinite(recall) or precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def point_density_and_knn(points: np.ndarray) -> tuple[float, np.ndarray]:
    points = np.asarray(points, dtype=float)
    area = float(ConvexHull(points).volume)
    knn = cKDTree(points).query(points, k=2)[0][:, 1]
    return float(len(points) / area), knn


def real_like_scenarios() -> list[dict]:
    strengths = tuple(STRENGTHS)
    rows = []
    for split, seeds in SPLIT_SEEDS.items():
        for index, seed in enumerate(seeds):
            strength = strengths[index % len(strengths)]
            rows.append(
                {
                    "scenario_id": f"real_like_{split.lower()}_{strength.lower()}_{seed}",
                    "split": split,
                    "seed": int(seed),
                    "strength": strength,
                    "template_group": f"base_pattern_{seed}",
                    **STRENGTHS[strength],
                }
            )
    return rows


def _sample_nonuniform_points(
    rng: np.random.Generator,
    count: int,
    extent: float,
) -> np.ndarray:
    """Mildly clustered tissue sampler; less concentrated than the legacy benchmark."""
    center = np.asarray([0.50 * extent, 0.51 * extent])
    radii = np.asarray([0.43 * extent, 0.38 * extent])
    cluster_centers = np.asarray([[0.30, 0.36], [0.55, 0.66], [0.71, 0.40]]) * extent
    points: list[np.ndarray] = []
    while len(points) < count:
        if rng.random() < 0.35:
            candidate = cluster_centers[rng.integers(len(cluster_centers))] + rng.normal(
                0.0, 0.14 * extent, 2
            )
        else:
            candidate = rng.uniform(0.06 * extent, 0.94 * extent, 2)
        if np.sum(((candidate - center) / radii) ** 2) <= 1.0:
            points.append(candidate)
    return np.asarray(points, dtype=float)


def generate_real_like_sample(*, seed: int, strength: str) -> CorrespondenceSample:
    """Generate low-density, nonuniform serial-section points with persistent IDs.

    Inclusion in the two sections is sampled independently. Geometry controls
    point density/deformation; identity labels are attached afterward and are
    never consumed by matching.
    """
    settings = STRENGTHS[strength]
    rng = np.random.default_rng(seed)
    size, spacing = 256, 2.0
    # Keep the post-missing moving cloud near the real 0.0019 nuclei/um2
    # across deformation/missingness strengths.
    n_latent = {"Weak": 300, "Moderate": 330, "Strong": 350}[strength]
    axes = np.arange(size, dtype=float) * spacing
    grid_x, grid_y = np.meshgrid(axes, axes)
    extent = float(axes[-1])
    field_x, field_y = _true_field(grid_x, grid_y, float(settings["amplitude_um"]))

    latent = _sample_nonuniform_points(rng, n_latent, extent)
    latent_ids = np.asarray([f"latent_{seed}_{index:05d}" for index in range(n_latent)], dtype=object)
    fixed_missing = _spatial_missing_probability(
        latent,
        np.random.default_rng(seed + 101),
        float(settings["fixed_missing_fraction"]),
        float(settings["local_missing_strength"]),
        extent,
    )
    moving_missing = _spatial_missing_probability(
        latent,
        np.random.default_rng(seed + 211),
        float(settings["moving_missing_fraction"]),
        float(settings["local_missing_strength"]),
        extent,
    )
    fixed_keep = rng.random(n_latent) >= fixed_missing
    overlap_cutoff = float(np.quantile(latent[:, 0], float(settings["overlap_fraction"])))
    moving_keep = (latent[:, 0] <= overlap_cutoff) & (rng.random(n_latent) >= moving_missing)

    fixed_points = latent[fixed_keep]
    fixed_ids = latent_ids[fixed_keep]
    moving_clean = _inverse_points(latent[moving_keep], field_x, field_y, spacing)
    moving_points = moving_clean + rng.normal(0.0, float(settings["noise_um"]), moving_clean.shape)
    moving_ids = latent_ids[moving_keep]

    # Section-specific nuclei preserve unequal cardinality without creating IDs.
    n_fixed_new = max(12, int(round(0.07 * len(fixed_points))))
    n_moving_new = max(12, int(round(0.09 * len(moving_points))))
    fixed_new = _sample_nonuniform_points(rng, n_fixed_new, extent)
    moving_new = _sample_nonuniform_points(rng, n_moving_new, extent)
    fixed_points = np.vstack([fixed_points, fixed_new])
    moving_points = np.vstack([moving_points, moving_new])
    fixed_ids = np.concatenate(
        [fixed_ids, np.asarray([f"fixed_new_{seed}_{i:04d}" for i in range(n_fixed_new)], dtype=object)]
    )
    moving_ids = np.concatenate(
        [moving_ids, np.asarray([f"moving_new_{seed}_{i:04d}" for i in range(n_moving_new)], dtype=object)]
    )

    fixed_order = rng.permutation(len(fixed_points))
    moving_order = rng.permutation(len(moving_points))
    fixed_points, fixed_ids = fixed_points[fixed_order], fixed_ids[fixed_order]
    moving_points, moving_ids = moving_points[moving_order], moving_ids[moving_order]

    fixed_lookup = {cell_id: index for index, cell_id in enumerate(fixed_ids)}
    true_fixed_index = np.asarray([fixed_lookup.get(cell_id, -1) for cell_id in moving_ids], dtype=int)
    common = true_fixed_index >= 0

    # Exact affine round trip: returned coordinates are post-affine matching coordinates.
    theta = np.deg2rad(2.5)
    affine = np.array(
        [
            [0.99 * np.cos(theta), -1.01 * np.sin(theta) + 0.008],
            [0.99 * np.sin(theta), 1.01 * np.cos(theta)],
        ]
    )
    translation = np.asarray([9.0, -6.0])
    raw = (moving_points - translation) @ np.linalg.inv(affine).T
    aligned = raw @ affine.T + translation
    np.testing.assert_allclose(aligned, moving_points, atol=1e-10, rtol=0)

    sample = CorrespondenceSample(
        fixed_points=fixed_points,
        moving_points=aligned,
        fixed_ids=fixed_ids,
        moving_ids=moving_ids,
        true_fixed_index=true_fixed_index,
        common_moving_mask=common,
        grid_x=grid_x,
        grid_y=grid_y,
        true_field_x=field_x,
        true_field_y=field_y,
        spacing=spacing,
        bounds=(0.0, 0.0, extent, extent),
        seed=int(seed),
        difficulty=f"Real-like {strength}",
        amplitude_um=float(settings["amplitude_um"]),
        missing_fraction=float(settings["moving_missing_fraction"]),
        unmatched_fraction=float(np.mean(~common)),
        overlap_fraction=float(settings["overlap_fraction"]),
        noise_um=float(settings["noise_um"]),
        local_missing_strength=float(settings["local_missing_strength"]),
    )
    assert_identity_alignment(sample)
    return sample


def make_negative_control(seed: int, kind: str) -> CorrespondenceSample:
    sample = generate_real_like_sample(seed=seed, strength="Moderate")
    rng = np.random.default_rng(seed + 7001)
    if kind == "identity_permutation":
        moving = sample.moving_points.copy()
        note = "Coordinates preserved; all persistent IDs replaced. Geometry is intentionally confounded."
    elif kind == "unrelated_pattern":
        other = generate_real_like_sample(seed=seed + 503, strength="Moderate")
        moving = other.moving_points.copy()
        if len(moving) > len(sample.moving_points):
            moving = moving[: len(sample.moving_points)]
        note = "Independent nonuniform point pattern with similar generator-level density."
    elif kind == "translated_disjoint":
        moving = sample.moving_points + np.asarray([115.0, -90.0])
        note = "Density/cardinality preserved; gross cross-cloud alignment destroyed by translation."
    else:
        raise ValueError(f"Unknown control: {kind}")
    ids = np.asarray([f"control_{kind}_{seed}_{i:05d}" for i in range(len(moving))], dtype=object)
    result = replace(
        sample,
        moving_points=moving,
        moving_ids=ids,
        true_fixed_index=np.full(len(moving), -1, dtype=int),
        common_moving_mask=np.zeros(len(moving), dtype=bool),
        difficulty=f"Negative control: {kind}",
    )
    object.__setattr__(result, "control_note", note) if hasattr(result, "control_note") else None
    assert_identity_alignment(result)
    return result


def method_runs(context: FixedCandidateContext) -> dict[str, UOTRun]:
    position = selected_cost(context, ("position",))
    position_graph = selected_cost(context, ("position", "graph"))
    full = selected_cost(context, ("position", "fingerprint", "region", "graph"))

    b0 = run_uot(context, METHOD_B0, position_graph, graph_in_condition=True)
    offsets = np.min(position_graph, axis=1)
    relative_selected = position_graph - offsets[:, None]
    # Center only the allowed candidate costs. The fixed non-candidate penalty
    # remains 25 so extreme negative controls cannot turn excluded pairs into
    # negative-cost transport destinations.
    relative_dense = dense_candidate_cost(context, relative_selected)
    if np.min(relative_dense) < -1e-12:
        raise AssertionError("Row-relative transformation produced negative cost")
    b1 = run_uot(
        context,
        METHOD_B1,
        relative_selected,
        dense_override=relative_dense,
        graph_in_condition=True,
    )
    b3 = run_uot(context, METHOD_B3, full, graph_in_condition=True)
    position_run = run_uot(context, METHOD_POSITION, position, graph_in_condition=False)
    hashes = {run.metrics["candidate_hash"] for run in (b0, b1, b3, position_run)}
    if hashes != {candidate_hash(context.candidates.indices)}:
        raise AssertionError("Candidate indices changed across methods")
    return {METHOD_POSITION: position_run, METHOD_B0: b0, METHOD_B1: b1, METHOD_B3: b3}


def _slots(candidate_indices: np.ndarray, selected: np.ndarray) -> np.ndarray:
    matches = candidate_indices == selected[:, None]
    if not np.all(matches.sum(axis=1) == 1):
        raise AssertionError("Selected destination is missing or duplicated in candidate set")
    return np.argmax(matches, axis=1)


def _raw_reciprocal(indices: np.ndarray, cost: np.ndarray, top1: np.ndarray, n_fixed: int) -> np.ndarray:
    best_cost = np.full(n_fixed, np.inf)
    best_moving = np.full(n_fixed, -1, dtype=int)
    for moving_index in range(len(indices)):
        for slot, fixed_index in enumerate(indices[moving_index]):
            value = float(cost[moving_index, slot])
            if value < best_cost[int(fixed_index)]:
                best_cost[int(fixed_index)] = value
                best_moving[int(fixed_index)] = moving_index
    return best_moving[top1] == np.arange(len(top1))


def _softmax_diagnostics(relative_cost: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = -np.asarray(relative_cost, dtype=float)
    logits -= np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(np.clip(logits, -700, 0))
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    ordered = np.sort(probabilities, axis=1)[:, ::-1]
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)), axis=1)
    entropy /= np.log(probabilities.shape[1])
    return ordered[:, 0], ordered[:, 0] - ordered[:, 1], entropy


def result_frame(
    sample: CorrespondenceSample,
    scenario: dict,
    context: FixedCandidateContext,
    run: UOTRun,
    *,
    method: str,
) -> pd.DataFrame:
    n = len(sample.moving_points)
    indices = context.candidates.indices
    absolute_cost = selected_cost(context, ("position", "graph"))
    order = np.argsort(absolute_cost, axis=1)
    sorted_absolute = np.take_along_axis(absolute_cost, order, axis=1)
    selected = run.result.top1_index.astype(int)
    slots = _slots(indices, selected)
    top2_slots = _slots(indices, run.result.top2_index.astype(int))
    common = sample.true_fixed_index >= 0
    correct = common & (selected == sample.true_fixed_index)
    top3 = common & np.any(run.result.ranked_indices[:, :3] == sample.true_fixed_index[:, None], axis=1)
    true_in_candidates = common & np.any(indices == sample.true_fixed_index[:, None], axis=1)
    graph_top = context.graph_support[np.arange(n), slots]
    graph_second = context.graph_support[np.arange(n), top2_slots]
    selected_absolute = absolute_cost[np.arange(n), slots]
    frame = pd.DataFrame(
        {
            "scenario_id": scenario["scenario_id"],
            "split": scenario.get("split", "Control"),
            "strength": scenario.get("strength", sample.difficulty),
            "seed": sample.seed,
            "template_group": scenario.get("template_group", f"control_{sample.seed}"),
            "method": method,
            "moving_index": np.arange(n),
            "moving_cell_id": sample.moving_ids,
            "true_fixed_index": sample.true_fixed_index,
            "is_common_cell": common,
            "true_in_candidate_set": true_in_candidates,
            "top1_fixed_index": selected,
            "top1_correct": correct,
            "top3_contains_true": top3,
            "status": run.result.status,
            "native_confidence": run.result.confidence,
            "calibrated_confidence": run.result.confidence,
            "absolute_min_cost": sorted_absolute[:, 0],
            "absolute_second_cost": sorted_absolute[:, 1],
            "absolute_cost_margin": sorted_absolute[:, 1] - sorted_absolute[:, 0],
            "selected_absolute_cost": selected_absolute,
            "relative_matching_score": -selected_absolute + sorted_absolute[:, 0],
            "top1_graph_support": graph_top,
            "graph_support_margin": graph_top - graph_second,
            "normalized_top1_mass": np.divide(
                run.result.top1_mass,
                run.result.row_transport_mass,
                out=np.zeros(n),
                where=run.result.row_transport_mass > 0,
            ),
            "top1_top2_mass_margin": run.result.top1_top2_margin,
            "normalized_row_entropy": run.result.normalized_row_entropy,
            "row_mass_ratio": run.result.row_mass_ratio,
            "reciprocal": run.result.reciprocal,
            "spatial_distance_um": np.linalg.norm(context.fixed[selected] - context.moving, axis=1),
            "candidate_hash": candidate_hash(indices),
            "candidate_count": indices.shape[1],
            "fixed_n": len(context.fixed),
            "moving_n": len(context.moving),
            "transported_total_mass": run.metrics["total_transported_mass"],
            "solver_converged": run.metrics["solver_converged"],
        }
    )
    _add_model_features(frame)
    validate_finite(f"{method} features", frame[list(UOT_FEATURES)].to_numpy(float))
    return frame


def graph_ranking_frame(
    sample: CorrespondenceSample,
    scenario: dict,
    context: FixedCandidateContext,
) -> pd.DataFrame:
    indices = context.candidates.indices
    cost = selected_cost(context, ("position", "graph"))
    order = np.argsort(cost, axis=1)
    ranked = np.take_along_axis(indices, order, axis=1)
    sorted_cost = np.take_along_axis(cost, order, axis=1)
    top1, top2 = ranked[:, 0], ranked[:, 1]
    top1_slots, top2_slots = order[:, 0], order[:, 1]
    common = sample.true_fixed_index >= 0
    correct = common & (top1 == sample.true_fixed_index)
    top3 = common & np.any(ranked[:, :3] == sample.true_fixed_index[:, None], axis=1)
    true_in_candidates = common & np.any(indices == sample.true_fixed_index[:, None], axis=1)
    relative = cost - np.min(cost, axis=1, keepdims=True)
    concentration, margin, entropy = _softmax_diagnostics(relative)
    reciprocal = _raw_reciprocal(indices, cost, top1, len(context.fixed))
    n = len(top1)
    frame = pd.DataFrame(
        {
            "scenario_id": scenario["scenario_id"],
            "split": scenario.get("split", "Control"),
            "strength": scenario.get("strength", sample.difficulty),
            "seed": sample.seed,
            "template_group": scenario.get("template_group", f"control_{sample.seed}"),
            "method": METHOD_GRAPH_RAW,
            "moving_index": np.arange(n),
            "moving_cell_id": sample.moving_ids,
            "true_fixed_index": sample.true_fixed_index,
            "is_common_cell": common,
            "true_in_candidate_set": true_in_candidates,
            "top1_fixed_index": top1,
            "top1_correct": correct,
            "top3_contains_true": top3,
            "status": "ambiguous",
            "native_confidence": concentration,
            "calibrated_confidence": concentration,
            "absolute_min_cost": sorted_cost[:, 0],
            "absolute_second_cost": sorted_cost[:, 1],
            "absolute_cost_margin": sorted_cost[:, 1] - sorted_cost[:, 0],
            "selected_absolute_cost": sorted_cost[:, 0],
            "relative_matching_score": np.zeros(n),
            "top1_graph_support": context.graph_support[np.arange(n), top1_slots],
            "graph_support_margin": context.graph_support[np.arange(n), top1_slots]
            - context.graph_support[np.arange(n), top2_slots],
            "normalized_top1_mass": concentration,
            "top1_top2_mass_margin": margin,
            "normalized_row_entropy": entropy,
            "row_mass_ratio": np.ones(n),
            "reciprocal": reciprocal,
            "spatial_distance_um": np.linalg.norm(context.fixed[top1] - context.moving, axis=1),
            "candidate_hash": candidate_hash(indices),
            "candidate_count": indices.shape[1],
            "fixed_n": len(context.fixed),
            "moving_n": len(context.moving),
            "transported_total_mass": np.nan,
            "solver_converged": True,
        }
    )
    _add_model_features(frame)
    validate_finite("raw graph features", frame[list(RAW_FEATURES)].to_numpy(float))
    return frame


def _add_model_features(frame: pd.DataFrame) -> None:
    frame["log1p_absolute_min_cost"] = np.log1p(np.maximum(frame["absolute_min_cost"], 0.0))
    frame["selection_certainty"] = 1.0 - frame["normalized_row_entropy"]
    frame["reciprocal_numeric"] = frame["reciprocal"].astype(float)
    frame["log1p_row_mass_ratio"] = np.log1p(np.maximum(frame["row_mass_ratio"], 0.0))
    frame["log1p_spatial_distance_um"] = np.log1p(np.maximum(frame["spatial_distance_um"], 0.0))


@dataclass
class LogisticCalibrator:
    feature_names: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    training_rows: int
    positive_fraction: float
    converged: bool

    @classmethod
    def fit(
        cls,
        table: pd.DataFrame,
        feature_names: Iterable[str],
        target: np.ndarray,
        *,
        l2: float = 0.05,
    ) -> "LogisticCalibrator":
        names = tuple(feature_names)
        x = table.loc[:, names].to_numpy(dtype=float)
        y = np.asarray(target, dtype=float)
        validate_finite("calibration X", x)
        validate_finite("calibration y", y)
        center = np.mean(x, axis=0)
        scale = np.maximum(np.std(x, axis=0), 1e-8)
        z = (x - center) / scale

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            intercept, coefficients = parameters[0], parameters[1:]
            logits = intercept + z @ coefficients
            probabilities = expit(logits)
            loss = float(np.mean(np.logaddexp(0.0, logits) - y * logits))
            loss += 0.5 * l2 * float(np.sum(coefficients**2))
            residual = probabilities - y
            gradient = np.r_[np.mean(residual), z.T @ residual / len(y) + l2 * coefficients]
            return loss, gradient

        prior = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
        initial = np.r_[np.log(prior / (1 - prior)), np.zeros(z.shape[1])]
        result = minimize(objective, initial, jac=True, method="L-BFGS-B")
        return cls(
            feature_names=names,
            center=center,
            scale=scale,
            coefficients=result.x[1:],
            intercept=float(result.x[0]),
            training_rows=len(table),
            positive_fraction=float(np.mean(y)),
            converged=bool(result.success),
        )

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        x = table.loc[:, self.feature_names].to_numpy(dtype=float)
        validate_finite("calibration prediction X", x)
        z = (x - self.center) / self.scale
        return expit(self.intercept + z @ self.coefficients)

    def to_dict(self) -> dict:
        return {
            "feature_names": list(self.feature_names),
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "training_rows": self.training_rows,
            "positive_fraction": self.positive_fraction,
            "converged": self.converged,
        }


def threshold_for_precision(
    scores: np.ndarray,
    correct: np.ndarray,
    target_precision: float,
    *,
    minimum_selected: int = 5,
) -> dict:
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores, sorted_correct = scores[order], correct[order]
    cumulative = np.cumsum(sorted_correct)
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(scores) - 1]
    selected = ends + 1
    precision = cumulative[ends] / selected
    eligible = (selected >= minimum_selected) & (precision >= target_precision)
    if not np.any(eligible):
        return {
            "threshold": float(np.nextafter(np.max(scores), np.inf)),
            "validation_precision": float("nan"),
            "validation_coverage": 0.0,
            "selected_n": 0,
            "achieved": False,
        }
    candidates = np.flatnonzero(eligible)
    chosen = candidates[np.argmax(selected[candidates])]
    end = ends[chosen]
    return {
        "threshold": float(sorted_scores[end]),
        "validation_precision": float(precision[chosen]),
        "validation_coverage": float(selected[chosen] / len(scores)),
        "selected_n": int(selected[chosen]),
        "achieved": True,
    }


def unmatched_threshold(scores: np.ndarray, truth: np.ndarray, eligible: np.ndarray) -> dict:
    scores = np.asarray(scores, dtype=float)
    truth = np.asarray(truth, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    values = np.unique(scores[eligible])
    if not len(values):
        return {"threshold": float("inf"), "validation_f1": 0.0}
    best = {"threshold": float(np.max(values)), "validation_f1": -1.0}
    for threshold in values:
        predicted = eligible & (scores >= threshold)
        precision = _divide(np.count_nonzero(predicted & truth), np.count_nonzero(predicted))
        recall = _divide(np.count_nonzero(predicted & truth), np.count_nonzero(truth))
        value = _f1(precision, recall)
        if value > best["validation_f1"] + 1e-15:
            best = {"threshold": float(threshold), "validation_f1": float(value)}
    return best


def apply_separated_adoption(
    table: pd.DataFrame,
    match_model: LogisticCalibrator,
    unmatched_model: LogisticCalibrator,
    match_threshold: float,
    unmatched_cutoff: float,
    *,
    method: str,
) -> pd.DataFrame:
    result = table.copy()
    match_probability = match_model.predict(result)
    unmatched_probability = unmatched_model.predict(result)
    matched = match_probability >= match_threshold
    unmatched = (~matched) & (unmatched_probability >= unmatched_cutoff)
    status = np.full(len(result), "ambiguous", dtype=object)
    status[unmatched] = "unmatched"
    status[matched] = "matched"
    result["method"] = method
    result["calibrated_confidence"] = match_probability
    result["predicted_unmatched_probability"] = unmatched_probability
    result["status"] = status
    return result


def fit_separated_models(
    frames: pd.DataFrame,
    *,
    source_method: str,
    feature_names: tuple[str, ...],
    output_method: str,
) -> tuple[LogisticCalibrator, LogisticCalibrator, dict, pd.DataFrame]:
    source = frames.loc[frames["method"] == source_method].copy()
    train = source.loc[source["split"] == "Train"]
    validation = source.loc[source["split"] == "Validation"]
    match_model = LogisticCalibrator.fit(train, feature_names, train["top1_correct"].to_numpy(bool))
    unmatched_model = LogisticCalibrator.fit(train, feature_names, ~train["is_common_cell"].to_numpy(bool))
    validation_match_scores = match_model.predict(validation)
    selection = threshold_for_precision(validation_match_scores, validation["top1_correct"], 0.95)
    unmatched_scores = unmatched_model.predict(validation)
    eligible = validation_match_scores < selection["threshold"]
    unmatched_selection = unmatched_threshold(
        unmatched_scores,
        ~validation["is_common_cell"].to_numpy(bool),
        eligible,
    )
    applied = apply_separated_adoption(
        source,
        match_model,
        unmatched_model,
        selection["threshold"],
        unmatched_selection["threshold"],
        method=output_method,
    )
    thresholds = {
        "match": selection,
        "unmatched": unmatched_selection,
        "source_method": source_method,
        "output_method": output_method,
    }
    return match_model, unmatched_model, thresholds, applied


def method_metrics(table: pd.DataFrame, *, record_type: str = "aggregate") -> dict:
    common = table["is_common_cell"].to_numpy(bool)
    correct = table["top1_correct"].to_numpy(bool)
    matched = table["status"].to_numpy(str) == "matched"
    predicted_unmatched = table["status"].to_numpy(str) == "unmatched"
    true_unmatched = ~common
    matched_precision = _divide(np.count_nonzero(matched & correct), np.count_nonzero(matched))
    matched_recall = _divide(np.count_nonzero(matched & correct), np.count_nonzero(common))
    unmatched_precision = _divide(
        np.count_nonzero(predicted_unmatched & true_unmatched), np.count_nonzero(predicted_unmatched)
    )
    unmatched_recall = _divide(
        np.count_nonzero(predicted_unmatched & true_unmatched), np.count_nonzero(true_unmatched)
    )
    matched_table = table.loc[matched]
    duplicate_count = int(
        sum(
            len(group) - group["top1_fixed_index"].nunique()
            for _, group in matched_table.groupby("scenario_id", sort=False)
        )
    )
    all_duplicate_count = int(
        sum(
            len(group) - group["top1_fixed_index"].nunique()
            for _, group in table.groupby("scenario_id", sort=False)
        )
    )
    scores = table["calibrated_confidence"].to_numpy(float)
    return {
        "record_type": record_type,
        "dataset_group": table.get("dataset_group", pd.Series(["Real-like"])).iloc[0],
        "split": table["split"].iloc[0] if table["split"].nunique() == 1 else "Pooled",
        "strength": table["strength"].iloc[0] if table["strength"].nunique() == 1 else "All",
        "scenario_id": table["scenario_id"].iloc[0] if table["scenario_id"].nunique() == 1 else "Pooled",
        "method": table["method"].iloc[0],
        "n_rows": len(table),
        "common_n": int(np.count_nonzero(common)),
        "true_unmatched_n": int(np.count_nonzero(true_unmatched)),
        "true_candidate_recall": _divide(
            np.count_nonzero(table["true_in_candidate_set"] & common), np.count_nonzero(common)
        ),
        "top1_accuracy": _divide(np.count_nonzero(correct), np.count_nonzero(common)),
        "top3_recall": _divide(np.count_nonzero(table["top3_contains_true"]), np.count_nonzero(common)),
        "matched_precision": matched_precision,
        "matched_recall": matched_recall,
        "matched_f1": _f1(matched_precision, matched_recall),
        "unmatched_precision": unmatched_precision,
        "unmatched_recall": unmatched_recall,
        "unmatched_f1": _f1(unmatched_precision, unmatched_recall),
        "false_match_rate": _divide(np.count_nonzero(matched & ~correct), np.count_nonzero(matched)),
        "coverage": float(np.mean(matched)),
        "matched_n": int(np.count_nonzero(matched)),
        "ambiguous_n": int(np.count_nonzero(table["status"].to_numpy(str) == "ambiguous")),
        "predicted_unmatched_n": int(np.count_nonzero(predicted_unmatched)),
        "wrong_matched_n": int(np.count_nonzero(matched & ~correct)),
        "duplicate_destination_count": duplicate_count,
        "duplicate_destination_rate": _divide(duplicate_count, np.count_nonzero(matched)),
        "top1_duplicate_destination_count_all": all_duplicate_count,
        "top1_duplicate_destination_rate_all": _divide(all_duplicate_count, len(table)),
        "reciprocal_fraction_matched": float(table.loc[matched, "reciprocal"].mean()) if np.any(matched) else np.nan,
        "row_mass_ratio_median": float(np.nanmedian(table["row_mass_ratio"])),
        "candidate_concentration_median": float(np.nanmedian(table["normalized_top1_mass"])),
        "graph_support_median": float(np.nanmedian(table["top1_graph_support"])),
        "confidence_median": float(np.nanmedian(scores)),
        "transported_total_mass_mean": float(np.nanmean(table["transported_total_mass"]))
        if np.isfinite(table["transported_total_mass"]).any()
        else np.nan,
    }


def expected_calibration_error(table: pd.DataFrame, bins: int = 10) -> tuple[float, float]:
    scores = np.clip(table["calibrated_confidence"].to_numpy(float), 0.0, 1.0)
    truth = table["top1_correct"].to_numpy(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for index in range(bins):
        mask = (scores >= edges[index]) & (
            scores <= edges[index + 1] if index == bins - 1 else scores < edges[index + 1]
        )
        if np.any(mask):
            ece += float(np.mean(mask) * abs(np.mean(scores[mask]) - np.mean(truth[mask])))
    brier = float(np.mean((scores - truth) ** 2))
    return float(ece), brier


def precision_coverage_curve(table: pd.DataFrame) -> pd.DataFrame:
    scores = table["calibrated_confidence"].to_numpy(float)
    correct = table["top1_correct"].to_numpy(bool)
    common = table["is_common_cell"].to_numpy(bool)
    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[order]
    cumulative_correct = np.cumsum(correct[order])
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(scores) - 1]
    selected = ends + 1
    return pd.DataFrame(
        {
            "dataset_group": table.get("dataset_group", pd.Series(["Real-like"])).iloc[0],
            "split": table["split"].iloc[0] if table["split"].nunique() == 1 else "Pooled",
            "method": table["method"].iloc[0],
            "threshold": sorted_scores[ends],
            "selected_n": selected,
            "coverage": selected / len(table),
            "precision": cumulative_correct[ends] / selected,
            "matched_recall": cumulative_correct[ends] / max(np.count_nonzero(common), 1),
        }
    )


def empirical_coverage_at_precision(curve: pd.DataFrame, target: float) -> float:
    eligible = curve.loc[curve["precision"] >= target]
    return float(eligible["coverage"].max()) if not eligible.empty else float("nan")


def precision_at_coverage(curve: pd.DataFrame, target: float) -> float:
    eligible = curve.loc[curve["coverage"] >= target]
    if eligible.empty:
        return float("nan")
    row = eligible.iloc[(eligible["coverage"] - target).abs().argmin()]
    return float(row["precision"])


def calibration_summary(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    method: str,
    targets: tuple[float, ...] = (0.90, 0.95, 0.98),
) -> list[dict]:
    rows = []
    val = validation.loc[validation["method"] == method]
    held = test.loc[test["method"] == method]
    ece, brier = expected_calibration_error(held)
    for target in targets:
        selection = threshold_for_precision(val["calibrated_confidence"], val["top1_correct"], target)
        selected = held["calibrated_confidence"].to_numpy(float) >= selection["threshold"]
        precision = float(held.loc[selected, "top1_correct"].mean()) if np.any(selected) else np.nan
        coverage = float(np.mean(selected))
        rows.append(
            {
                "method": method,
                "target_precision": target,
                "validation_threshold": selection["threshold"],
                "validation_precision": selection["validation_precision"],
                "validation_coverage": selection["validation_coverage"],
                "test_precision_at_validation_threshold": precision,
                "test_coverage_at_validation_threshold": coverage,
                "test_target_achieved": bool(np.isfinite(precision) and precision >= target),
                "test_coverage_if_target_achieved": coverage
                if np.isfinite(precision) and precision >= target
                else np.nan,
                "test_ece": ece,
                "test_brier": brier,
            }
        )
    return rows


def graph_discrimination_rows(
    sample: CorrespondenceSample,
    scenario: dict,
    context: FixedCandidateContext,
) -> pd.DataFrame:
    true_index = sample.true_fixed_index
    rows = []
    for moving_index, candidates in enumerate(context.candidates.indices):
        for slot, fixed_index in enumerate(candidates):
            rows.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "split": scenario.get("split", "Control"),
                    "strength": scenario.get("strength", sample.difficulty),
                    "moving_index": moving_index,
                    "fixed_index": int(fixed_index),
                    "is_true_candidate": bool(true_index[moving_index] >= 0 and fixed_index == true_index[moving_index]),
                    "graph_support": float(context.graph_support[moving_index, slot]),
                    "position_cost": float(context.components["position"][moving_index, slot]),
                }
            )
    return pd.DataFrame(rows)


def candidate_source_diagnostics(
    sample: CorrespondenceSample,
    scenario: dict,
    context: FixedCandidateContext,
) -> pd.DataFrame:
    """Compare Graph ranking on Full-generated candidates and spatial top-30."""
    n, k = context.candidates.indices.shape
    full_cost = selected_cost(context, ("position", "graph"))
    full_order = np.argsort(full_cost, axis=1)
    full_top = np.take_along_axis(context.candidates.indices, full_order[:, :1], axis=1)[:, 0]

    distances = cdist(context.moving, context.fixed)
    spatial_indices = np.argsort(distances, axis=1)[:, :k]
    spatial_distance = np.take_along_axis(distances, spatial_indices, axis=1)
    position_raw = spatial_distance**2 / (2.0 * float(CANDIDATES["position_scale_um"]) ** 2)
    zeros = np.zeros_like(position_raw)
    spatial_candidates = CandidateSet(
        indices=spatial_indices,
        position_cost=position_raw,
        fingerprint_cost=zeros,
        region_cost=zeros,
        base_cost=float(CANDIDATES["weights_full"]["position"]) * position_raw,
        spatial_distance_um=spatial_distance,
        region_residual_um=spatial_distance,
        fingerprint_similarity=np.ones_like(position_raw),
        region_similarity=np.ones_like(position_raw),
    )
    spatial_support = graph_support_scores(context.fixed, context.moving, spatial_candidates)
    spatial_cost = spatial_candidates.base_cost + float(CANDIDATES["graph_weight"]) * (1.0 - spatial_support)
    spatial_order = np.argsort(spatial_cost, axis=1)
    spatial_top = np.take_along_axis(spatial_indices, spatial_order[:, :1], axis=1)[:, 0]

    common = sample.true_fixed_index >= 0
    rows = []
    for name, indices, support, top in (
        ("Full contextual candidate-30", context.candidates.indices, context.graph_support, full_top),
        ("Spatial candidate-30", spatial_indices, spatial_support, spatial_top),
    ):
        labels = (indices == sample.true_fixed_index[:, None]) & common[:, None]
        rows.append(
            {
                "scenario_id": scenario["scenario_id"],
                "split": scenario.get("split", "Control"),
                "strength": scenario.get("strength", sample.difficulty),
                "candidate_source": name,
                "candidate_recall_30": float(np.mean(np.any(labels[common], axis=1))),
                "position_graph_raw_top1_accuracy": float(np.mean(top[common] == sample.true_fixed_index[common])),
                "graph_candidate_auc": rank_auc(labels.ravel(), support.ravel()),
                "duplicate_destination_count": int(len(top) - len(np.unique(top))),
            }
        )
    return pd.DataFrame(rows)


def rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    positives, negatives = np.count_nonzero(labels), np.count_nonzero(~labels)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    for group in np.flatnonzero(counts > 1):
        ranks[inverse == group] = np.mean(ranks[inverse == group])
    return float((np.sum(ranks[labels]) - positives * (positives + 1) / 2) / (positives * negatives))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")

