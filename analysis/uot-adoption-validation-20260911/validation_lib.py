"""Analysis-only fixed-configuration helpers for the UOT adoption validation.

Nothing in this module is imported by Workflow C/D or the Streamlit application.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

ROOT = Path("/mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu")
OUT = ROOT / "analysis/uot-adoption-validation-20260911"
PRIOR_UOT = Path("/mnt/mib-dbia08_1/users/noa.ganeko/analysis/cpd-uot-comparison-20260909")
PRIOR_DENSITY = Path("/mnt/mib-dbia08_1/users/noa.ganeko/analysis/density-vs-joint-20260909")
VENDOR = PRIOR_UOT / "vendor"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VENDOR))

import ot

from src.density_flow import (
    _checkpoint_improves_affine,
    _density_flow_trial_is_safe,
    _jacobian,
    _mutual_nearest_fraction,
    tissue_aware_density_flow_registration,
    joint_density_tissue_structure_registration,
)
from src.flow_ablation import FlowAblationConfig
from src.pointset_registration import point_bidirectional_distance_metrics
from src.registration_evaluation import sample_displacement_field


UOT_CONFIGS = {
    "UOT conservative": {
        "cost_scale_um": 10.0,
        "entropic_regularization": 0.1,
        "mass_relaxation_kl": 0.5,
        "minimum_row_mass_ratio": 0.25,
        "confidence_weighting": "min(row_mass / source_mass, 1)",
        "field_kernel": "linear",
        "field_smoothing": 10.0,
        "field_neighbors": 40,
        "solver": "sinkhorn_translation_invariant",
        "reg_type": "kl",
        "max_iterations": 2000,
        "stop_threshold": 1e-7,
        "source_trial": "coarse_09 / cost10_e0.1_m0.5",
    },
    "UOT primary": {
        "cost_scale_um": 20.0,
        "entropic_regularization": 0.1,
        "mass_relaxation_kl": 0.5,
        "minimum_row_mass_ratio": 0.25,
        "confidence_weighting": "min(row_mass / source_mass, 1)",
        "field_kernel": "linear",
        "field_smoothing": 10.0,
        "field_neighbors": 40,
        "solver": "sinkhorn_translation_invariant",
        "reg_type": "kl",
        "max_iterations": 2000,
        "stop_threshold": 1e-7,
        "source_trial": "coarse_04 / core_e0.1_m0.5",
    },
}


# Exact effective Density Safe configuration from the prior analysis record.
DENSITY_CONFIG = {
    "density_pixel_size": 2.0,
    "density_blur_scales": (8.0, 4.0, 2.0),
    "optimization_levels": 3,
    "iterations_per_level": 8,
    "learning_rate": 0.08,
    "update_smoothing_sigma": 4.0,
    "smoothness_weight": 0.1,
    "magnitude_weight": 0.002,
    "jacobian_barrier_weight": 1.0,
    "tissue_boundary_weight": 0.05,
    "inverse_consistency_weight": 0.0,
    "jacobian_min_threshold": 0.1,
    "jacobian_max_threshold": 3.0,
    "max_displacement": 35.0,
    "displacement_p95_limit": 30.0,
    "global_translation_initialization": "off",
    "objective_tolerance": 1e-12,
    "early_stopping_patience": 3,
    "point_metric_patience": 6,
    "max_backtracking_steps": 8,
    "minimum_absolute_median_improvement": 0.1,
    "minimum_relative_median_improvement": 0.005,
    "maximum_mutual_nearest_decrease": 0.005,
    "maximum_within_fraction_decrease": 0.01,
    "minimum_jacobian_p05": 0.8,
    "maximum_jacobian_p95": 1.25,
    "local_region_block_size": 100.0,
    "density_channel_weight": 1.0,
    "tissue_support_channel_weight": 0.0,
    "structure_channel_weight": 0.0,
    "soft_jacobian_weight": 0.0,
    "checkpoint_policy": "point_metric",
    "stage_a_max_absolute_median_worsening_um": 0.5,
    "stage_a_max_relative_median_worsening": 0.05,
    "stage_a_max_mutual_nearest_decrease": 0.02,
    "stage_a_max_within_fraction_decrease": 0.03,
    "detect_axis_reversal": True,
    "max_grid_side": 1024,
    "ablation_config": FlowAblationConfig(),
    "retain_research_diagnostic_fields": False,
    "dtype": "float64",
}


# Exact effective Joint trial 0005 values. None values deliberately preserve the
# implementation's recorded fallback behavior (Stage B support is 0.15 * 0.70).
JOINT_CONFIG = {
    "density_pixel_size": 2.0,
    "density_weight": 1.0,
    "support_weight": 0.70,
    "structure_weight": 0.35,
    "soft_jacobian_weight": 0.05,
    "stage_a_scales_um": (32.0, 16.0, 8.0),
    "stage_b_scales_um": (8.0, 4.0),
    "stage_a_iterations": 8,
    "stage_b_iterations": 10,
    "stage_a_learning_rate": 0.05,
    "stage_b_learning_rate": 0.05,
    "stage_a_update_smoothing": 6.0,
    "stage_b_update_smoothing": 3.0,
    "stage_a_density_weight": 0.35,
    "stage_a_support_weight": None,
    "stage_b_density_weight": None,
    "stage_b_structure_weight": None,
    "exploratory_max_displacement": 35.0,
    "exploratory_p95_displacement": 25.0,
    "exploratory_jacobian_min": 0.02,
    "exploratory_jacobian_max": 6.0,
    "stage_a_checkpoint_policy": "stage_objective",
    "stage_a_max_absolute_median_worsening_um": 0.5,
    "stage_a_max_relative_median_worsening": 0.05,
    "stage_a_max_mutual_nearest_decrease": 0.02,
    "stage_a_max_within_fraction_decrease": 0.03,
    "joint_preset": "Joint Safe",
    "stage_a_ablation_config": FlowAblationConfig(use_structure_term=False),
    "stage_b_ablation_config": FlowAblationConfig(),
    "retain_research_diagnostic_fields": False,
    "smoothness_weight": 0.1,
    "magnitude_weight": 0.002,
    "jacobian_barrier_weight": 1.0,
    "tissue_boundary_weight": 0.05,
    "inverse_consistency_weight": 0.0,
    "jacobian_min_threshold": 0.1,
    "jacobian_max_threshold": 3.0,
    "max_displacement": 35.0,
    "displacement_p95_limit": 30.0,
    "global_translation_initialization": "off",
    "objective_tolerance": 1e-12,
    "early_stopping_patience": 3,
    "point_metric_patience": 6,
    "max_backtracking_steps": 8,
    "minimum_absolute_median_improvement": 0.1,
    "minimum_relative_median_improvement": 0.005,
    "maximum_mutual_nearest_decrease": 0.005,
    "maximum_within_fraction_decrease": 0.01,
    "minimum_jacobian_p05": 0.8,
    "maximum_jacobian_p95": 1.25,
    "local_region_block_size": 100.0,
    "detect_axis_reversal": True,
    "max_grid_side": 1024,
    "dtype": "float64",
}


SAFETY = {
    "jacobian_min_threshold": 0.1,
    "jacobian_max_threshold": 3.0,
    "max_displacement": 35.0,
    "displacement_p95_limit": 30.0,
    "minimum_jacobian_p05": 0.8,
    "maximum_jacobian_p95": 1.25,
    "minimum_absolute_median_improvement": 0.1,
    "minimum_relative_median_improvement": 0.005,
    "maximum_mutual_nearest_decrease": 0.005,
    "maximum_within_fraction_decrease": 0.01,
}


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
    if isinstance(value, FlowAblationConfig):
        return value.to_dict()
    return value


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def point_metrics(fixed: np.ndarray, moving: np.ndarray) -> dict:
    values = point_bidirectional_distance_metrics(np.asarray(fixed), np.asarray(moving))
    values["mutual_nearest_fraction"] = _mutual_nearest_fraction(fixed, moving)
    return values


def field_stats(dx: np.ndarray, dy: np.ndarray, spacing: float) -> dict:
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    magnitude = np.hypot(dx, dy)
    jac = _jacobian(dx, dy, spacing)
    return {
        "finite": bool(np.isfinite(dx).all() and np.isfinite(dy).all() and np.isfinite(jac).all()),
        "displacement_median_um": float(np.median(magnitude)),
        "displacement_p95_um": float(np.percentile(magnitude, 95)),
        "displacement_max_um": float(np.max(magnitude)),
        "jacobian_min": float(np.min(jac)),
        "jacobian_p05": float(np.percentile(jac, 5)),
        "jacobian_median": float(np.median(jac)),
        "jacobian_p95": float(np.percentile(jac, 95)),
        "jacobian_max": float(np.max(jac)),
        "foldover_fraction": float(np.mean(jac <= 0.0)),
    }


def common_audit(
    fixed: np.ndarray,
    affine_moving: np.ndarray,
    attempted_moving: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    spacing: float,
) -> dict:
    before = point_metrics(fixed, affine_moving)
    after = point_metrics(fixed, attempted_moving)
    stats = field_stats(dx, dy, spacing)
    hard_state = {
        "finite": stats["finite"],
        "jacobian_min": stats["jacobian_min"],
        "jacobian_max": stats["jacobian_max"],
        "max_displacement": stats["displacement_max_um"],
        "p95_displacement": stats["displacement_p95_um"],
    }
    hard, reason = _density_flow_trial_is_safe(
        hard_state,
        jacobian_min_threshold=SAFETY["jacobian_min_threshold"],
        jacobian_max_threshold=SAFETY["jacobian_max_threshold"],
        max_displacement=SAFETY["max_displacement"],
        displacement_p95_limit=SAFETY["displacement_p95_limit"],
    )
    topology = bool(
        stats["finite"]
        and stats["foldover_fraction"] == 0.0
        and stats["jacobian_p05"] >= SAFETY["minimum_jacobian_p05"]
        and stats["jacobian_p95"] <= SAFETY["maximum_jacobian_p95"]
    )
    improvement = _checkpoint_improves_affine(
        after,
        affine_median=before["symmetric_median_distance"],
        affine_mutual_fraction=before["mutual_nearest_fraction"],
        minimum_absolute_improvement=SAFETY["minimum_absolute_median_improvement"],
        minimum_relative_improvement=SAFETY["minimum_relative_median_improvement"],
        maximum_mutual_decrease=SAFETY["maximum_mutual_nearest_decrease"],
        maximum_within_fraction_decrease=SAFETY["maximum_within_fraction_decrease"],
        affine_metrics=before,
    )
    reasons = []
    if not hard:
        reasons.append(reason or "hard_deformation_safety")
    if not topology:
        reasons.append("jacobian_percentile_or_foldover_gate")
    if not improvement:
        reasons.append("point_improvement_or_degradation_gate")
    return {
        "common_audit_pass": bool(hard and topology and improvement),
        "hard_safety_pass": bool(hard),
        "topology_percentiles_pass": topology,
        "point_improvement_gate_pass": bool(improvement),
        "rejection_reason": ";".join(filter(None, reasons)) or None,
        "before": before,
        "after": after,
        **stats,
    }


def fit_linear_rbf(
    anchors: np.ndarray,
    anchor_displacements: np.ndarray,
    moving: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    *,
    smoothing: float,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    interpolator = RBFInterpolator(
        anchors,
        anchor_displacements,
        kernel="linear",
        smoothing=float(smoothing),
        neighbors=min(int(neighbors), len(anchors)),
    )
    registered = moving + interpolator(moving)
    query = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    field = interpolator(query).reshape(*grid_x.shape, 2)
    return registered, field[..., 0], field[..., 1]


def run_uot(
    fixed: np.ndarray,
    moving: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    spacing: float,
    method: str,
    *,
    metric_fixed: np.ndarray | None = None,
    metric_moving_indices: np.ndarray | None = None,
) -> dict:
    config = UOT_CONFIGS[method]
    fixed = np.asarray(fixed, dtype=np.float64)
    moving = np.asarray(moving, dtype=np.float64)
    denominator = float(max(len(moving), len(fixed)))
    source_mass = np.ones(len(moving), dtype=np.float64) / denominator
    target_mass = np.ones(len(fixed), dtype=np.float64) / denominator
    cost = cdist(moving, fixed, metric="sqeuclidean") / (2.0 * config["cost_scale_um"] ** 2)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan, log = ot.unbalanced.sinkhorn_unbalanced(
            source_mass,
            target_mass,
            cost,
            reg=config["entropic_regularization"],
            reg_m=config["mass_relaxation_kl"],
            method=config["solver"],
            reg_type=config["reg_type"],
            numItermax=config["max_iterations"],
            stopThr=config["stop_threshold"],
            log=True,
        )
    errors = [float(value) for value in log.get("err", [])]
    final_error = errors[-1] if errors else None
    plan_valid = bool(
        np.isfinite(plan).all()
        and np.all(plan >= 0.0)
        and final_error is not None
        and final_error <= config["stop_threshold"]
    )
    if not plan_valid:
        raise RuntimeError(f"invalid UOT plan: final_error={final_error}")
    row_mass = np.sum(plan, axis=1)
    confidence = row_mass / source_mass
    keep = confidence >= config["minimum_row_mass_ratio"]
    if np.count_nonzero(keep) < 3:
        raise RuntimeError("fewer than three UOT anchors retained")
    barycentric = (plan[keep] @ fixed) / row_mass[keep, None]
    raw_displacement = barycentric - moving[keep]
    scaled_displacement = raw_displacement * np.minimum(confidence[keep], 1.0)[:, None]
    attempted_points, attempted_x, attempted_y = fit_linear_rbf(
        moving[keep],
        scaled_displacement,
        moving,
        grid_x,
        grid_y,
        smoothing=config["field_smoothing"],
        neighbors=config["field_neighbors"],
    )
    audit_fixed = fixed if metric_fixed is None else np.asarray(metric_fixed, dtype=np.float64)
    if metric_moving_indices is None:
        audit_moving = moving
        audit_attempted = attempted_points
    else:
        audit_moving = moving[metric_moving_indices]
        audit_attempted = attempted_points[metric_moving_indices]
    audit = common_audit(
        audit_fixed, audit_moving, audit_attempted, attempted_x, attempted_y, spacing
    )
    applied = audit["common_audit_pass"]
    if applied:
        points, field_x, field_y = attempted_points, attempted_x, attempted_y
    else:
        points = moving.copy()
        field_x = np.zeros_like(grid_x)
        field_y = np.zeros_like(grid_y)
    kept_confidence = confidence[keep]
    diagnostics = {
        "transported_mass": float(np.sum(plan)),
        "source_total_mass": float(np.sum(source_mass)),
        "target_total_mass": float(np.sum(target_mass)),
        "anchors_kept": int(np.count_nonzero(keep)),
        "anchors_discarded": int(np.count_nonzero(~keep)),
        "anchor_retained_fraction": float(np.mean(keep)),
        "anchor_discarded_fraction": float(np.mean(~keep)),
        "confidence_min": float(np.min(kept_confidence)),
        "confidence_p05": float(np.percentile(kept_confidence, 5)),
        "confidence_median": float(np.median(kept_confidence)),
        "confidence_p95": float(np.percentile(kept_confidence, 95)),
        "confidence_max": float(np.max(kept_confidence)),
        "solver_iterations": len(errors),
        "solver_final_error": final_error,
        "solver_warnings": [str(item.message) for item in caught],
    }
    return {
        "method": method,
        "points": points,
        "field_x": field_x,
        "field_y": field_y,
        "attempted_points": attempted_points,
        "attempted_field_x": attempted_x,
        "attempted_field_y": attempted_y,
        "applied": bool(applied),
        "rejection_reason": audit["rejection_reason"],
        "audit": audit,
        "diagnostics": diagnostics,
        "runtime_seconds": time.perf_counter() - started,
    }


def _result_to_dict(method: str, result, fixed: np.ndarray, moving: np.ndarray) -> dict:
    attempted_x = np.asarray(result.attempted_displacement_x, dtype=float)
    attempted_y = np.asarray(result.attempted_displacement_y, dtype=float)
    attempted_points = np.asarray(result.attempted_transformed_points, dtype=float)
    attempted_audit = common_audit(
        fixed, moving, attempted_points, attempted_x, attempted_y, result.grid_spacing
    )
    applied_points = np.asarray(result.transformed_points, dtype=float)
    applied_x = np.asarray(result.displacement_x, dtype=float)
    applied_y = np.asarray(result.displacement_y, dtype=float)
    audit = common_audit(
        fixed, moving, applied_points, applied_x, applied_y, result.grid_spacing
    )
    # Preserve the implementation's application decision. The explicit audit
    # should agree; disagreement is retained as a diagnostic, not hidden.
    return {
        "method": method,
        "points": applied_points,
        "field_x": applied_x,
        "field_y": applied_y,
        "attempted_points": attempted_points,
        "attempted_field_x": attempted_x,
        "attempted_field_y": attempted_y,
        "applied": bool(result.applied),
        "rejection_reason": result.rejection_reason,
        "audit": audit,
        "attempted_audit": attempted_audit,
        "application_audit_agree": bool(result.applied) == bool(audit["common_audit_pass"]),
        "diagnostics": {},
        "runtime_seconds": float((result.metrics or {}).get("runtime_breakdown", {}).get("total_runtime_seconds", np.nan)),
    }


def run_density(sample, *, device: str) -> dict:
    config = dict(DENSITY_CONFIG)
    config.update(bounds=sample.bounds, device=device)
    started = time.perf_counter()
    result = tissue_aware_density_flow_registration(sample.fixed_points, sample.moving_points, **config)
    output = _result_to_dict("Density Flow baseline", result, sample.fixed_points, sample.moving_points)
    output["runtime_seconds"] = time.perf_counter() - started
    return output


def run_joint(sample, *, device: str) -> dict:
    config = dict(JOINT_CONFIG)
    config.update(bounds=sample.bounds, device=device)
    started = time.perf_counter()
    result = joint_density_tissue_structure_registration(
        sample.fixed_points,
        sample.moving_points,
        affine_he_image=sample.moving_image,
        affine_he_tissue_mask=sample.moving_tissue_mask,
        affine_he_metadata=sample.metadata,
        **config,
    )
    output = _result_to_dict("Joint Flow", result, sample.fixed_points, sample.moving_points)
    output["runtime_seconds"] = time.perf_counter() - started
    return output


def affine_result(sample) -> dict:
    zeros = np.zeros_like(sample.grid_x)
    stats = field_stats(zeros, zeros, sample.spacing)
    return {
        "method": "Affine only",
        "points": sample.moving_points.copy(),
        "field_x": zeros,
        "field_y": zeros.copy(),
        "attempted_points": sample.moving_points.copy(),
        "attempted_field_x": zeros,
        "attempted_field_y": zeros.copy(),
        "applied": True,
        "rejection_reason": None,
        "audit": {
            "common_audit_pass": True,
            "hard_safety_pass": True,
            "topology_percentiles_pass": True,
            "point_improvement_gate_pass": None,
            "rejection_reason": None,
            "before": point_metrics(sample.fixed_points, sample.moving_points),
            "after": point_metrics(sample.fixed_points, sample.moving_points),
            **stats,
        },
        "diagnostics": {},
        "runtime_seconds": 0.0,
    }


@dataclass
class SyntheticSample:
    fixed_points: np.ndarray
    moving_points: np.ndarray
    matched_moving_points: np.ndarray
    matched_fixed_targets: np.ndarray
    unmatched_moving_points: np.ndarray
    independent_moving_landmarks: np.ndarray
    independent_fixed_landmarks: np.ndarray
    moving_image: np.ndarray
    moving_tissue_mask: np.ndarray
    metadata: dict
    grid_x: np.ndarray
    grid_y: np.ndarray
    true_x: np.ndarray
    true_y: np.ndarray
    true_jacobian: np.ndarray
    ground_truth_evaluation_mask: np.ndarray
    bounds: tuple[float, float, float, float]
    spacing: float
    known_affine_matrix: np.ndarray
    known_affine_translation: np.ndarray
    affine_reconstruction_error_max_um: float
    seed: int
    amplitude_um: float
    missing_fraction: float
    unmatched_fraction: float
    overlap_fraction: float
    noise_um: float


def _true_field(grid_x: np.ndarray, grid_y: np.ndarray, amplitude: float) -> tuple[np.ndarray, np.ndarray]:
    min_x, max_x = float(grid_x.min()), float(grid_x.max())
    min_y, max_y = float(grid_y.min()), float(grid_y.max())
    width, height = max_x - min_x, max_y - min_y
    xn = (grid_x - min_x) / width
    yn = (grid_y - min_y) / height
    envelope = np.sin(np.pi * xn) * np.sin(np.pi * yn)
    local = np.exp(-0.5 * (((xn - 0.63) / 0.18) ** 2 + ((yn - 0.42) / 0.22) ** 2))
    dx = amplitude * envelope * (0.62 * np.sin(2 * np.pi * yn) + 0.38 * local)
    dy = amplitude * envelope * (0.48 * np.sin(2 * np.pi * xn + np.pi / 5) - 0.24 * local)
    return dx, dy


def _sample_field(points: np.ndarray, dx: np.ndarray, dy: np.ndarray, spacing: float) -> np.ndarray:
    cols = points[:, 0] / spacing
    rows = points[:, 1] / spacing
    return np.column_stack([
        map_coordinates(dx, [rows, cols], order=1, mode="nearest"),
        map_coordinates(dy, [rows, cols], order=1, mode="nearest"),
    ])


def _inverse_points(target: np.ndarray, dx: np.ndarray, dy: np.ndarray, spacing: float) -> np.ndarray:
    moving = np.asarray(target, dtype=float).copy()
    for _ in range(30):
        moving = target - _sample_field(moving, dx, dy, spacing)
    return moving


def _sample_ellipse_points(rng: np.random.Generator, count: int, extent: float) -> np.ndarray:
    center = np.array([0.50 * extent, 0.51 * extent])
    radii = np.array([0.43 * extent, 0.38 * extent])
    cluster_centers = np.array([
        [0.29, 0.35], [0.54, 0.67], [0.72, 0.38], [0.43, 0.49]
    ]) * extent
    points = []
    while len(points) < count:
        if rng.random() < 0.72:
            candidate = cluster_centers[rng.integers(len(cluster_centers))] + rng.normal(0, 0.075 * extent, 2)
        else:
            candidate = rng.uniform(0.06 * extent, 0.94 * extent, 2)
        if np.sum(((candidate - center) / radii) ** 2) <= 1.0:
            points.append(candidate)
    return np.asarray(points, dtype=float)


def _render_he(points: np.ndarray, mask: np.ndarray, spacing: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 7919)
    height, width = mask.shape
    yy, xx = np.indices(mask.shape, dtype=float)
    base = np.full((height, width, 3), 242.0, dtype=float)
    texture = 7.0 * np.sin(xx / 13.0) + 5.0 * np.cos(yy / 17.0)
    base[mask, 0] = 190.0 + texture[mask]
    base[mask, 1] = 130.0 + 0.6 * texture[mask]
    base[mask, 2] = 175.0 + 0.4 * texture[mask]
    nuclei = np.zeros(mask.shape, dtype=float)
    for x, y in points:
        col = int(round(x / spacing))
        row = int(round(y / spacing))
        if 0 <= row < height and 0 <= col < width:
            nuclei[row, col] += rng.uniform(0.7, 1.0)
    nuclei = gaussian_filter(nuclei, 0.85)
    if nuclei.max() > 0:
        nuclei /= nuclei.max()
    base[..., 0] -= 125.0 * nuclei
    base[..., 1] -= 100.0 * nuclei
    base[..., 2] -= 80.0 * nuclei
    return np.clip(base, 0, 255).astype(np.uint8)


def generate_synthetic_sample(
    *,
    seed: int,
    amplitude_um: float,
    missing_fraction: float,
    unmatched_fraction: float,
    overlap_fraction: float,
    noise_um: float,
    size: int = 128,
    spacing: float = 2.0,
    n_latent: int = 320,
) -> SyntheticSample:
    rng = np.random.default_rng(seed)
    axes = np.arange(size, dtype=float) * spacing
    grid_x, grid_y = np.meshgrid(axes, axes)
    extent = float(axes[-1])
    true_x, true_y = _true_field(grid_x, grid_y, amplitude_um)
    latent_fixed = _sample_ellipse_points(rng, n_latent, extent)
    aligned_moving_all = _inverse_points(latent_fixed, true_x, true_y, spacing)

    overlap_cutoff = float(np.quantile(latent_fixed[:, 0], overlap_fraction)) if overlap_fraction < 1 else np.inf
    in_overlap = latent_fixed[:, 0] <= overlap_cutoff
    randomly_retained = rng.random(n_latent) >= missing_fraction
    keep = in_overlap & randomly_retained
    matched_fixed = latent_fixed[keep]
    matched_moving_clean = aligned_moving_all[keep]
    matched_moving = matched_moving_clean + rng.normal(0.0, noise_um, matched_moving_clean.shape)

    n_unmatched = int(round(max(len(matched_moving), 1) * unmatched_fraction))
    unmatched = _sample_ellipse_points(rng, n_unmatched, extent) if n_unmatched else np.empty((0, 2))
    moving = np.vstack([matched_moving, unmatched])
    order = rng.permutation(len(moving))
    moving = moving[order]

    # Fixed observations include all latent points plus a small section-specific
    # population, ensuring unequal counts without inventing correspondences.
    fixed_added = _sample_ellipse_points(rng, max(8, n_latent // 20), extent)
    fixed = np.vstack([latent_fixed, fixed_added])

    landmark_axis = np.linspace(0.18 * extent, 0.82 * extent, 6)
    candidate_landmarks = np.array([(x, y) for y in landmark_axis for x in landmark_axis])
    center = np.array([0.50 * extent, 0.51 * extent])
    radii = np.array([0.43 * extent, 0.38 * extent])
    landmark_inside = np.sum(((candidate_landmarks - center) / radii) ** 2, axis=1) <= 0.90
    fixed_landmarks = candidate_landmarks[landmark_inside]
    moving_landmarks = _inverse_points(fixed_landmarks, true_x, true_y, spacing)

    # Construct and exactly recover a nontrivial known affine before any fine
    # method. Fine registration therefore starts from identical aligned points.
    theta = np.deg2rad(3.0)
    affine_matrix = np.array([
        [0.985 * np.cos(theta), -1.015 * np.sin(theta) + 0.012],
        [0.985 * np.sin(theta), 1.015 * np.cos(theta)],
    ])
    affine_translation = np.array([11.0, -7.0])
    raw_moving = (moving - affine_translation) @ np.linalg.inv(affine_matrix).T
    reconstructed = raw_moving @ affine_matrix.T + affine_translation
    affine_error = float(np.max(np.linalg.norm(reconstructed - moving, axis=1)))

    yy, xx = np.indices((size, size), dtype=float)
    world_x, world_y = xx * spacing, yy * spacing
    mask = (((world_x - 0.50 * extent) / (0.45 * extent)) ** 2 +
            ((world_y - 0.51 * extent) / (0.40 * extent)) ** 2 <= 1.0)
    evaluation_mask = mask.copy()
    if overlap_fraction < 1.0:
        mask &= world_x <= overlap_cutoff + 0.05 * extent
    moving_image = _render_he(moving, mask, spacing, seed)
    metadata = {
        "bounds_um": [0.0, 0.0, extent, extent],
        "output_pixel_size_um": spacing,
        "output_origin": "upper-left",
        "width": size,
        "height": size,
        "row0_world_y": 0.0,
        "col0_world_x": 0.0,
    }
    return SyntheticSample(
        fixed_points=fixed,
        moving_points=reconstructed,
        matched_moving_points=matched_moving,
        matched_fixed_targets=matched_fixed,
        unmatched_moving_points=unmatched,
        independent_moving_landmarks=moving_landmarks,
        independent_fixed_landmarks=fixed_landmarks,
        moving_image=moving_image,
        moving_tissue_mask=mask,
        metadata=metadata,
        grid_x=grid_x,
        grid_y=grid_y,
        true_x=true_x,
        true_y=true_y,
        true_jacobian=_jacobian(true_x, true_y, spacing),
        ground_truth_evaluation_mask=evaluation_mask,
        bounds=(0.0, 0.0, extent, extent),
        spacing=spacing,
        known_affine_matrix=affine_matrix,
        known_affine_translation=affine_translation,
        affine_reconstruction_error_max_um=affine_error,
        seed=seed,
        amplitude_um=amplitude_um,
        missing_fraction=missing_fraction,
        unmatched_fraction=unmatched_fraction,
        overlap_fraction=overlap_fraction,
        noise_um=noise_um,
    )


def _quantiles(values: np.ndarray, prefix: str) -> dict:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {f"{prefix}_{name}": np.nan for name in ("median", "p95", "max")}
    return {
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_max": float(np.max(values)),
    }


def evaluate_synthetic(sample: SyntheticSample, result: dict) -> dict:
    field_x = np.asarray(result["field_x"], dtype=float)
    field_y = np.asarray(result["field_y"], dtype=float)
    attempted_x = np.asarray(result["attempted_field_x"], dtype=float)
    attempted_y = np.asarray(result["attempted_field_y"], dtype=float)
    field_error_global = np.hypot(field_x - sample.true_x, field_y - sample.true_y)
    attempted_error_global = np.hypot(attempted_x - sample.true_x, attempted_y - sample.true_y)
    evaluation_mask = np.asarray(sample.ground_truth_evaluation_mask, dtype=bool)
    field_error = field_error_global[evaluation_mask]
    attempted_error = attempted_error_global[evaluation_mask]
    estimated_jac = _jacobian(field_x, field_y, sample.spacing)
    jac_error = np.abs(estimated_jac - sample.true_jacobian)[evaluation_mask]

    landmark_disp = sample_displacement_field(
        sample.independent_moving_landmarks,
        field_x,
        field_y,
        bounds=sample.bounds,
        spacing=sample.spacing,
    )
    landmark_after = sample.independent_moving_landmarks + landmark_disp
    landmark_tre = np.linalg.norm(landmark_after - sample.independent_fixed_landmarks, axis=1)

    matched_disp = sample_displacement_field(
        sample.matched_moving_points,
        field_x,
        field_y,
        bounds=sample.bounds,
        spacing=sample.spacing,
    )
    matched_after = sample.matched_moving_points + matched_disp
    matched_tre = np.linalg.norm(matched_after - sample.matched_fixed_targets, axis=1)
    true_matched_disp = _sample_field(sample.matched_moving_points, sample.true_x, sample.true_y, sample.spacing)
    matched_disp_error = np.linalg.norm(matched_disp - true_matched_disp, axis=1)

    if len(sample.unmatched_moving_points):
        unmatched_disp = sample_displacement_field(
            sample.unmatched_moving_points,
            field_x,
            field_y,
            bounds=sample.bounds,
            spacing=sample.spacing,
        )
        unmatched_movement = np.linalg.norm(unmatched_disp, axis=1)
    else:
        unmatched_movement = np.empty(0)

    internal = point_metrics(sample.fixed_points, np.asarray(result["points"]))
    stats = field_stats(field_x, field_y, sample.spacing)
    attempted_stats = field_stats(attempted_x, attempted_y, sample.spacing)
    diagnostics = result.get("diagnostics", {})
    return {
        "method": result["method"],
        "seed": sample.seed,
        "amplitude_um": sample.amplitude_um,
        "missing_fraction": sample.missing_fraction,
        "unmatched_fraction": sample.unmatched_fraction,
        "overlap_fraction": sample.overlap_fraction,
        "noise_um": sample.noise_um,
        "fixed_n": len(sample.fixed_points),
        "moving_n": len(sample.moving_points),
        "matched_n": len(sample.matched_moving_points),
        "unmatched_n": len(sample.unmatched_moving_points),
        "independent_landmark_n": len(sample.independent_moving_landmarks),
        "known_affine_reconstruction_error_max_um": sample.affine_reconstruction_error_max_um,
        "applied": bool(result["applied"]),
        "common_audit_pass": bool(result["audit"]["common_audit_pass"]),
        "hard_safety_pass": bool(result["audit"]["hard_safety_pass"]),
        "topology_percentiles_pass": bool(result["audit"]["topology_percentiles_pass"]),
        "point_improvement_gate_pass": result["audit"]["point_improvement_gate_pass"],
        "rejection_reason": result.get("rejection_reason"),
        "application_audit_agree": result.get("application_audit_agree", True),
        "symmetric_median_nn_um": internal["symmetric_median_distance"],
        "moving_to_fixed_median_nn_um": internal["he_to_geojson_median_distance"],
        "fixed_to_moving_median_nn_um": internal["geojson_to_he_median_distance"],
        "within3_fraction": internal["symmetric_within_3"],
        "within5_fraction": internal["symmetric_within_5"],
        "within10_fraction": internal["symmetric_within_10"],
        "mutual_nn_fraction": internal["mutual_nearest_fraction"],
        **stats,
        **_quantiles(field_error, "field_error_um"),
        "endpoint_error_mean_um": float(np.mean(field_error)),
        **_quantiles(field_error_global, "field_error_global_um"),
        **_quantiles(attempted_error, "attempted_field_error_um"),
        **{f"attempted_{key}": value for key, value in attempted_stats.items()},
        **_quantiles(jac_error, "jacobian_abs_error"),
        **_quantiles(landmark_tre, "independent_landmark_tre_um"),
        **_quantiles(matched_tre, "known_matched_point_tre_um"),
        **_quantiles(matched_disp_error, "matched_displacement_error_um"),
        **_quantiles(unmatched_movement, "unmatched_movement_um"),
        "runtime_seconds": result["runtime_seconds"],
        "transported_mass": diagnostics.get("transported_mass"),
        "anchors_kept": diagnostics.get("anchors_kept"),
        "anchors_discarded": diagnostics.get("anchors_discarded"),
        "anchor_retained_fraction": diagnostics.get("anchor_retained_fraction"),
        "anchor_discarded_fraction": diagnostics.get("anchor_discarded_fraction"),
        "confidence_min": diagnostics.get("confidence_min"),
        "confidence_p05": diagnostics.get("confidence_p05"),
        "confidence_median": diagnostics.get("confidence_median"),
        "confidence_p95": diagnostics.get("confidence_p95"),
        "confidence_max": diagnostics.get("confidence_max"),
        "solver_final_error": diagnostics.get("solver_final_error"),
    }


def fixed_config_record() -> dict:
    primary_source = json.loads(
        (PRIOR_UOT / "trials/uot/coarse_04/result.json").read_text(encoding="utf-8")
    )["parameters"]
    conservative_source = json.loads(
        (PRIOR_UOT / "trials/uot/coarse_09/result.json").read_text(encoding="utf-8")
    )["parameters"]
    joint_source = json.loads(
        (PRIOR_DENSITY / "joint_exact_config.json").read_text(encoding="utf-8")
    )
    density_source = json.loads(
        (PRIOR_DENSITY / "density_effective_parameters.json").read_text(encoding="utf-8")
    )
    return {
        "source_of_truth": {
            "uot_report": str(PRIOR_UOT / "REPORT.md"),
            "uot_trials_csv": str(PRIOR_UOT / "uot_trials.csv"),
            "uot_primary_result": str(PRIOR_UOT / "trials/uot/coarse_04/result.json"),
            "uot_conservative_result": str(PRIOR_UOT / "trials/uot/coarse_09/result.json"),
            "joint_exact_config": str(PRIOR_DENSITY / "joint_exact_config.json"),
            "density_exact_config": str(PRIOR_DENSITY / "density_effective_parameters.json"),
        },
        "uot": UOT_CONFIGS,
        "density": DENSITY_CONFIG,
        "joint": JOINT_CONFIG,
        "verbatim_source_records": {
            "uot_primary_parameters": primary_source,
            "uot_conservative_parameters": conservative_source,
            "joint_trial_0005": joint_source,
            "density_baseline": density_source,
        },
        "common_safety": SAFETY,
        "stage_b_effective_support_weight": 0.15 * JOINT_CONFIG["support_weight"],
        "parameter_tuning_performed": False,
    }
