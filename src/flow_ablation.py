from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import combinations
from typing import Mapping

import numpy as np


DATA_TERMS = ("density", "support", "structure")
REGULARIZATION_TERMS = (
    "smoothness",
    "magnitude",
    "tissue_boundary",
    "inverse_consistency",
    "jacobian_barrier",
    "soft_jacobian",
)
ALL_OBJECTIVE_TERMS = DATA_TERMS + REGULARIZATION_TERMS


@dataclass(frozen=True)
class FlowAblationConfig:
    """Optimization-channel switches; hard mathematical validity is intentionally absent."""

    use_density_term: bool = True
    use_support_term: bool = True
    use_structure_term: bool = True
    use_smoothness_regularization: bool = True
    use_magnitude_regularization: bool = True
    use_boundary_regularization: bool = True
    use_inverse_consistency: bool = True
    use_soft_jacobian_regularization: bool = True
    use_jacobian_barrier: bool = True

    def enabled_for(self, term: str) -> bool:
        attribute = {
            "density": "use_density_term",
            "support": "use_support_term",
            "structure": "use_structure_term",
            "smoothness": "use_smoothness_regularization",
            "magnitude": "use_magnitude_regularization",
            "tissue_boundary": "use_boundary_regularization",
            "inverse_consistency": "use_inverse_consistency",
            "soft_jacobian": "use_soft_jacobian_regularization",
            "jacobian_barrier": "use_jacobian_barrier",
        }[term]
        return bool(getattr(self, attribute))

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


def normalize_ablation_config(value: FlowAblationConfig | Mapping[str, object] | None) -> FlowAblationConfig:
    if value is None:
        return FlowAblationConfig()
    if isinstance(value, FlowAblationConfig):
        return value
    allowed = FlowAblationConfig.__dataclass_fields__
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown flow ablation flags: {sorted(unknown)}")
    return FlowAblationConfig(**{key: bool(item) for key, item in value.items()})


def ablation_preset(name: str, *, base: FlowAblationConfig | None = None) -> FlowAblationConfig:
    config = base or FlowAblationConfig()
    key = str(name).strip().lower().replace(" ", "_")
    disabled = {
        "no_density": "use_density_term",
        "-density": "use_density_term",
        "no_support": "use_support_term",
        "-support": "use_support_term",
        "no_structure": "use_structure_term",
        "-structure": "use_structure_term",
        "no_smoothness": "use_smoothness_regularization",
        "-smoothness": "use_smoothness_regularization",
        "no_magnitude": "use_magnitude_regularization",
        "-magnitude": "use_magnitude_regularization",
        "no_boundary": "use_boundary_regularization",
        "-boundary": "use_boundary_regularization",
        "no_soft_jacobian": "use_soft_jacobian_regularization",
        "-softjacobian": "use_soft_jacobian_regularization",
    }
    if key in {"full", "custom"}:
        return config
    if key in disabled:
        return replace(config, **{disabled[key]: False})
    if key == "stage_a_support_only":
        return replace(config, use_density_term=False, use_structure_term=False)
    if key == "stage_a_density_only":
        return replace(config, use_support_term=False, use_structure_term=False)
    if key == "stage_b_density_only":
        return replace(config, use_support_term=False, use_structure_term=False)
    if key == "stage_b_structure_only":
        return replace(config, use_density_term=False, use_support_term=False)
    raise ValueError(f"Unknown ablation preset: {name}")


def objective_decomposition(raw_values: Mapping[str, float], weights: Mapping[str, float], config: FlowAblationConfig) -> dict:
    terms = {}
    for term in ALL_OBJECTIVE_TERMS:
        raw = float(raw_values.get(term, 0.0))
        weight = float(weights.get(term, 0.0))
        enabled = config.enabled_for(term)
        weighted = raw * weight if enabled else 0.0
        terms[term] = {
            "group": "data" if term in DATA_TERMS else "regularization",
            "raw": raw,
            "raw_value": raw,
            "weight": weight,
            "weighted": weighted,
            "weighted_value": weighted,
            "enabled": enabled,
        }
    data_total = float(sum(terms[name]["weighted"] for name in DATA_TERMS))
    regularization_total = float(sum(terms[name]["weighted"] for name in REGULARIZATION_TERMS))
    total = data_total + regularization_total
    for term in terms.values():
        term["fraction_of_total"] = float(term["weighted"] / total) if total != 0 else 0.0
    return {
        "terms": terms,
        "weighted_data_total": data_total,
        "weighted_regularization_total": regularization_total,
        "weighted_total": total,
        "data_fraction": float(data_total / total) if total != 0 else 0.0,
        "regularization_fraction": float(regularization_total / total) if total != 0 else 0.0,
    }


def objective_rows(decomposition: Mapping[str, object], *, state: str, stage: str) -> list[dict]:
    rows = []
    for term, values in decomposition.get("terms", {}).items():
        rows.append({"state": state, "stage": stage, "term": term, **dict(values)})
    return rows


def vector_field_summary(field_x: np.ndarray, field_y: np.ndarray) -> dict[str, float]:
    magnitude = np.hypot(np.asarray(field_x, dtype=float), np.asarray(field_y, dtype=float))
    return {
        "median": float(np.median(magnitude)),
        "p95": float(np.percentile(magnitude, 95)),
        "max": float(np.max(magnitude)),
    }


def vector_cosine(a_x: np.ndarray, a_y: np.ndarray, b_x: np.ndarray, b_y: np.ndarray) -> float | None:
    a = np.column_stack([np.asarray(a_x).ravel(), np.asarray(a_y).ravel()])
    b = np.column_stack([np.asarray(b_x).ravel(), np.asarray(b_y).ravel()])
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.sum(a * b) / denominator) if denominator > 0 else None


def term_interaction_summary(components: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    summaries = {name: vector_field_summary(*field) for name, field in components.items()}
    cosines = {}
    for first, second in combinations((name for name in DATA_TERMS if name in components), 2):
        value = vector_cosine(*components[first], *components[second])
        cosines[f"{first}_vs_{second}"] = value
    return {"update_magnitude": summaries, "cosine_similarity": cosines}


def density_mismatch_diagnostics(
    fixed_impulses: np.ndarray,
    moving_impulses: np.ndarray,
    fixed_density: np.ndarray,
    moving_density: np.ndarray,
    tissue_weight: np.ndarray,
    *,
    fixed_point_count: int,
    moving_point_count: int,
    retain_arrays: bool = False,
) -> tuple[dict, dict[str, np.ndarray]]:
    residual = np.asarray(fixed_density, dtype=float) - np.asarray(moving_density, dtype=float)
    squared = residual**2
    weighted_squared = np.asarray(tissue_weight, dtype=float) * squared
    fixed_energy = float(np.mean(np.asarray(tissue_weight) * np.asarray(fixed_density) ** 2))
    mismatch = float(np.mean(weighted_squared) / max(fixed_energy, np.finfo(float).eps))
    summary = {
        "fixed_point_count": int(fixed_point_count),
        "moving_point_count": int(moving_point_count),
        "fixed_raster_mass_before_normalization": float(np.sum(fixed_impulses)),
        "moving_raster_mass_before_normalization": float(np.sum(moving_impulses)),
        "fixed_density_mass_after_normalization": float(np.sum(fixed_density)),
        "moving_density_mass_after_normalization": float(np.sum(moving_density)),
        "raw_residual_mean": float(np.mean(residual)),
        "raw_residual_rms": float(np.sqrt(np.mean(squared))),
        "raw_residual_p95_absolute": float(np.percentile(np.abs(residual), 95)),
        "tissue_weighted_residual_rms": float(np.sqrt(np.mean(weighted_squared))),
        "fixed_density_normalization_energy": fixed_energy,
        "final_normalized_density_mismatch": mismatch,
    }
    arrays = {}
    if retain_arrays:
        arrays = {
            "fixed_density": np.asarray(fixed_density),
            "moving_density": np.asarray(moving_density),
            "density_residual": residual,
            "squared_density_residual": squared,
            "tissue_weight": np.asarray(tissue_weight),
            "weighted_density_residual": np.sign(residual) * np.sqrt(weighted_squared),
        }
    return summary, arrays
