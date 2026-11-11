"""Reproduce the two frozen UOT conditions and assemble the single real dataset audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from validation_lib import (
    OUT,
    PRIOR_DENSITY,
    PRIOR_UOT,
    common_audit,
    field_stats,
    fixed_config_record,
    point_metrics,
    run_uot,
    save_json,
)

from src.workflow_c_result import load_workflow_c_result_artifact


DATASET = "serial_section_pair_20260904"
INPUT = Path(
    "/mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu/"
    "sweeps/real-grid27-20260908-v2/input"
)
JOINT_ARTIFACT = PRIOR_DENSITY / "joint_workflow_c_registration_result.zip"
DENSITY_ARTIFACT = PRIOR_DENSITY / "density_run/result/workflow_c_registration_result.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_method(name: str, artifact_path: Path, metric_fixed, metric_indices) -> dict:
    artifact = load_workflow_c_result_artifact(artifact_path.read_bytes())
    arrays = artifact.arrays
    affine = np.asarray(arrays["affine_he_points"], dtype=float)
    attempted = np.asarray(arrays["attempted_registered_he_points"], dtype=float)
    applied = np.asarray(arrays["applied_registered_he_points"], dtype=float)
    attempted_x = np.asarray(arrays["attempted_displacement_x"], dtype=float)
    attempted_y = np.asarray(arrays["attempted_displacement_y"], dtype=float)
    applied_x = np.asarray(arrays["applied_displacement_x"], dtype=float)
    applied_y = np.asarray(arrays["applied_displacement_y"], dtype=float)
    spacing = float(artifact.field_spacing)
    audit = common_audit(
        metric_fixed,
        affine[metric_indices],
        attempted[metric_indices],
        attempted_x,
        attempted_y,
        spacing,
    )
    return {
        "method": name,
        "points": applied,
        "field_x": applied_x,
        "field_y": applied_y,
        "attempted_points": attempted,
        "attempted_field_x": attempted_x,
        "attempted_field_y": attempted_y,
        "applied": bool(artifact.manifest["fine_applied"]),
        "rejection_reason": None if artifact.manifest["fine_applied"] else "artifact_fine_rejected",
        "audit": audit,
        "runtime_seconds": np.nan,
        "diagnostics": {},
        "artifact": artifact,
    }


def affine_method(affine, grid_x, grid_y, metric_fixed, metric_indices, spacing) -> dict:
    zeros = np.zeros_like(grid_x)
    metrics = point_metrics(metric_fixed, affine[metric_indices])
    return {
        "method": "Affine only",
        "points": affine,
        "field_x": zeros,
        "field_y": np.zeros_like(grid_y),
        "applied": True,
        "rejection_reason": None,
        "audit": {
            "common_audit_pass": True,
            "hard_safety_pass": True,
            "topology_percentiles_pass": True,
            "point_improvement_gate_pass": None,
            "before": metrics,
            "after": metrics,
            **field_stats(zeros, zeros, spacing),
        },
        "runtime_seconds": 0.0,
        "diagnostics": {},
    }


def metrics_row(result: dict, metric_fixed, metric_indices, spacing: float) -> dict:
    audit = result["audit"]
    metrics = point_metrics(metric_fixed, result["points"][metric_indices])
    stats = field_stats(result["field_x"], result["field_y"], spacing)
    diagnostics = result.get("diagnostics", {})
    return {
        "dataset": DATASET,
        "method": result["method"],
        "fixed_n": len(metric_fixed),
        "moving_n": len(metric_indices),
        "symmetric_median_nn_um": metrics["symmetric_median_distance"],
        "moving_to_fixed_median_nn_um": metrics["he_to_geojson_median_distance"],
        "fixed_to_moving_median_nn_um": metrics["geojson_to_he_median_distance"],
        "within3_fraction": metrics["symmetric_within_3"],
        "within5_fraction": metrics["symmetric_within_5"],
        "within10_fraction": metrics["symmetric_within_10"],
        "mutual_nn_fraction": metrics["mutual_nearest_fraction"],
        **stats,
        "hard_safety_pass": audit["hard_safety_pass"],
        "topology_percentiles_pass": audit["topology_percentiles_pass"],
        "point_improvement_gate_pass": audit["point_improvement_gate_pass"],
        "common_audit_pass": bool(audit["common_audit_pass"]) if result["method"] != "Affine only" else True,
        "applied": result["applied"],
        "rejection_reason": result["rejection_reason"] or audit.get("rejection_reason"),
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
        "runtime_seconds_shared_not_benchmark": result.get("runtime_seconds"),
        "metric_interpretation": "nearest-neighbor point-set proxy; not ground-truth correspondence accuracy",
    }


def manual_candidates(results: dict, fixed: np.ndarray, affine: np.ndarray) -> pd.DataFrame:
    uot = results["UOT primary"]["points"]
    conservative = results["UOT conservative"]["points"]
    joint = results["Joint Flow"]["points"]
    uot_disp = np.linalg.norm(uot - affine, axis=1)
    conservative_disp = np.linalg.norm(conservative - affine, axis=1)
    joint_disp = np.linalg.norm(joint - affine, axis=1)
    difference = np.linalg.norm(uot - joint, axis=1)
    conservative_difference = np.linalg.norm(conservative - joint, axis=1)
    review_score = np.maximum(difference, conservative_difference)
    ranking = np.argsort(review_score)[::-1]
    selected = []
    for index in ranking:
        if all(np.linalg.norm(affine[index] - affine[other]) >= 65.0 for other in selected):
            selected.append(int(index))
        if len(selected) == 12:
            break
    rows = []
    review_dir = OUT / "manual_review_candidates"
    for region_id, index in enumerate(selected, start=1):
        center = affine[index]
        radius = 55.0
        fixed_mask = np.linalg.norm(fixed - center, axis=1) <= radius
        moving_mask = np.linalg.norm(affine - center, axis=1) <= radius
        row = {
            "region": f"region_{region_id:02d}",
            "moving_index": index,
            "center_x_um": center[0],
            "center_y_um": center[1],
            "review_score_max_uot_joint_difference_um": review_score[index],
            "uot_primary_joint_position_difference_um": difference[index],
            "uot_conservative_joint_position_difference_um": conservative_difference[index],
            "uot_primary_displacement_um": uot_disp[index],
            "uot_conservative_displacement_um": conservative_disp[index],
            "joint_displacement_um": joint_disp[index],
            "fixed_points_in_view": int(np.count_nonzero(fixed_mask)),
            "moving_points_in_view": int(np.count_nonzero(moving_mask)),
        }
        rows.append(row)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(fixed[fixed_mask, 0], fixed[fixed_mask, 1], s=14, c="black", alpha=0.55, label="Fixed")
        ax.scatter(affine[moving_mask, 0], affine[moving_mask, 1], s=12, c="0.70", alpha=0.45, label="Affine")
        ax.scatter(conservative[moving_mask, 0], conservative[moving_mask, 1], s=15, c="#009E73", alpha=0.75, label="UOT conservative")
        ax.scatter(uot[moving_mask, 0], uot[moving_mask, 1], s=15, c="#0072B2", alpha=0.75, label="UOT primary")
        ax.scatter(joint[moving_mask, 0], joint[moving_mask, 1], s=15, c="#D55E00", alpha=0.75, label="Joint")
        ax.plot([uot[index, 0], joint[index, 0]], [uot[index, 1], joint[index, 1]], color="#CC79A7", lw=2.5)
        ax.plot([conservative[index, 0], joint[index, 0]], [conservative[index, 1], joint[index, 1]], color="#F0E442", lw=2.0)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_aspect("equal")
        ax.set_title(f"Blind-review candidate {region_id:02d}")
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(review_dir / f"region_{region_id:02d}.png", dpi=160)
        plt.close(fig)
    table = pd.DataFrame(rows)
    table.to_csv(review_dir / "candidates.csv", index=False)
    return table


def plot_real(table: pd.DataFrame) -> None:
    order = ["Affine only", "Density Flow baseline", "UOT conservative", "UOT primary", "Joint Flow"]
    data = table.set_index("method").loc[order]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].bar(range(len(data)), data["symmetric_median_nn_um"], color=["#999999", "#56B4E9", "#009E73", "#0072B2", "#D55E00"])
    axes[0].set_ylabel("Symmetric median NN (µm)")
    axes[1].bar(range(len(data)), 100 * data["within5_fraction"], color=["#999999", "#56B4E9", "#009E73", "#0072B2", "#D55E00"])
    axes[1].set_ylabel("Within 5 µm (%)")
    axes[2].bar(range(len(data)), data["displacement_p95_um"], color=["#999999", "#56B4E9", "#009E73", "#0072B2", "#D55E00"])
    axes[2].set_ylabel("Displacement p95 (µm)")
    labels = ["Affine", "Density", "UOT cons.", "UOT primary", "Joint"]
    for ax in axes:
        ax.set_xticks(range(len(data)), labels, rotation=28, ha="right")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Real dataset comparison (one unique dataset; NN is a proxy)")
    fig.tight_layout()
    fig.savefig(OUT / "real_dataset_comparison.png", dpi=180)
    plt.close(fig)

    pass_rates = data["common_audit_pass"].astype(float) * 100
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(data)), pass_rates, color=["#999999", "#56B4E9", "#009E73", "#0072B2", "#D55E00"])
    ax.set_xticks(range(len(data)), labels, rotation=25, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Common safety PASS rate (%)")
    ax.set_title("Real-data PASS rate (n=1; not a robustness estimate)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "pass_rate_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    save_json(OUT / "fixed_configurations.json", fixed_config_record())
    joint = load_workflow_c_result_artifact(JOINT_ARTIFACT.read_bytes())
    affine = np.asarray(joint.arrays["affine_he_points"], dtype=float)
    fixed_all = np.asarray(joint.arrays["fixed_geojson_points"], dtype=float)
    grid_x = np.asarray(joint.arrays["grid_x"], dtype=float)
    grid_y = np.asarray(joint.arrays["grid_y"], dtype=float)
    spacing = float(joint.field_spacing)
    metric_fixed = np.load(INPUT / "success_metric_fixed_points.npy", allow_pickle=False)
    metric_moving = np.load(INPUT / "success_metric_moving_points.npy", allow_pickle=False)
    tree = {tuple(np.asarray(point)): index for index, point in enumerate(affine)}
    metric_indices = np.array([tree[tuple(np.asarray(point))] for point in metric_moving], dtype=int)
    np.testing.assert_array_equal(affine[metric_indices], metric_moving)

    results = {
        "Affine only": affine_method(affine, grid_x, grid_y, metric_fixed, metric_indices, spacing),
        "Density Flow baseline": artifact_method("Density Flow baseline", DENSITY_ARTIFACT, metric_fixed, metric_indices),
        "Joint Flow": artifact_method("Joint Flow", JOINT_ARTIFACT, metric_fixed, metric_indices),
    }
    for method in ("UOT conservative", "UOT primary"):
        results[method] = run_uot(
            metric_fixed,
            affine,
            grid_x,
            grid_y,
            spacing,
            method,
            metric_fixed=metric_fixed,
            metric_moving_indices=metric_indices,
        )
        np.savez_compressed(
            OUT / ("real_" + method.lower().replace(" ", "_") + ".npz"),
            points=results[method]["points"],
            field_x=results[method]["field_x"],
            field_y=results[method]["field_y"],
            attempted_points=results[method]["attempted_points"],
            attempted_field_x=results[method]["attempted_field_x"],
            attempted_field_y=results[method]["attempted_field_y"],
        )

    order = ["Affine only", "Density Flow baseline", "UOT conservative", "UOT primary", "Joint Flow"]
    table = pd.DataFrame([metrics_row(results[name], metric_fixed, metric_indices, spacing) for name in order])
    table.to_csv(OUT / "real_dataset_metrics.csv", index=False)

    joint_row = table.set_index("method").loc["Joint Flow"]
    deltas = []
    for method in ("UOT conservative", "UOT primary"):
        row = table.set_index("method").loc[method]
        deltas.append({
            "dataset": DATASET,
            "comparison": f"{method} - Joint Flow",
            "symmetric_median_delta_um": row["symmetric_median_nn_um"] - joint_row["symmetric_median_nn_um"],
            "within5_delta_percentage_points": 100 * (row["within5_fraction"] - joint_row["within5_fraction"]),
            "within10_delta_percentage_points": 100 * (row["within10_fraction"] - joint_row["within10_fraction"]),
            "mutual_nn_delta_percentage_points": 100 * (row["mutual_nn_fraction"] - joint_row["mutual_nn_fraction"]),
            "displacement_p95_delta_um": row["displacement_p95_um"] - joint_row["displacement_p95_um"],
            "jacobian_p05_delta": row["jacobian_p05"] - joint_row["jacobian_p05"],
            "jacobian_min_delta": row["jacobian_min"] - joint_row["jacobian_min"],
            "uot_pass": bool(row["common_audit_pass"]),
            "joint_pass": bool(joint_row["common_audit_pass"]),
            "pass_comparison": f"{'PASS' if row['common_audit_pass'] else 'FAIL'} / {'PASS' if joint_row['common_audit_pass'] else 'FAIL'}",
        })
    pd.DataFrame(deltas).to_csv(OUT / "real_dataset_pairwise_deltas.csv", index=False)

    source_rows = {}
    for method, trial in (("UOT primary", "coarse_04"), ("UOT conservative", "coarse_09")):
        prior = json.loads((PRIOR_UOT / f"trials/uot/{trial}/result.json").read_text())
        source_rows[method] = prior
    reproduction = {}
    for method in ("UOT primary", "UOT conservative"):
        current = table.set_index("method").loc[method]
        prior = source_rows[method]
        comparisons = {
            "symmetric_median_nn_um": (current["symmetric_median_nn_um"], prior["audit"]["after"]["symmetric_median_distance"]),
            "displacement_p95_um": (current["displacement_p95_um"], prior["audit"]["displacement_p95_um"]),
            "jacobian_p05": (current["jacobian_p05"], prior["audit"]["jacobian_p05"]),
        }
        reproduction[method] = {
            name: {"current": float(values[0]), "source": float(values[1]), "delta": float(values[0] - values[1])}
            for name, values in comparisons.items()
        }
        reproduction[method]["exact_within_1e-10"] = all(abs(values[0] - values[1]) <= 1e-10 for values in comparisons.values())
    save_json(OUT / "real_reproduction_check.json", reproduction)

    inventory = pd.DataFrame([
        {
            "dataset": DATASET,
            "fixed_n_original": len(fixed_all),
            "moving_n_original": len(affine),
            "fixed_n_evaluation": len(metric_fixed),
            "moving_n_evaluation": len(metric_indices),
            "usable": True,
            "reason": "HE nuclei, fluorescence/GeoJSON nuclei, recoverable coordinates, saved affine, common Workflow C inputs",
            "unique_fixed_sha256": sha256(INPUT / "fixed_points.npy"),
            "unique_moving_sha256": sha256(INPUT / "moving_points.npy"),
        },
        {
            "dataset": "synthetic_smoke_20260904",
            "fixed_n_original": 80,
            "moving_n_original": 80,
            "fixed_n_evaluation": 65,
            "moving_n_evaluation": 66,
            "usable": False,
            "reason": "synthetic engineering smoke test, not an independent serial-section dataset",
            "unique_fixed_sha256": "f7bb63cb44892a341071d49939a920c66793ac74f8b9e231cfe514376992d5b6",
            "unique_moving_sha256": "2585b1f0a003069566b484acf11b73a822ab352aff1a78703f7a8020dfc39528",
        },
    ])
    inventory.to_csv(OUT / "dataset_inventory.csv", index=False)

    robustness = []
    for method, group in table.groupby("method", sort=False):
        values = group["symmetric_median_nn_um"]
        robustness.append({
            "method": method,
            "datasets": len(group),
            "pass_rate": float(group["common_audit_pass"].mean()),
            "failure_count": int((~group["common_audit_pass"].astype(bool)).sum()),
            "median_symmetric_nn_um": float(values.median()),
            "best_symmetric_nn_um": float(values.min()),
            "worst_symmetric_nn_um": float(values.max()),
            "variance_symmetric_nn_um2": float(values.var(ddof=0)),
            "iqr_symmetric_nn_um": float(values.quantile(0.75) - values.quantile(0.25)),
            "warning": "n=1; variance and IQR are mechanically zero and do not establish robustness",
        })
    pd.DataFrame(robustness).to_csv(OUT / "real_dataset_summary.csv", index=False)
    manual_candidates(results, fixed_all, affine)
    plot_real(table)
    save_json(OUT / "real_validation_status.json", {
        "state": "completed",
        "dataset": DATASET,
        "usable_real_dataset_count": 1,
        "input_hash_equivalence": "all saved real Workflow C runs/sweeps found on server share the same fixed/moving hashes",
        "workflow_integration_performed": False,
    })
    print(table[["method", "symmetric_median_nn_um", "common_audit_pass"]].to_string(index=False))


if __name__ == "__main__":
    main()
