"""Combine completed seed shards and create fixed synthetic summaries/figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from validation_lib import OUT, save_json


METHOD_ORDER = [
    "Affine only", "Density Flow baseline", "UOT conservative", "UOT primary", "Joint Flow"
]
COLORS = {
    "Affine only": "#999999",
    "Density Flow baseline": "#56B4E9",
    "UOT conservative": "#009E73",
    "UOT primary": "#0072B2",
    "Joint Flow": "#D55E00",
}


def summarize_group(scope: str, value: str, method: str, group: pd.DataFrame) -> dict:
    field = group["field_error_um_median"]
    tre = group["independent_landmark_tre_um_median"]
    unmatched = group["unmatched_movement_um_median"].dropna()
    return {
        "summary_scope": scope,
        "scope_value": value,
        "method": method,
        "cases": len(group),
        "seed_count": group["seed"].nunique(),
        "common_pass_rate": float(group["common_audit_pass"].mean()),
        "failure_count": int((~group["common_audit_pass"].astype(bool)).sum()),
        "exception_count": int(group["rejection_reason"].fillna("").str.startswith("exception:").sum()),
        "foldover_case_count": int((group["foldover_fraction"] > 0).sum()),
        "attempted_foldover_case_count": int((group["attempted_foldover_fraction"] > 0).sum()),
        "field_error_median_um": float(field.median()),
        "field_error_best_um": float(field.min()),
        "field_error_worst_um": float(field.max()),
        "field_error_variance_um2": float(field.var(ddof=0)),
        "field_error_iqr_um": float(field.quantile(0.75) - field.quantile(0.25)),
        "field_error_p95_median_um": float(group["field_error_um_p95"].median()),
        "endpoint_error_mean_median_um": float(group["endpoint_error_mean_um"].median()),
        "independent_tre_median_um": float(tre.median()),
        "independent_tre_best_um": float(tre.min()),
        "independent_tre_worst_um": float(tre.max()),
        "independent_tre_iqr_um": float(tre.quantile(0.75) - tre.quantile(0.25)),
        "known_matched_tre_median_um": float(group["known_matched_point_tre_um_median"].median()),
        "matched_displacement_error_median_um": float(group["matched_displacement_error_um_median"].median()),
        "unmatched_movement_median_um": float(unmatched.median()) if len(unmatched) else np.nan,
        "unmatched_movement_worst_um": float(unmatched.max()) if len(unmatched) else np.nan,
        "jacobian_abs_error_median": float(group["jacobian_abs_error_median"].median()),
        "displacement_p95_median_um": float(group["displacement_p95_um"].median()),
        "attempted_displacement_p95_worst_um": float(group["attempted_displacement_p95_um"].max()),
        "attempted_jacobian_min_worst": float(group["attempted_jacobian_min"].min()),
        "runtime_median_seconds_shared_not_benchmark": float(group["runtime_seconds"].median()),
    }


def make_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in table.groupby("method", sort=False):
        rows.append(summarize_group("overall", "all_predeclared_scenarios", method, group))
    difficulty = table[table["difficulty"].notna()]
    for (level, method), group in difficulty.groupby(["difficulty", "method"], sort=False):
        rows.append(summarize_group("difficulty", str(level), method, group))
    missing = table[table["is_missingness_sensitivity"].astype(bool)]
    for (fraction, method), group in missing.groupby(["missing_fraction", "method"], sort=True):
        rows.append(summarize_group("missingness", f"{float(fraction):.1f}", method, group))
    deformation = table[table["is_deformation_sensitivity"].astype(bool)]
    for (amplitude, method), group in deformation.groupby(["amplitude_um", "method"], sort=True):
        rows.append(summarize_group("deformation_amplitude_um", f"{float(amplitude):.1f}", method, group))
    result = pd.DataFrame(rows)
    result["method"] = pd.Categorical(result["method"], METHOD_ORDER, ordered=True)
    return result.sort_values(["summary_scope", "scope_value", "method"]).reset_index(drop=True)


def pairwise(table: pd.DataFrame) -> pd.DataFrame:
    index = ["scenario_id", "seed"]
    metrics = [
        "field_error_um_median", "field_error_um_p95", "independent_landmark_tre_um_median",
        "known_matched_point_tre_um_median", "unmatched_movement_um_median",
        "symmetric_median_nn_um", "within5_fraction", "within10_fraction", "mutual_nn_fraction",
        "displacement_p95_um", "jacobian_p05", "jacobian_min", "common_audit_pass",
    ]
    wide = table.pivot(index=index, columns="method", values=metrics)
    metadata = table.drop_duplicates(index).set_index(index)[[
        "difficulty", "is_missingness_sensitivity", "is_deformation_sensitivity",
        "amplitude_um", "missing_fraction", "unmatched_fraction", "overlap_fraction", "noise_um",
    ]]
    rows = []
    for key in wide.index:
        base = metadata.loc[key].to_dict()
        for method in ("UOT conservative", "UOT primary"):
            row = {"scenario_id": key[0], "seed": key[1], **base, "comparison": f"{method} - Joint Flow"}
            for metric in metrics[:-1]:
                row[f"{metric}_delta"] = float(wide.loc[key, (metric, method)] - wide.loc[key, (metric, "Joint Flow")])
            row["uot_pass"] = bool(wide.loc[key, ("common_audit_pass", method)])
            row["joint_pass"] = bool(wide.loc[key, ("common_audit_pass", "Joint Flow")])
            epe_delta = row["field_error_um_median_delta"]
            row["field_error_winner"] = method if epe_delta < -1e-12 else ("Joint Flow" if epe_delta > 1e-12 else "tie")
            row["nn_gt_discordant"] = bool(
                row["symmetric_median_nn_um_delta"] < 0.0
                and row["field_error_um_median_delta"] > 0.0
            )
            rows.append(row)
    return pd.DataFrame(rows)


def plot_ground_truth(table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    positions = np.arange(len(METHOD_ORDER))
    for ax, column, ylabel in (
        (axes[0], "field_error_um_median", "Median field EPE (µm)"),
        (axes[1], "independent_landmark_tre_um_median", "Independent landmark TRE (µm)"),
    ):
        values = [table.loc[table["method"] == method, column].to_numpy() for method in METHOD_ORDER]
        box = ax.boxplot(values, positions=positions, patch_artist=True, showfliers=False)
        for patch, method in zip(box["boxes"], METHOD_ORDER):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.75)
        ax.set_xticks(positions, ["Affine", "Density", "UOT cons.", "UOT primary", "Joint"], rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Synthetic known-ground-truth validation (50 scenarios)")
    fig.tight_layout()
    fig.savefig(OUT / "synthetic_ground_truth_error.png", dpi=180)
    plt.close(fig)


def plot_missingness(table: pd.DataFrame) -> None:
    data = table[table["is_missingness_sensitivity"].astype(bool)]
    methods = METHOD_ORDER[1:]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for method in methods:
        group = data[data["method"] == method].groupby("missing_fraction")
        x = np.array(sorted(group.groups))
        field = group["field_error_um_median"].median().reindex(x)
        unmatched = group["unmatched_movement_um_median"].median().reindex(x)
        axes[0].plot(100 * x, field, marker="o", label=method, color=COLORS[method])
        axes[1].plot(100 * x, unmatched, marker="o", label=method, color=COLORS[method])
    axes[0].set_ylabel("Median field EPE (µm)")
    axes[1].set_ylabel("Median movement of unmatched points (µm)")
    for ax in axes:
        ax.set_xlabel("Random missing fraction (%)")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Missingness sensitivity (5 seeds; other factors fixed)")
    fig.tight_layout()
    fig.savefig(OUT / "missingness_sensitivity.png", dpi=180)
    plt.close(fig)


def plot_pareto(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for method in METHOD_ORDER:
        group = table[table["method"] == method]
        ax.scatter(
            group["displacement_p95_um"], group["field_error_um_median"],
            s=20, alpha=0.22, color=COLORS[method],
        )
        x = group["displacement_p95_um"].median()
        y = group["field_error_um_median"].median()
        ax.scatter(x, y, s=120, marker="D", color=COLORS[method], edgecolor="black", linewidth=0.7, label=method)
    ax.set_xlabel("Applied field displacement p95 (µm)")
    ax.set_ylabel("Median ground-truth field EPE (µm; lower is better)")
    ax.set_title("Accuracy–deformation trade-off; diamonds are method medians")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_deformation_pareto.png", dpi=180)
    plt.close(fig)


def main() -> None:
    shard_paths = sorted((OUT / "synthetic_shards_v2").glob("synthetic_seed_*.csv"))
    if len(shard_paths) != 5:
        raise SystemExit(f"expected 5 completed seed shards, found {len(shard_paths)}")
    table = pd.concat([pd.read_csv(path) for path in shard_paths], ignore_index=True)
    expected_rows = 5 * 10 * 5
    if len(table) != expected_rows:
        raise SystemExit(f"expected {expected_rows} rows, found {len(table)}")
    if table[["seed", "scenario_id", "method"]].duplicated().any():
        raise SystemExit("duplicate synthetic result keys")
    table["difficulty"] = table["difficulty"].replace({np.nan: None})
    table.to_csv(OUT / "synthetic_metrics.csv", index=False)
    summary = make_summary(table)
    summary.to_csv(OUT / "synthetic_summary.csv", index=False)
    deltas = pairwise(table)
    deltas.to_csv(OUT / "synthetic_uot_joint_deltas.csv", index=False)

    independent = summary[
        summary["summary_scope"].isin(["overall", "difficulty", "missingness", "deformation_amplitude_um"])
    ][[
        "summary_scope", "scope_value", "method", "cases", "independent_tre_median_um",
        "independent_tre_worst_um", "known_matched_tre_median_um",
        "matched_displacement_error_median_um", "unmatched_movement_median_um",
        "field_error_median_um", "jacobian_abs_error_median", "common_pass_rate",
    ]]
    independent.to_csv(OUT / "independent_validation.csv", index=False)
    plot_ground_truth(table)
    plot_missingness(table)
    plot_pareto(table)

    overall = summary[summary["summary_scope"] == "overall"].set_index("method")
    comparisons = {}
    for method in ("UOT conservative", "UOT primary"):
        comparisons[method] = {
            "field_error_delta_vs_joint_um": float(overall.loc[method, "field_error_median_um"] - overall.loc["Joint Flow", "field_error_median_um"]),
            "independent_tre_delta_vs_joint_um": float(overall.loc[method, "independent_tre_median_um"] - overall.loc["Joint Flow", "independent_tre_median_um"]),
            "unmatched_movement_delta_vs_joint_um": float(overall.loc[method, "unmatched_movement_median_um"] - overall.loc["Joint Flow", "unmatched_movement_median_um"]),
            "pass_rate_delta_vs_joint": float(overall.loc[method, "common_pass_rate"] - overall.loc["Joint Flow", "common_pass_rate"]),
            "field_error_wins": int((deltas[deltas["comparison"] == f"{method} - Joint Flow"]["field_error_winner"] == method).sum()),
            "field_error_losses": int((deltas[deltas["comparison"] == f"{method} - Joint Flow"]["field_error_winner"] == "Joint Flow").sum()),
            "field_error_ties": int((deltas[deltas["comparison"] == f"{method} - Joint Flow"]["field_error_winner"] == "tie").sum()),
            "nn_better_but_field_error_worse_count": int(
                deltas[deltas["comparison"] == f"{method} - Joint Flow"]["nn_gt_discordant"].sum()
            ),
        }
    save_json(OUT / "synthetic_validation_status.json", {
        "state": "completed",
        "rows": len(table),
        "scenarios": int(table[["seed", "scenario_id"]].drop_duplicates().shape[0]),
        "seeds": sorted(map(int, table["seed"].unique())),
        "methods": METHOD_ORDER,
        "exception_count": int(table["rejection_reason"].fillna("").str.startswith("exception:").sum()),
        "affine_reconstruction_error_max_um": float(table["known_affine_reconstruction_error_max_um"].max()),
        "uot_vs_joint": comparisons,
        "runtime_is_formal_benchmark": False,
        "parameter_tuning_performed": False,
    })
    print(overall[["common_pass_rate", "field_error_median_um", "independent_tre_median_um", "unmatched_movement_median_um"]].to_string())


if __name__ == "__main__":
    main()
