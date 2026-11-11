"""Figures for the candidate-fixed UOT cost ablation."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORDER = [
    "A0 Position",
    "A1 Position+FP",
    "A2 Position+Region",
    "A3 Position+Graph",
    "A4 Position+FP+Graph",
    "A5 Position+FP+Region",
    "A6 Full",
]
SHORT = {name: name.split()[0] for name in ORDER}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def cost_component_distributions(table: pd.DataFrame, path: Path) -> None:
    columns = [
        ("position_contribution", "Position"),
        ("fingerprint_contribution", "Fingerprint"),
        ("region_contribution", "Region"),
        ("graph_contribution", "Graph"),
    ]
    values = [table[column].to_numpy() for column, _ in columns]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    boxes = axes[0].boxplot(values, tick_labels=[label for _, label in columns], showfliers=False, patch_artist=True)
    for box, color in zip(boxes["boxes"], ["#4e79a7", "#59a14f", "#f28e2b", "#e15759"], strict=True):
        box.set_facecolor(color)
        box.set_alpha(0.72)
    axes[0].set_ylabel("Weighted candidate-cost contribution")
    axes[0].set_title("Absolute contribution distribution")
    axes[0].grid(axis="y", alpha=0.2)
    row_stats = []
    for column, label in columns:
        matrix = table.pivot(index="moving_index", columns="candidate_slot", values=column).to_numpy()
        row_stats.append((label, np.mean(matrix, axis=1), np.std(matrix, axis=1)))
    for label, mean, std in row_stats:
        axes[1].scatter(mean, std, s=7, alpha=0.25, label=label)
    axes[1].set_xlabel("Per-moving candidate mean (absolute uplift)")
    axes[1].set_ylabel("Per-moving candidate SD (relative discrimination)")
    axes[1].set_title("Absolute level vs within-row spread")
    axes[1].legend(markerscale=2)
    axes[1].grid(alpha=0.2)
    fig.suptitle("Real candidate cost components")
    fig.tight_layout()
    _save(fig, path)


def transported_mass_by_cost(real: pd.DataFrame, scale: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    ordered = real.set_index("condition").loc[ORDER]
    axes[0].bar([SHORT[name] for name in ORDER], ordered["total_transported_mass"], color="#4e79a7")
    axes[0].set_ylabel("Total transported mass")
    axes[0].set_title("Candidate-fixed cost ablation")
    axes[0].grid(axis="y", alpha=0.2)
    scale_names = scale["condition"].tolist()
    axes[1].bar(range(len(scale)), scale["total_transported_mass"], color="#f28e2b")
    axes[1].set_xticks(range(len(scale)), [name.split()[0] for name in scale_names])
    axes[1].set_ylabel("Total transported mass")
    axes[1].set_title("Absolute-cost sensitivity; raw ranks preserved")
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, path)


def row_mass_distribution(table: pd.DataFrame, path: Path) -> None:
    values = [table.loc[table["condition"] == condition, "row_mass_ratio"].to_numpy() for condition in ORDER]
    fig, axis = plt.subplots(figsize=(11, 5.5))
    boxes = axis.boxplot(values, tick_labels=[SHORT[name] for name in ORDER], showfliers=False, patch_artist=True)
    for box in boxes["boxes"]:
        box.set_facecolor("#76b7b2")
        box.set_alpha(0.75)
    axis.axhline(0.12, color="#b44343", linestyle="--", label="unmatched floor 0.12")
    axis.axhline(0.25, color="#168a5b", linestyle=":", label="matched mass gate 0.25")
    axis.set_ylabel("Row mass ratio")
    axis.set_title("Real row-mass distribution under fixed candidates")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, path)


def cost_vs_mass(table: pd.DataFrame, path: Path) -> None:
    selected = ["A0 Position", "A1 Position+FP", "A4 Position+FP+Graph", "A6 Full"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for axis, condition in zip(axes.flat, selected, strict=True):
        subset = table.loc[table["condition"] == condition]
        axis.scatter(subset["minimum_candidate_cost"], subset["row_mass_ratio"], s=8, alpha=0.3)
        correlation = subset[["minimum_candidate_cost", "row_mass_ratio"]].corr(method="spearman").iloc[0, 1]
        axis.set_title(f"{condition}  Spearman={correlation:.3f}")
        axis.set_xlabel("Minimum candidate cost")
        axis.set_ylabel("Row mass ratio")
        axis.grid(alpha=0.2)
    fig.suptitle("Absolute cost and UOT row mass")
    fig.tight_layout()
    _save(fig, path)


def graph_effect_paired(table: pd.DataFrame, path: Path) -> None:
    real = table.loc[table["dataset"] == "Real"]
    synthetic = table.loc[(table["dataset"] == "Synthetic") & table["is_common_cell"].eq(True)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    colors = np.where(real["uot_top1_changed"], "#e15759", "#4e79a7")
    axes[0].scatter(
        real["minimum_cost_delta_a4_minus_a1"],
        real["row_mass_delta_a4_minus_a1"],
        c=colors,
        s=9,
        alpha=0.35,
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Minimum cost delta (A4-A1)")
    axes[0].set_ylabel("Row mass delta (A4-A1)")
    axes[0].set_title("Real paired effect\nred: UOT top-1 changed")
    pattern_order = [
        "rank_changed_mass_increased",
        "rank_changed_mass_decreased",
        "rank_unchanged_mass_increased",
        "rank_unchanged_mass_decreased",
    ]
    counts = real["neutral_real_pattern"].value_counts()
    axes[1].bar(range(4), [counts.get(name, 0) for name in pattern_order], color="#f28e2b")
    axes[1].set_xticks(range(4), ["changed\nmass+", "changed\nmass-", "same\nmass+", "same\nmass-"])
    axes[1].set_ylabel("Real nuclei")
    axes[1].set_title("Neutral real categories")
    effects = synthetic["uot_identity_effect"].value_counts()
    effect_order = ["improved", "worsened", "remained_correct", "remained_incorrect"]
    axes[2].bar(range(4), [effects.get(name, 0) for name in effect_order], color="#59a14f")
    axes[2].set_xticks(range(4), ["improved", "worsened", "still\ncorrect", "still\nwrong"])
    axes[2].set_ylabel("Common synthetic cells")
    axes[2].set_title("Identity effect using truth")
    for axis in axes[1:]:
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Adding Graph to Position+Fingerprint")
    fig.tight_layout()
    _save(fig, path)


def synthetic_accuracy_vs_mass(metrics: pd.DataFrame, path: Path) -> None:
    aggregate = metrics.loc[metrics["record_type"] == "aggregate"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {"Easy": "#59a14f", "Medium": "#f28e2b", "Hard": "#e15759"}
    for difficulty in ("Easy", "Medium", "Hard"):
        table = aggregate.loc[aggregate["difficulty"] == difficulty].set_index("condition").loc[ORDER]
        axes[0].plot(table["row_mass_ratio_median"], table["uot_top1_accuracy"], marker="o", label=difficulty, color=colors[difficulty])
        axes[1].plot(table["row_mass_ratio_median"], table["matched_recall"], marker="o", label=difficulty, color=colors[difficulty])
        axes[2].plot(table["uot_top1_accuracy"], table["matched_f1"], marker="o", label=difficulty, color=colors[difficulty])
    axes[0].set_xlabel("Median row mass ratio")
    axes[0].set_ylabel("UOT Top-1 accuracy")
    axes[0].set_title("Discrimination vs mass")
    axes[1].set_xlabel("Median row mass ratio")
    axes[1].set_ylabel("Matched recall")
    axes[1].set_title("Adoption recall vs mass")
    axes[2].set_xlabel("UOT Top-1 accuracy")
    axes[2].set_ylabel("Matched F1")
    axes[2].set_title("Accuracy vs adopted-match quality")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    fig.tight_layout()
    _save(fig, path)


def matched_coverage_by_method(real: pd.DataFrame, synthetic: pd.DataFrame, path: Path) -> None:
    synthetic_all = synthetic.loc[
        (synthetic["record_type"] == "aggregate") & (synthetic["difficulty"] == "All")
    ].set_index("condition").loc[ORDER]
    real_ordered = real.set_index("condition").loc[ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar([SHORT[name] for name in ORDER], real_ordered["matched_n"] / 1754, color="#4e79a7")
    axes[0].set_ylabel("Matched coverage")
    axes[0].set_title("Real coverage only (no truth)")
    x = np.arange(len(ORDER))
    width = 0.38
    axes[1].bar(x - width / 2, synthetic_all["matched_precision"], width, label="precision", color="#59a14f")
    axes[1].bar(x + width / 2, synthetic_all["matched_recall"], width, label="recall", color="#f28e2b")
    axes[1].set_xticks(x, [SHORT[name] for name in ORDER])
    axes[1].set_title("Synthetic adopted-match precision/recall")
    axes[1].legend()
    for axis in axes:
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, path)
