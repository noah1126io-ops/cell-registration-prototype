"""Deterministic figures for the graph matching/adoption separation study."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "Position + Graph": "#4477AA",
    "Relative-cost Graph-UOT": "#66CCEE",
    "Separated Matching / Adoption": "#228833",
    "Full contextual UOT": "#CC6677",
    "Graph ranking only + separated adoption": "#AA3377",
    "Position only": "#BBBBBB",
}


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def precision_coverage(curves: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for method, table in curves.groupby("method", sort=False):
        ax.plot(
            table["coverage"],
            table["precision"],
            label=method,
            color=COLORS.get(method),
            linewidth=2,
        )
    for value in (0.90, 0.95, 0.98):
        ax.axhline(value, color="#999999", linestyle="--", linewidth=0.7)
    ax.set(xlabel="Coverage (accepted / all moving nuclei)", ylabel="Matched precision", ylim=(0, 1.02))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    _save(fig, path)


def matched_f1(table: pd.DataFrame, path: Path) -> None:
    data = table.set_index("method")
    methods = list(data.index)
    fig, ax = plt.subplots(figsize=(8.0, 4.7))
    ax.bar(
        np.arange(len(methods)),
        data["matched_f1"],
        color=[COLORS.get(method, "#999999") for method in methods],
    )
    ax.set_xticks(np.arange(len(methods)), [m.replace(" ", "\n", 1) for m in methods], fontsize=8)
    ax.set(ylabel="Held-out matched F1", ylim=(0, 1))
    ax.grid(axis="y", alpha=0.2)
    _save(fig, path)


def graph_discrimination(pairs: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0), sharey=True)
    for ax, (strength, table) in zip(axes, pairs.groupby("strength", sort=False)):
        true = table.loc[table["is_true_candidate"], "graph_support"]
        wrong = table.loc[~table["is_true_candidate"], "graph_support"]
        ax.boxplot([true, wrong], tick_labels=["True", "Other"], showfliers=False)
        ax.set_title(strength)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Graph support")
    _save(fig, path)


def unmatched_detection(table: pd.DataFrame, path: Path) -> None:
    methods = table["method"].tolist()
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(8.0, 4.7))
    ax.bar(x - 0.18, table["unmatched_precision"], width=0.36, label="Precision", color="#4477AA")
    ax.bar(x + 0.18, table["unmatched_recall"], width=0.36, label="Recall", color="#EE6677")
    ax.set_xticks(x, [m.replace(" ", "\n", 1) for m in methods], fontsize=8)
    ax.set(ylim=(0, 1), ylabel="Held-out unmatched metric")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    _save(fig, path)


def mass_vs_accuracy(rows: pd.DataFrame, path: Path) -> None:
    table = rows.copy()
    table["mass_bin"] = pd.qcut(table["row_mass_ratio"], 10, duplicates="drop")
    grouped = table.groupby("mass_bin", observed=True).agg(
        mass=("row_mass_ratio", "median"), accuracy=("top1_correct", "mean"), count=("top1_correct", "size")
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.plot(grouped["mass"], grouped["accuracy"], marker="o", color="#4477AA")
    ax.set(xlabel="Row mass ratio (bin median)", ylabel="Top-1 accuracy", ylim=(0, 1))
    ax.grid(alpha=0.2)
    _save(fig, path)


def real_like_domain_gap(summary: pd.DataFrame, path: Path) -> None:
    variables = [
        "nuclear_density_per_um2",
        "knn_distance_um",
        "candidate_distance_um",
        "graph_support",
        "minimum_contextual_cost",
        "row_mass_ratio",
        "candidate_margin",
        "confidence",
    ]
    pivot = summary.loc[summary["variable"].isin(variables)].pivot(
        index="variable", columns="dataset_group", values="median"
    )
    normalized = pivot.div(pivot.max(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x = np.arange(len(normalized))
    width = 0.36
    for index, column in enumerate(normalized.columns):
        ax.bar(x + (index - 0.5) * width, normalized[column], width, label=column)
    ax.set_xticks(x, [value.replace("_", "\n") for value in normalized.index], fontsize=8)
    ax.set(ylabel="Median / row maximum", ylim=(0, 1.08))
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    _save(fig, path)


def real_candidates(
    fixed: np.ndarray,
    moving: np.ndarray,
    rows: pd.DataFrame,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.scatter(fixed[:, 0], fixed[:, 1], s=4, color="#BBBBBB", label="Fixed", alpha=0.65)
    ax.scatter(moving[:, 0], moving[:, 1], s=4, color="#4477AA", label="Moving", alpha=0.55)
    chosen = rows.loc[rows["status"] == "matched"].nlargest(80, "calibrated_confidence")
    for row in chosen.itertuples():
        destination = fixed[int(row.top1_fixed_index)]
        source = moving[int(row.moving_index)]
        ax.plot([source[0], destination[0]], [source[1], destination[1]], color="#228833", alpha=0.25, linewidth=0.6)
    ax.set_aspect("equal")
    ax.set(title="Real exploratory candidates (synthetic-derived adoption)", xlabel="x (um)", ylabel="y (um)")
    ax.legend(markerscale=3)
    _save(fig, path)

