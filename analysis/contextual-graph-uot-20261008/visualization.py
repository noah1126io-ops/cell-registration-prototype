"""Figures required by the contextual correspondence study."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = ["#777777", "#CC79A7", "#E69F00", "#56B4E9", "#009E73", "#0072B2"]


def method_comparison(table: pd.DataFrame, output: Path) -> None:
    metrics = ["top1_accuracy", "matched_f1", "unmatched_f1"]
    labels = ["Top-1 accuracy", "Matched F1", "Unmatched F1"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    x = np.arange(len(table))
    for ax, metric, label in zip(axes, metrics, labels, strict=True):
        ax.bar(x, table[metric], color=COLORS[: len(table)])
        ax.set_title(label)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.2)
        ax.set_xticks(x, table["short_method"], rotation=35, ha="right")
    axes[0].set_ylabel("Fraction")
    fig.suptitle("Cell-correspondence outcomes (primary synthetic cases)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def candidate_recall_plot(table: pd.DataFrame, output: Path) -> None:
    grouped = table.groupby(["candidate_strategy", "k"], as_index=False)["candidate_recall"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy, group in grouped.groupby("candidate_strategy", sort=False):
        ax.plot(group["k"], group["candidate_recall"], marker="o", lw=2, label=strategy)
    ax.set_xlabel("Candidate set size K")
    ax.set_ylabel("True-match candidate recall")
    ax.set_ylim(0, 1.02)
    ax.set_xticks((10, 20, 30))
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title("Candidate discovery before UOT")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def calibration_plot(table: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for method, group in table.groupby("method", sort=False):
        ax.plot(group["mean_confidence"], group["empirical_accuracy"], marker="o", label=method.split()[0])
    ax.plot([0, 1], [0, 1], "--", color="black", lw=1, label="Ideal")
    ax.set_xlabel("Mean reported confidence")
    ax.set_ylabel("Empirical top-1 accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title("Confidence calibration")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def missingness_plot(table: pd.DataFrame, output: Path) -> None:
    grouped = table.groupby(["method", "missing_fraction"], as_index=False)["top1_accuracy"].mean()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method, group in grouped.groupby("method", sort=False):
        ax.plot(100 * group["missing_fraction"], group["top1_accuracy"], marker="o", lw=1.8, label=method.split()[0])
    ax.set_xlabel("Moving-slice missingness (%)")
    ax.set_ylabel("Top-1 correspondence accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    ax.set_title("Missingness sensitivity (fixed settings)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def graph_support_plot(table: pd.DataFrame, output: Path, auc: float) -> None:
    correct = table.loc[table["is_true_candidate"], "graph_support"].to_numpy()
    wrong = table.loc[~table["is_true_candidate"], "graph_support"].to_numpy()
    bins = np.linspace(0, 1, 25)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(wrong, bins=bins, density=True, alpha=0.55, color="#D55E00", label=f"Incorrect (n={len(wrong)})")
    ax.hist(correct, bins=bins, density=True, alpha=0.65, color="#009E73", label=f"Correct (n={len(correct)})")
    ax.set_xlabel("Local graph support score")
    ax.set_ylabel("Density")
    ax.set_title(f"Graph support separation; rank AUC={auc:.3f}")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def synthetic_example(sample, result, output: Path) -> None:
    common = sample.true_fixed_index >= 0
    correct = common & (result.top1_index == sample.true_fixed_index)
    wrong = common & ~correct
    unmatched = ~common
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(sample.fixed_points[:, 0], sample.fixed_points[:, 1], s=10, c="black", alpha=0.35, label="Fixed nuclei")
    ax.scatter(sample.moving_points[unmatched, 0], sample.moving_points[unmatched, 1], s=24, marker="x", c="#999999", label="True unmatched")
    for indices, color, label in ((np.flatnonzero(correct), "#009E73", "Correct"), (np.flatnonzero(wrong), "#D55E00", "Wrong")):
        for position, moving_index in enumerate(indices[:90]):
            fixed_index = result.top1_index[moving_index]
            ax.plot(
                [sample.moving_points[moving_index, 0], sample.fixed_points[fixed_index, 0]],
                [sample.moving_points[moving_index, 1], sample.fixed_points[fixed_index, 1]],
                color=color,
                alpha=0.45,
                lw=0.8,
                label=label if position == 0 else None,
            )
    ax.scatter(sample.moving_points[:, 0], sample.moving_points[:, 1], s=9, c="#0072B2", alpha=0.45, label="Moving nuclei")
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title("M5 synthetic correspondence example")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def real_candidate_map(fixed: np.ndarray, moving: np.ndarray, table: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(fixed[:, 0], fixed[:, 1], s=5, c="black", alpha=0.20, label="Fixed")
    palette = {"matched": "#009E73", "ambiguous": "#E69F00", "unmatched": "#999999"}
    for status in ("matched", "ambiguous", "unmatched"):
        subset = table.loc[table["status"] == status]
        if subset.empty:
            continue
        ids = subset["moving_index"].to_numpy(dtype=int)
        ax.scatter(moving[ids, 0], moving[ids, 1], s=7, alpha=0.55, c=palette[status], label=status)
    high = table.sort_values("confidence", ascending=False).head(80)
    for row in high.itertuples():
        source = moving[int(row.moving_index)]
        target = fixed[int(row.top1_fixed_index)]
        ax.plot([source[0], target[0]], [source[1], target[1]], color="#0072B2", lw=0.45, alpha=0.4)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title("Real serial-section exploratory M5 candidates (no correspondence truth)")
    ax.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)

