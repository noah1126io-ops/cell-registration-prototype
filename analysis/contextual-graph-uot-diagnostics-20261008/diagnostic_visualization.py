"""Diagnostic-only figures for the frozen contextual graph-UOT audit."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"matched": "#168a5b", "ambiguous": "#d99100", "unmatched": "#b44343"}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def stage_attrition(stage: pd.DataFrame, trace: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = ["Affine", "Regional", "Fingerprint", "Graph", "UOT"]
    valid = [len(trace)] * len(labels)
    axes[0].bar(labels, valid, color="#3a78b4")
    axes[0].set_ylim(0, len(trace) * 1.13)
    axes[0].set_ylabel("Moving nuclei retained")
    axes[0].set_title("No moving-nucleus attrition before classification")
    for index, value in enumerate(valid):
        axes[0].text(index, value + 25, f"{value:,}", ha="center", fontsize=9)

    counts = trace["status"].value_counts()
    status_order = ["matched", "ambiguous", "unmatched"]
    values = [int(counts.get(status, 0)) for status in status_order]
    axes[1].bar(status_order, values, color=[COLORS[status] for status in status_order])
    axes[1].set_ylabel("Moving nuclei")
    axes[1].set_title("Final labels (coverage, not real accuracy)")
    for index, value in enumerate(values):
        axes[1].text(index, value + 20, f"{value:,}\n({value / len(trace):.1%})", ha="center", fontsize=9)
    fig.suptitle("Stage audit: candidates are retained; classification changes at UOT mass")
    fig.tight_layout()
    _save(fig, path)


def synthetic_vs_real_scores(points: pd.DataFrame, path: Path) -> None:
    variables = [
        ("affine_candidate_distance_um", "Top-1 distance (um)"),
        ("regional_ncc", "Regional NCC"),
        ("fingerprint_similarity", "Fingerprint similarity"),
        ("graph_support_score", "Graph support"),
        ("minimum_candidate_cost", "Minimum contextual cost"),
        ("row_mass_ratio", "UOT row mass ratio"),
        ("normalized_row_entropy", "Normalized row entropy"),
        ("confidence", "Composite confidence"),
    ]
    groups = ["Easy", "Medium", "Hard", "Real"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for axis, (variable, label) in zip(axes.flat, variables, strict=True):
        values = [
            points.loc[points["dataset_group"] == group, variable].dropna().to_numpy()
            for group in groups
        ]
        boxes = axis.boxplot(values, tick_labels=groups, showfliers=False, patch_artist=True)
        for patch, color in zip(boxes["boxes"], ["#76b7b2", "#59a14f", "#edc949", "#e15759"], strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
        axis.set_title(label)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Synthetic–real score distributions (real has no correspondence truth)", fontsize=14)
    fig.tight_layout()
    _save(fig, path)


def classification_reason_breakdown(reasons: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    panels = [
        ("exclusive_actual_unmatched_rule", "Actual exclusive unmatched rule"),
        ("overlapping_unmatched_diagnostic_flag", "Overlapping diagnostic flags"),
    ]
    english = {
        "Candidateなし": "No candidate",
        "Cost不適合": "Explicit cost rejection",
        "UOT mass不足": "Low UOT mass",
        "候補が曖昧": "Explicit ambiguity rejection",
        "Graph support不足": "Explicit graph rejection",
        "Reciprocal条件不成立": "Explicit reciprocity rejection",
        "Confidence threshold不達": "Explicit confidence rejection",
        "その他": "Other",
        "CostがSynthetic Hard p95超": "Cost above synthetic Hard p95",
        "行内候補が曖昧": "Diffuse within-row mass",
        "Graph supportがSynthetic Hard p05未満": "Graph below synthetic Hard p05",
        "Reciprocalでない": "Non-reciprocal",
    }
    for axis, (scope, title) in zip(axes, panels, strict=True):
        table = reasons.loc[reasons["scope"] == scope].copy()
        table = table.sort_values("nuclei")
        display = table["cause"].map(english).fillna(table["cause"])
        axis.barh(display, table["nuclei"], color="#4e79a7" if "exclusive" in scope else "#f28e2b")
        axis.set_title(title)
        axis.set_xlabel("Unmatched nuclei")
        axis.grid(axis="x", alpha=0.2)
        for index, value in enumerate(table["nuclei"]):
            axis.text(value + 8, index, f"{int(value):,}", va="center", fontsize=8)
    fig.suptitle("Formal classifier cause is separate from correlated upstream diagnostics")
    fig.tight_layout()
    _save(fig, path)


def uot_mass_vs_confidence(trace: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for status in ("unmatched", "ambiguous", "matched"):
        table = trace.loc[trace["status"] == status]
        axes[0].scatter(
            table["row_mass_ratio"],
            table["confidence_saved_algorithm"],
            s=13,
            alpha=0.45,
            color=COLORS[status],
            label=f"{status} ({len(table)})",
        )
        axes[1].scatter(
            table["row_mass_ratio"],
            table["normalized_top1_mass"],
            s=13,
            alpha=0.45,
            color=COLORS[status],
        )
    for axis in axes:
        axis.axvline(0.12, color="black", linestyle="--", linewidth=1, label="unmatched floor 0.12")
        axis.axvline(0.25, color="black", linestyle=":", linewidth=1.2, label="matched mass gate 0.25")
        axis.set_xlabel("UOT row mass ratio (absolute mass/source mass)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Composite confidence (not status threshold)")
    axes[0].set_title("Absolute mass largely drives status and confidence")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0.16, color="#555555", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Row-normalized top-1 mass")
    axes[1].set_title("Many low-mass rows are internally concentrated")
    fig.tight_layout()
    _save(fig, path)


def confidence_coverage(confidence: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    synthetic = confidence.loc[
        (confidence["record_type"] == "synthetic_threshold_curve")
        & (confidence["dataset_group"] == "Synthetic all")
    ]
    axes[0].plot(synthetic["coverage_all"], synthetic["precision_selected"], marker="o", ms=3)
    axes[0].set_xlabel("Coverage")
    axes[0].set_ylabel("Empirical top-1 precision")
    axes[0].set_title("Synthetic precision–coverage")
    axes[1].plot(synthetic["threshold"], synthetic["matched_recall"], label="Matched recall")
    axes[1].plot(synthetic["threshold"], synthetic["unmatched_f1"], label="Unmatched F1")
    axes[1].set_xlabel("Diagnostic confidence threshold")
    axes[1].set_title("Synthetic threshold sensitivity")
    axes[1].legend()
    real = confidence.loc[confidence["record_type"] == "real_threshold_coverage_only"]
    axes[2].plot(real["threshold"], real["coverage_all"], color="#e15759")
    axes[2].axvline(0.75, color="black", linestyle="--", linewidth=1)
    axes[2].set_xlabel("Hypothetical confidence threshold")
    axes[2].set_ylabel("Coverage only")
    axes[2].set_title("Real: no accuracy interpretation")
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.03)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    _save(fig, path)


def individual_match_examples(
    fixed: np.ndarray,
    moving: np.ndarray,
    examples: pd.DataFrame,
    path: Path,
) -> None:
    groups = ["Matched", "Ambiguous", "Unmatched high top-1 score", "Unmatched small mass"]
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    rank_colors = ["#d62728", "#ff7f0e", "#bcbd22"]
    for row, group in enumerate(groups):
        table = examples.loc[examples["example_group"] == group].head(4)
        for column, axis in enumerate(axes[row]):
            if column >= len(table):
                axis.axis("off")
                continue
            item = table.iloc[column]
            center = np.asarray([item["moving_x_um"], item["moving_y_um"]])
            fixed_distance = np.linalg.norm(fixed - center, axis=1)
            moving_distance = np.linalg.norm(moving - center, axis=1)
            fixed_local = fixed[fixed_distance <= 35]
            moving_local = moving[moving_distance <= 35]
            axis.scatter(fixed_local[:, 0], fixed_local[:, 1], s=10, c="#888888", alpha=0.48, label="fixed")
            axis.scatter(moving_local[:, 0], moving_local[:, 1], s=9, c="#4e79a7", alpha=0.35, label="moving")
            axis.scatter(center[0], center[1], marker="*", s=90, c="#111111", zorder=5)
            for rank, color in zip(range(1, 4), rank_colors, strict=True):
                candidate = np.asarray([item[f"rank{rank}_fixed_x_um"], item[f"rank{rank}_fixed_y_um"]])
                axis.plot([center[0], candidate[0]], [center[1], candidate[1]], color=color, linewidth=1.6, alpha=0.9)
                axis.scatter(candidate[0], candidate[1], s=30, c=color, edgecolor="white", linewidth=0.4, zorder=4)
            axis.set_xlim(center[0] - 38, center[0] + 38)
            axis.set_ylim(center[1] - 38, center[1] + 38)
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(
                f"{item['moving_nucleus_id']}\nmass={item['row_mass_ratio']:.3f}, conf={item['confidence_saved_algorithm']:.3f}",
                fontsize=8,
            )
            if column == 0:
                axis.set_ylabel(group, fontsize=9)
    fig.suptitle("Real candidate proposals — colored lines are top-3 candidates, not ground truth", fontsize=14)
    fig.tight_layout()
    _save(fig, path)
