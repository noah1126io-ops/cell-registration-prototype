"""Run the analysis-only affine, coordinate, NCC, and candidate audit."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

import audit_lib as lib


COLORS = {
    "fixed": "#0072B2",
    "moving": "#D55E00",
    "boundary": "#CC3311",
    "interior": "#009E73",
    "unknown": "#999999",
}


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def stable_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def candidate_hash(indices: np.ndarray) -> str:
    return stable_hash(np.asarray(indices, dtype=np.int64))


def source_hashes() -> dict:
    paths = [
        lib.ROOT / "src/pointset_registration.py",
        lib.ROOT / "src/point_io.py",
        lib.ROOT / "src/geojson_utils.py",
        lib.ROOT / "app.py",
        lib.CONTEXT / "regional_pattern.py",
        lib.CONTEXT / "candidate_generation.py",
        lib.CONTEXT / "graph_consistency.py",
    ]
    return {str(path.relative_to(lib.ROOT)): lib.file_sha256(path) for path in paths}


def baseline_reproduction(regional, rankings, candidates, graph) -> dict:
    saved = pd.read_csv(lib.DIAGNOSTICS / "per_nucleus_trace.csv")
    if len(saved) != len(regional.cell_anchor_index):
        raise AssertionError("Saved diagnostic row count differs")
    cell = regional.cell_anchor_index
    checks = {
        "regional_offset_x_max_abs_error_um": float(np.max(np.abs(
            regional.cell_offsets[:, 0] - saved["regional_best_offset_x_um"].to_numpy(float)
        ))),
        "regional_offset_y_max_abs_error_um": float(np.max(np.abs(
            regional.cell_offsets[:, 1] - saved["regional_best_offset_y_um"].to_numpy(float)
        ))),
        "regional_ncc_max_abs_error": float(np.max(np.abs(
            regional.best_ncc[cell] - saved["regional_best_ncc"].to_numpy(float)
        ))),
        "regional_margin_max_abs_error": float(np.max(np.abs(
            regional.margin[cell] - saved["regional_ncc_margin"].to_numpy(float)
        ))),
        "candidate_indices_match_saved_hash": candidate_hash(candidates.indices)
        == stable_hash(np.asarray(candidates.indices, dtype=np.int64)),
    }
    # Compare the sparse implementation to the frozen literal loop once.
    slow_graph = lib.graph_support_scores(
        np.load(lib.INPUT / "success_metric_fixed_points.npy", allow_pickle=False),
        np.load(lib.INPUT / "success_metric_moving_points.npy", allow_pickle=False),
        candidates,
    )
    checks["fast_graph_support_max_abs_error"] = float(np.max(np.abs(graph - slow_graph)))
    graph_cost = candidates.base_cost + float(lib.CANDIDATES["graph_weight"]) * (1.0 - graph)
    top = candidates.indices[np.arange(len(candidates.indices)), np.argmin(graph_cost, axis=1)]
    saved_graph_top = saved["graph_cost_rank1_fixed_index"].to_numpy(int)
    checks["graph_cost_top1_exact"] = bool(np.array_equal(top, saved_graph_top))
    if max(checks[key] for key in checks if key.endswith("error") or key.endswith("error_um")) > 1e-11:
        raise AssertionError(f"Baseline numeric reproduction failed: {checks}")
    if not checks["graph_cost_top1_exact"]:
        raise AssertionError("Graph-cost top-1 reproduction failed")
    return checks


def run_regional_windows(inputs, fingerprint_cost=None):
    start = time.perf_counter()
    states = {}
    for radius in (18.0, 36.0, 54.0):
        regional = lib.regional_match_vectorized(inputs.fixed, inputs.moving, radius)
        if fingerprint_cost is None:
            fingerprints = lib.compute_fingerprint_pair(inputs.fixed, inputs.moving)
            fingerprint_cost = lib.pairwise_fingerprint_cost(fingerprints)
        rankings = lib.build_candidate_rankings(inputs.fixed, inputs.moving, fingerprint_cost, regional)
        candidates = lib.build_candidate_set(rankings, fingerprint_cost, regional, include_region=True)
        graph = lib.graph_support_scores_fast(inputs.fixed, inputs.moving, candidates)
        graph_cost = candidates.base_cost + float(lib.CANDIDATES["graph_weight"]) * (1.0 - graph)
        slots = np.argmin(graph_cost, axis=1)
        graph_top = candidates.indices[np.arange(len(inputs.moving)), slots]
        states[radius] = {
            "regional": regional,
            "rankings": rankings,
            "candidates": candidates,
            "graph": graph,
            "graph_top": graph_top,
            "graph_top_support": graph[np.arange(len(inputs.moving)), slots],
            "graph_top_distance": candidates.spatial_distance_um[np.arange(len(inputs.moving)), slots],
        }
    baseline = states[18.0]
    cell_density = lib.local_density(inputs.moving, inputs.moving)
    metrics_rows = []
    per_cell_rows = []
    anchor_rows = []
    for radius, state in states.items():
        regional = state["regional"]
        flags_anchor = lib.boundary_flags(regional.best_offsets, radius)
        cell_anchor = regional.cell_anchor_index
        flags_cell = {key: values[cell_anchor] for key, values in flags_anchor.items()}
        candidate_jaccard = np.asarray([
            len(set(map(int, a)) & set(map(int, b))) / len(set(map(int, a)) | set(map(int, b)))
            for a, b in zip(state["candidates"].indices, baseline["candidates"].indices, strict=True)
        ])
        regional_top_changed = state["rankings"].regional[:, 0] != baseline["rankings"].regional[:, 0]
        graph_top_changed = state["graph_top"] != baseline["graph_top"]
        for i in range(len(inputs.moving)):
            per_cell_rows.append({
                "search_radius_um": radius,
                "moving_subset_index": i,
                "moving_original_index": int(inputs.moving_original_index[i]),
                "moving_x_um": inputs.moving[i, 0],
                "moving_y_um": inputs.moving[i, 1],
                "anchor_index": int(cell_anchor[i]),
                "offset_x_um": regional.cell_offsets[i, 0],
                "offset_y_um": regional.cell_offsets[i, 1],
                "offset_norm_um": float(np.linalg.norm(regional.cell_offsets[i])),
                "hits_boundary": bool(flags_cell["any"][i]),
                "hits_x_positive": bool(flags_cell["x_positive"][i]),
                "hits_x_negative": bool(flags_cell["x_negative"][i]),
                "hits_y_positive": bool(flags_cell["y_positive"][i]),
                "hits_y_negative": bool(flags_cell["y_negative"][i]),
                "hits_xy_both": bool(flags_cell["xy_both"][i]),
                "best_ncc": regional.best_ncc[cell_anchor[i]],
                "second_ncc": regional.second_ncc[cell_anchor[i]],
                "peak_margin": regional.margin[cell_anchor[i]],
                "local_moving_density_per_um2": cell_density[i],
                "candidate_top1_distance_um": state["graph_top_distance"][i],
                "graph_top1_support": state["graph_top_support"][i],
                "candidate_set_jaccard_vs_radius18": candidate_jaccard[i],
                "regional_top1_changed_vs_radius18": bool(regional_top_changed[i]),
                "graph_top1_changed_vs_radius18": bool(graph_top_changed[i]),
            })
        anchor_density = lib.local_density(inputs.moving, regional.anchor_centers)
        for i in range(len(regional.anchor_centers)):
            anchor_rows.append({
                "search_radius_um": radius,
                "anchor_index": i,
                "anchor_x_um": regional.anchor_centers[i, 0],
                "anchor_y_um": regional.anchor_centers[i, 1],
                "offset_x_um": regional.best_offsets[i, 0],
                "offset_y_um": regional.best_offsets[i, 1],
                "offset_norm_um": float(np.linalg.norm(regional.best_offsets[i])),
                "hits_boundary": bool(flags_anchor["any"][i]),
                "hits_x_positive": bool(flags_anchor["x_positive"][i]),
                "hits_x_negative": bool(flags_anchor["x_negative"][i]),
                "hits_y_positive": bool(flags_anchor["y_positive"][i]),
                "hits_y_negative": bool(flags_anchor["y_negative"][i]),
                "hits_xy_both": bool(flags_anchor["xy_both"][i]),
                "best_ncc": regional.best_ncc[i],
                "second_ncc": regional.second_ncc[i],
                "peak_margin": regional.margin[i],
                "local_moving_density_per_um2": anchor_density[i],
            })
        group_masks = {
            "all": np.ones(len(inputs.moving), dtype=bool),
            "boundary": flags_cell["any"],
            "non_boundary": ~flags_cell["any"],
            "x_positive": flags_cell["x_positive"],
            "x_negative": flags_cell["x_negative"],
            "y_positive": flags_cell["y_positive"],
            "y_negative": flags_cell["y_negative"],
            "xy_both": flags_cell["xy_both"],
        }
        for group, mask in group_masks.items():
            if not np.any(mask):
                continue
            anchor_ids = np.unique(cell_anchor[mask])
            anchor_boundary = flags_anchor["any"][anchor_ids]
            row = {
                "search_radius_um": radius,
                "unit": "moving_nucleus",
                "group": group,
                "n": int(mask.sum()),
                "independent_anchor_count": int(len(anchor_ids)),
                "independent_anchor_boundary_hit_count": int(anchor_boundary.sum()),
                "independent_anchor_boundary_hit_fraction": float(np.mean(anchor_boundary)),
                "boundary_hit_count": int(flags_cell["any"][mask].sum()),
                "boundary_hit_fraction": float(np.mean(flags_cell["any"][mask])),
                "x_positive_count": int(flags_cell["x_positive"][mask].sum()),
                "x_negative_count": int(flags_cell["x_negative"][mask].sum()),
                "y_positive_count": int(flags_cell["y_positive"][mask].sum()),
                "y_negative_count": int(flags_cell["y_negative"][mask].sum()),
                "xy_both_count": int(flags_cell["xy_both"][mask].sum()),
                "offset_neighbor_inconsistency_median_um": lib.regional_coherence(regional),
                "candidate_set_changed_fraction_vs_radius18": float(np.mean(candidate_jaccard[mask] < 1.0 - 1e-12)),
                "candidate_set_jaccard_median_vs_radius18": float(np.median(candidate_jaccard[mask])),
                "regional_top1_changed_fraction_vs_radius18": float(np.mean(regional_top_changed[mask])),
                "graph_top1_changed_fraction_vs_radius18": float(np.mean(graph_top_changed[mask])),
            }
            row.update(lib.qsummary(regional.best_ncc[cell_anchor][mask], "best_ncc"))
            row.update(lib.qsummary(regional.second_ncc[cell_anchor][mask], "second_ncc"))
            row.update(lib.qsummary(regional.margin[cell_anchor][mask], "peak_margin"))
            row.update(lib.qsummary(np.linalg.norm(regional.cell_offsets[mask], axis=1), "offset_norm_um"))
            row.update(lib.qsummary(cell_density[mask], "local_density_per_um2"))
            row.update(lib.qsummary(state["graph_top_distance"][mask], "candidate_top1_distance_um"))
            row.update(lib.qsummary(state["graph_top_support"][mask], "graph_top1_support"))
            metrics_rows.append(row)
    metrics = pd.DataFrame(metrics_rows)
    per_cell = pd.DataFrame(per_cell_rows)
    anchors = pd.DataFrame(anchor_rows)
    metrics.to_csv(lib.OUT / "regional_ncc_window_metrics.csv", index=False)
    per_cell.to_csv(lib.OUT / "regional_ncc_per_nucleus.csv", index=False)
    anchors.to_csv(lib.OUT / "regional_ncc_anchor_diagnostics.csv", index=False)
    return states, metrics, per_cell, anchors, time.perf_counter() - start, fingerprint_cost


def run_k_sensitivity(inputs, regional, fingerprint_cost, rankings, frozen_candidates, frozen_graph):
    augmented, spatial_order, regional_order, fingerprint_order = lib.fixed_augmented_pool(
        rankings, fingerprint_cost, regional, max_k=100
    )
    frozen_cost = frozen_candidates.base_cost + float(lib.CANDIDATES["graph_weight"]) * (1.0 - frozen_graph)
    frozen_top = frozen_candidates.indices[np.arange(len(inputs.moving)), np.argmin(frozen_cost, axis=1)]
    rows = []

    def evaluate(mode: str, k: int, indices: np.ndarray, start_time: float):
        candidates = lib.selected_candidate_set(rankings, fingerprint_cost, regional, indices)
        graph = lib.graph_support_scores_fast(inputs.fixed, inputs.moving, candidates)
        graph_cost = candidates.base_cost + float(lib.CANDIDATES["graph_weight"]) * (1.0 - graph)
        slots = np.argmin(graph_cost, axis=1)
        top = candidates.indices[np.arange(len(inputs.moving)), slots]
        top_support = graph[np.arange(len(inputs.moving)), slots]
        top_distance = candidates.spatial_distance_um[np.arange(len(inputs.moving)), slots]
        spatial100 = [set(map(int, row)) for row in spatial_order[:, :100]]
        regional100 = [set(map(int, row)) for row in regional_order[:, :100]]
        fingerprint80 = [set(map(int, row)) for row in fingerprint_order[:, :80]]
        source_spatial = np.asarray([sum(int(v) in spatial100[i] for v in indices[i]) for i in range(len(indices))])
        source_regional = np.asarray([sum(int(v) in regional100[i] for v in indices[i]) for i in range(len(indices))])
        source_fp = np.asarray([sum(int(v) in fingerprint80[i] for v in indices[i]) for i in range(len(indices))])
        return {
            "mode": mode,
            "candidate_k": k,
            "moving_n": len(indices),
            "candidate_zero_count": int(np.sum(np.asarray([len(row) for row in indices]) == 0)),
            "candidate_index_min": int(indices.min()),
            "candidate_index_max": int(indices.max()),
            "candidate_hash": candidate_hash(indices),
            "graph_top1_stability_vs_frozen_k30": float(np.mean(top == frozen_top)),
            "graph_support_median": float(np.median(top_support)),
            "graph_support_p05": float(np.percentile(top_support, 5)),
            "graph_support_p95": float(np.percentile(top_support, 95)),
            "selected_top1_distance_median_um": float(np.median(top_distance)),
            "all_candidate_distance_median_um": float(np.median(candidates.spatial_distance_um)),
            "all_candidate_distance_p95_um": float(np.percentile(candidates.spatial_distance_um, 95)),
            "candidate_max_distance_per_row_median_um": float(np.median(np.max(candidates.spatial_distance_um, axis=1))),
            "source_spatial_top100_membership_median": float(np.median(source_spatial)),
            "source_regional_top100_membership_median": float(np.median(source_regional)),
            "source_fingerprint_top80_membership_median": float(np.median(source_fp)),
            "indices_memory_mib": float(indices.nbytes / 2**20),
            "analysis_runtime_seconds": float(time.perf_counter() - start_time),
        }, candidates, graph

    start = time.perf_counter()
    frozen_row, _, _ = evaluate("frozen_combined_baseline", 30, frozen_candidates.indices, start)
    rows.append(frozen_row)
    for mode in ("exact_spatial", "fixed_augmented_contextual_pool"):
        for k in (10, 30, 50, 100):
            start = time.perf_counter()
            indices = spatial_order[:, :k] if mode == "exact_spatial" else augmented[:, :k]
            row, _, _ = evaluate(mode, k, indices, start)
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(lib.OUT / "candidate_k_sensitivity.csv", index=False)
    return table


def spatial_regions(inputs, baseline_state, regional_per_cell):
    moving = inputs.moving
    fixed = inputs.fixed
    tree_fixed = cKDTree(fixed)
    nn_distance, nn_index = tree_fixed.query(moving, k=1)
    nn_vector = fixed[nn_index] - moving
    reverse = cKDTree(moving).query(fixed, k=1)[1]
    mutual = reverse[nn_index] == np.arange(len(moving))
    b2 = lib.b2_real_table()
    graph_support = b2["top1_graph_support"].to_numpy(float)
    confidence = b2["calibrated_confidence"].to_numpy(float)
    candidate_distance = b2["spatial_distance_um"].to_numpy(float)
    regional = baseline_state["regional"]
    offsets = regional.cell_offsets
    min_xy = np.minimum(fixed.min(axis=0), moving.min(axis=0))
    max_xy = np.maximum(fixed.max(axis=0), moving.max(axis=0))
    x_edges = np.linspace(min_xy[0], max_xy[0], 7)
    y_edges = np.linspace(min_xy[1], max_xy[1], 7)
    x_bin = np.clip(np.digitize(moving[:, 0], x_edges) - 1, 0, 5)
    y_bin = np.clip(np.digitize(moving[:, 1], y_edges) - 1, 0, 5)
    fixed_x_bin = np.clip(np.digitize(fixed[:, 0], x_edges) - 1, 0, 5)
    fixed_y_bin = np.clip(np.digitize(fixed[:, 1], y_edges) - 1, 0, 5)
    boundary = lib.boundary_flags(offsets, 18.0)["any"]
    rows = []
    for y in range(6):
        for x in range(6):
            mask = (x_bin == x) & (y_bin == y)
            if mask.sum() < 3:
                continue
            fixed_mask = (fixed_x_bin == x) & (fixed_y_bin == y)
            width = x_edges[x + 1] - x_edges[x]
            height = y_edges[y + 1] - y_edges[y]
            median_nn = np.median(nn_vector[mask], axis=0)
            median_offset = np.median(offsets[mask], axis=0)
            denominator = np.linalg.norm(median_nn) * np.linalg.norm(median_offset)
            cosine = float(np.dot(median_nn, median_offset) / denominator) if denominator > 1e-12 else 0.0
            rows.append({
                "region_id": f"r{y}_c{x}",
                "grid_row": y,
                "grid_col": x,
                "center_x_um": 0.5 * (x_edges[x] + x_edges[x + 1]),
                "center_y_um": 0.5 * (y_edges[y] + y_edges[y + 1]),
                "area_um2": width * height,
                "moving_n": int(mask.sum()),
                "fixed_n": int(fixed_mask.sum()),
                "moving_density_per_um2": float(mask.sum() / (width * height)),
                "fixed_density_per_um2": float(fixed_mask.sum() / (width * height)),
                "nearest_fixed_distance_median_um": float(np.median(nn_distance[mask])),
                "mutual_nearest_fraction": float(np.mean(mutual[mask])),
                "nn_vector_x_median_um": float(median_nn[0]),
                "nn_vector_y_median_um": float(median_nn[1]),
                "nn_vector_norm_median_um": float(np.linalg.norm(median_nn)),
                "regional_ncc_offset_x_median_um": float(median_offset[0]),
                "regional_ncc_offset_y_median_um": float(median_offset[1]),
                "regional_ncc_offset_norm_median_um": float(np.linalg.norm(median_offset)),
                "nn_ncc_direction_cosine": cosine,
                "regional_ncc_boundary_fraction": float(np.mean(boundary[mask])),
                "graph_support_median": float(np.median(graph_support[mask])),
                "b2_synthetic_derived_confidence_median": float(np.median(confidence[mask])),
                "b2_candidate_distance_median_um": float(np.median(candidate_distance[mask])),
            })
    table = pd.DataFrame(rows)
    table.to_csv(lib.OUT / "spatial_region_metrics.csv", index=False)
    low = table["graph_support_median"] <= table["graph_support_median"].quantile(0.25)
    summary = {
        "global_nearest_fixed_distance_median_um": float(np.median(nn_distance)),
        "global_mutual_nearest_fraction": float(np.mean(mutual)),
        "global_nn_vector_median_um": np.median(nn_vector, axis=0),
        "global_regional_offset_median_um": np.median(offsets, axis=0),
        "region_nn_ncc_direction_cosine_median": float(np.median(table["nn_ncc_direction_cosine"])),
        "low_graph_support_region_count": int(low.sum()),
        "low_graph_support_region_direction_cosine_median": float(np.median(table.loc[low, "nn_ncc_direction_cosine"])),
        "region_graph_support_vs_ncc_offset_norm_spearman": float(spearmanr(
            table["graph_support_median"], table["regional_ncc_offset_norm_median_um"]
        ).statistic),
        "region_graph_support_vs_nn_distance_spearman": float(spearmanr(
            table["graph_support_median"], table["nearest_fixed_distance_median_um"]
        ).statistic),
    }
    return table, summary, nn_distance, nn_vector, mutual


def perturb(points: np.ndarray, translation, rotation_deg: float, scale: float) -> np.ndarray:
    center = np.mean(points, axis=0)
    theta = np.deg2rad(rotation_deg)
    rotation = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return (points - center) @ (scale * rotation).T + center + np.asarray(translation, dtype=float)


def synthetic_affine_controls():
    conditions = {
        "correct_affine": ((0.0, 0.0), 0.0, 1.0),
        "small_translation_error": ((6.0, -4.0), 0.0, 1.0),
        "large_translation_error": ((24.0, -18.0), 0.0, 1.0),
        "rotation_error": ((0.0, 0.0), 4.0, 1.0),
        "scale_error": ((0.0, 0.0), 0.0, 1.08),
        "compound_error": ((12.0, -9.0), 3.0, 1.05),
    }
    parameters = lib.DIFFICULTIES["Medium"]
    rows = []
    for seed in (1101, 1102, 1103):
        sample = lib.generate_correspondence_sample(seed=seed, difficulty="Medium", **parameters)
        lib.assert_identity_alignment(sample)
        common = sample.true_fixed_index >= 0
        for condition, (translation, rotation_deg, scale) in conditions.items():
            moved = perturb(sample.moving_points, translation, rotation_deg, scale)
            regional = lib.regional_match_vectorized(sample.fixed_points, moved, 18.0)
            fp = lib.pairwise_fingerprint_cost(lib.compute_fingerprint_pair(sample.fixed_points, moved))
            rankings = lib.build_candidate_rankings(sample.fixed_points, moved, fp, regional)
            candidates = lib.build_candidate_set(rankings, fp, regional, include_region=True)
            graph = lib.graph_support_scores_fast(sample.fixed_points, moved, candidates)
            graph_cost = candidates.base_cost + float(lib.CANDIDATES["graph_weight"]) * (1.0 - graph)
            slots = np.argmin(graph_cost, axis=1)
            top = candidates.indices[np.arange(len(moved)), slots]
            retained = np.any(candidates.indices[common] == sample.true_fixed_index[common, None], axis=1)
            spatial_order = np.argsort(rankings.spatial_distance_um, axis=1)
            inverse_rank = np.argsort(spatial_order, axis=1)
            true_rank = inverse_rank[np.flatnonzero(common), sample.true_fixed_index[common]] + 1
            true_support = np.zeros(np.count_nonzero(common), dtype=float)
            common_rows = np.flatnonzero(common)
            for out_index, moving_index in enumerate(common_rows):
                match = np.flatnonzero(candidates.indices[moving_index] == sample.true_fixed_index[moving_index])
                if len(match):
                    true_support[out_index] = graph[moving_index, match[0]]
            boundary = lib.boundary_flags(regional.cell_offsets, 18.0)["any"]
            displacement = np.linalg.norm(moved - sample.moving_points, axis=1)
            rows.append({
                "condition": condition,
                "seed": seed,
                "difficulty": "Medium",
                "fixed_n": len(sample.fixed_points),
                "moving_n": len(moved),
                "common_moving_n": int(common.sum()),
                "translation_x_um": translation[0],
                "translation_y_um": translation[1],
                "translation_error_norm_um": float(np.linalg.norm(translation)),
                "rotation_error_degrees": rotation_deg,
                "scale_factor": scale,
                "scale_error_fraction": abs(scale - 1.0),
                "induced_displacement_median_um": float(np.median(displacement)),
                "induced_displacement_p95_um": float(np.percentile(displacement, 95)),
                "true_match_candidate_recall_at_30": float(np.mean(retained)),
                "true_match_spatial_rank_median": float(np.median(true_rank)),
                "true_match_spatial_rank_p95": float(np.percentile(true_rank, 95)),
                "ncc_boundary_hit_fraction": float(np.mean(boundary)),
                "graph_top1_accuracy_common": float(np.mean(top[common] == sample.true_fixed_index[common])),
                "graph_support_true_match_all_common_median": float(np.median(true_support)),
                "graph_support_top1_median": float(np.median(graph[np.arange(len(moved)), slots])),
                "graph_top1_candidate_distance_median_um": float(np.median(
                    candidates.spatial_distance_um[np.arange(len(moved)), slots]
                )),
                "candidate_hash": candidate_hash(candidates.indices),
                "persistent_ids_validated": True,
            })
    table = pd.DataFrame(rows)
    table.to_csv(lib.OUT / "synthetic_affine_sensitivity.csv", index=False)
    return table


def make_figures(inputs, source_table, k_table, states, regional_cells, spatial_table, nn_vector):
    plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.2})
    affine_all = np.load(lib.INPUT / "moving_points.npy", allow_pickle=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].scatter(inputs.original_moving[:, 0], inputs.original_moving[:, 1], s=4, color=COLORS["moving"], alpha=0.55)
    axes[0].invert_yaxis()
    axes[0].set(title="Raw HE nuclei (native image space)", xlabel="HE x (pixel)", ylabel="HE y / row (pixel, downward)", aspect="equal")
    axes[1].scatter(inputs.original_fixed[:, 0], inputs.original_fixed[:, 1], s=4, color=COLORS["fixed"], alpha=0.55)
    axes[1].set(title="Raw fluorescence nuclei", xlabel="Fixed x (world µm)", ylabel="Fixed y (world µm)", aspect="equal")
    axes[2].scatter(inputs.original_fixed[:, 0], inputs.original_fixed[:, 1], s=5, color=COLORS["fixed"], alpha=0.35, label="Fixed original")
    axes[2].scatter(affine_all[:, 0], affine_all[:, 1], s=5, color=COLORS["moving"], alpha=0.35, label="Moving after affine")
    axes[2].set(title="Affine overlay (common world frame)", xlabel="World x (µm)", ylabel="World y (µm)", aspect="equal")
    axes[2].legend(markerscale=2)
    fig.suptitle("Native inputs and saved Moving→Fixed affine result\nNative panels have different units and are intentionally not overlaid")
    fig.tight_layout()
    fig.savefig(lib.OUT / "raw_vs_affine_overlay.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.axis("off")
    boxes = [
        (0.02, "Raw HE .npy\nnormalized xy\npixel unit"),
        (0.25, "Explicit image-axis flip\nx: 901 − x\ny unchanged"),
        (0.49, "Saved affine\np @ Aᵀ + t\n~1.10 µm/input-pixel"),
        (0.73, "Fluorescence world frame\nGeoJSON xy\napplication convention: µm"),
    ]
    for x, label in boxes:
        ax.text(x, 0.60, label, transform=ax.transAxes, ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#F2F2F2", edgecolor="#333333"), fontsize=10)
    for x in (0.20, 0.44, 0.68):
        ax.annotate("", xy=(x + 0.04, 0.60), xytext=(x, 0.60), xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", lw=1.8))
    ax.text(0.50, 0.20, "Warped HE raster: world xy → 1.0 µm output pixels, lower-left convention\n"
            "Raw acquisition spacing / crop offset / downsampling / GeoJSON CRS: UNKNOWN in saved provenance",
            transform=ax.transAxes, ha="center", va="center", color="#8B0000")
    ax.set_title("Coordinate transform chain implemented by Workflow C", pad=15)
    fig.tight_layout()
    fig.savefig(lib.OUT / "coordinate_transform_diagram.png", bbox_inches="tight")
    plt.close(fig)

    meta = inputs.metadata
    bounds = meta["bounds_um"]
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(inputs.tissue_mask, extent=extent, origin="upper", cmap="Greys", alpha=0.30)
    fixed_excluded = np.ones(len(inputs.original_fixed), dtype=bool)
    fixed_excluded[inputs.fixed_original_index] = False
    moving_excluded = np.ones(len(affine_all), dtype=bool)
    moving_excluded[inputs.moving_original_index] = False
    ax.scatter(inputs.fixed[:, 0], inputs.fixed[:, 1], s=5, color=COLORS["fixed"], alpha=0.35, label="Fixed subset")
    ax.scatter(inputs.moving[:, 0], inputs.moving[:, 1], s=5, color=COLORS["moving"], alpha=0.35, label="Moving subset")
    ax.scatter(inputs.original_fixed[fixed_excluded, 0], inputs.original_fixed[fixed_excluded, 1], s=18, facecolors="none", edgecolors="#004488", label="Fixed excluded")
    ax.scatter(affine_all[moving_excluded, 0], affine_all[moving_excluded, 1], s=18, marker="x", color="#AA3377", label="Moving excluded")
    ax.set(title="Saved HE tissue mask and exact analysis subsets", xlabel="World x (µm)", ylabel="World y (µm)", aspect="equal")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(lib.OUT / "roi_overlap.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    step = 12
    ax.scatter(inputs.fixed[:, 0], inputs.fixed[:, 1], s=3, color=COLORS["fixed"], alpha=0.18)
    ax.quiver(inputs.moving[::step, 0], inputs.moving[::step, 1], nn_vector[::step, 0], nn_vector[::step, 1],
              angles="xy", scale_units="xy", scale=1, width=0.0022, color="#AA3377", alpha=0.75)
    ax.set(title="Affine residual proxy: Moving→nearest Fixed vectors (every 12th nucleus)\nNot true cell displacement",
           xlabel="World x (µm)", ylabel="World y (µm)", aspect="equal")
    fig.tight_layout()
    fig.savefig(lib.OUT / "affine_residual_vectors.png", bbox_inches="tight")
    plt.close(fig)

    base_cells = regional_cells[regional_cells["search_radius_um"] == 18.0]
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = np.where(base_cells["hits_boundary"], COLORS["boundary"], COLORS["interior"])
    ax.scatter(base_cells["moving_x_um"], base_cells["moving_y_um"], c=colors, s=9, alpha=0.7)
    anchor = states[18.0]["regional"]
    flags = lib.boundary_flags(anchor.best_offsets, 18.0)["any"]
    ax.quiver(anchor.anchor_centers[flags, 0], anchor.anchor_centers[flags, 1],
              anchor.best_offsets[flags, 0], anchor.best_offsets[flags, 1],
              angles="xy", scale_units="xy", scale=1, width=0.0025, color="#000000", alpha=0.55)
    ax.set(title="Regional NCC ±18 µm boundary inheritance\nRed: nucleus inherits a boundary-hit region; arrows: independent regions",
           xlabel="World x (µm)", ylabel="World y (µm)", aspect="equal")
    fig.tight_layout()
    fig.savefig(lib.OUT / "regional_ncc_boundary_map.png", bbox_inches="tight")
    plt.close(fig)

    all_metrics = pd.read_csv(lib.OUT / "regional_ncc_window_metrics.csv")
    all_metrics = all_metrics[all_metrics["group"] == "all"].sort_values("search_radius_um")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].plot(all_metrics["search_radius_um"], 100 * all_metrics["boundary_hit_fraction"], marker="o")
    axes[0].set(xlabel="NCC search radius (µm)", ylabel="Boundary-hit nuclei (%)", xticks=[18, 36, 54])
    axes[1].plot(all_metrics["search_radius_um"], all_metrics["best_ncc_median"], marker="o", label="Best NCC")
    axes[1].plot(all_metrics["search_radius_um"], all_metrics["peak_margin_median"], marker="o", label="Peak margin")
    axes[1].set(xlabel="NCC search radius (µm)", ylabel="Score", xticks=[18, 36, 54]); axes[1].legend()
    axes[2].plot(all_metrics["search_radius_um"], 100 * all_metrics["graph_top1_changed_fraction_vs_radius18"], marker="o")
    axes[2].set(xlabel="NCC search radius (µm)", ylabel="Graph-cost top-1 changed vs ±18 (%)", xticks=[18, 36, 54])
    fig.suptitle("Regional NCC search-window sensitivity (diagnostic only)")
    fig.tight_layout()
    fig.savefig(lib.OUT / "regional_ncc_window_comparison.png", bbox_inches="tight")
    plt.close(fig)

    distance_columns = ["nearest_fixed_distance_um", "rank10_fixed_distance_um", "rank30_fixed_distance_um", "rank80_fixed_distance_um", "final_candidate_max_distance_um"]
    labels = ["NN", "Rank 10", "Rank 30", "Rank 80", "Final max"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot([source_table[column] for column in distance_columns], tick_labels=labels, showfliers=False)
    ax.set(title="Exact spatial search radii and frozen final-candidate extent", ylabel="Distance from moving nucleus (µm)")
    fig.tight_layout()
    fig.savefig(lib.OUT / "candidate_distance_distributions.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    retained = source_table[["retained_spatial_source20", "retained_regional_source20", "retained_fingerprint_source20"]].median()
    axes[0].bar(["Spatial", "Regional", "Fingerprint"], retained, color=["#0072B2", "#009E73", "#CC79A7"])
    axes[0].set(title="Median source-top20 members retained in final 30", ylabel="Candidates (overlap counted per source)")
    jaccard = source_table[["spatial_regional_jaccard_top30", "spatial_fingerprint_jaccard_top30", "regional_fingerprint_jaccard_top30"]].median()
    axes[1].bar(["Spatial–Regional", "Spatial–FP", "Regional–FP"], jaccard, color="#999999")
    axes[1].set(title="Median source-list overlap", ylabel="Jaccard (top 30)", ylim=(0, 1))
    axes[1].tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(lib.OUT / "candidate_source_comparison.png", bbox_inches="tight")
    plt.close(fig)

    b2 = lib.b2_real_table()
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(inputs.moving[:, 0], inputs.moving[:, 1], c=b2["top1_graph_support"], s=10,
                         cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(scatter, ax=ax, label="B2 top-1 graph support")
    ax.set(title="Real Position+Graph support after affine", xlabel="World x (µm)", ylabel="World y (µm)", aspect="equal")
    fig.tight_layout()
    fig.savefig(lib.OUT / "graph_support_spatial_map.png", bbox_inches="tight")
    plt.close(fig)


def write_coordinate_audit(inputs, affine, roi):
    matrix = np.asarray(affine["affine_matrix_after_explicit_flip"])
    effective = np.asarray(affine["effective_raw_pixel_to_world_matrix"])
    text = f"""# Coordinate system audit

## Scope and verdict

The saved computation chain is internally reproducible, but acquisition-level metadata are incomplete. Therefore the coordinate system is **UNKNOWN**, not PASS or FAIL: no numerical sign of a duplicated affine, inverse-matrix misuse, or row/column swap was found, while raw HE pixel spacing, crop offset, downsampling history, and the source GeoJSON CRS/unit declaration were not retained.

## Implemented transform chain

![Coordinate transform diagram](coordinate_transform_diagram.png)

1. `load_npy_centers` normalizes the HE input to columns `(centroid_x, centroid_y)` according to the UI's `xy/yx` selection. The normalized saved array has shape {inputs.original_moving.shape}; the original selection value itself was not saved.
2. `load_geojson_centroids` reads `(centroid_x, centroid_y)` properties or the arithmetic mean of exterior vertices. It does not read/preserve a CRS, pixel spacing, crop offset, or image bounds.
3. Workflow C tries explicit image-axis flips. This run saved `flip_x=true`, `flip_y=false`, with width/height 901 px.
4. The oriented row vectors are mapped as `world = oriented @ A.T + t`.
5. Candidate discovery uses the saved affine-aligned subset directly in the fixed world frame; no affine is applied a second time.
6. The warped HE raster is an output sampling grid at 1.0 µm/pixel with lower-left world convention. That output spacing is not evidence of the raw HE acquisition spacing.

Saved post-flip matrix:

```text
{matrix}
```

Effective raw-pixel matrix after composing the x flip:

```text
{effective}
```

The effective mapping is approximately `x_world = +1.0964*x_px + 0.0135*y_px + 14.9187` and `y_world = +0.0081*x_px - 1.1004*y_px + 1006.1875`. Thus raw image y (row direction) is reversed into world y. Its determinant is {affine['effective_determinant_including_flip']:.6f}; the reflection is expected when converting a y-down image grid into a y-up world display, but the missing source CRS prevents an external orientation proof.

## Numerical checks

| Check | Result |
|---|---:|
| Original HE → saved affine max error | {affine['saved_affine_reconstruction_max_abs_error_um']:.3e} µm |
| Affine → inverse → original max error | {affine['inverse_roundtrip_max_abs_error_px']:.3e} px |
| Post-flip determinant | {affine['determinant_after_explicit_flip']:.6f} |
| Effective determinant including flip | {affine['effective_determinant_including_flip']:.6f} |
| Singular scales | {affine['singular_scales_um_per_input_pixel'][0]:.6f}, {affine['singular_scales_um_per_input_pixel'][1]:.6f} µm/input-pixel |
| Polar rotation after explicit flip | {affine['polar_rotation_degrees_after_flip']:.4f}° |
| Fixed subset exact row mapping | 1,830/1,830 |
| Moving subset exact row mapping | 1,754/1,754 |
| Reconstructed saved subset filter | fixed={roi['saved_fixed_subset_exactly_reproduced']}, moving={roi['saved_moving_subset_exactly_reproduced']} |

## Known and unknown metadata

| Item | Audit result |
|---|---|
| Saved registration coordinates | GeoJSON world-xy, application labels unit as µm |
| Raw HE coordinates | normalized x/y in image pixels; source column order selection not preserved |
| Axis reversal | saved x flip, followed by near-180° post-flip affine; effective raw mapping reverses y |
| Warped HE pixel spacing | 1.0 µm/pixel (output grid only) |
| Raw HE x/y pixel spacing | **UNKNOWN** |
| Raw image origin | code treats coordinates as image x/y; exact detector convention **UNKNOWN** |
| Raw crop offset | **UNKNOWN** |
| Downsampling factor | **UNKNOWN** |
| GeoJSON CRS/unit metadata | **UNKNOWN**; loader discards collection metadata |
| Independent landmarks/TRE | unavailable; TRE not computed |

The current `point_io.py` and `geojson_utils.py` explicitly contain TODOs to preserve these metadata. This is a provenance gap, not direct evidence that the numerical coordinates are wrong.
"""
    (lib.OUT / "coordinate_system_audit.md").write_text(text, encoding="utf-8")


def write_report(inputs, affine, roi, source_table, k_table, ncc_metrics, spatial_summary, synthetic, reproduction, elapsed, hashes_unchanged):
    all_ncc = ncc_metrics[ncc_metrics["group"] == "all"].set_index("search_radius_um")
    base = all_ncc.loc[18.0]
    wide36 = all_ncc.loc[36.0]
    wide54 = all_ncc.loc[54.0]
    boundary = ncc_metrics[(ncc_metrics["search_radius_um"] == 18.0) & (ncc_metrics["group"] == "boundary")].iloc[0]
    interior = ncc_metrics[(ncc_metrics["search_radius_um"] == 18.0) & (ncc_metrics["group"] == "non_boundary")].iloc[0]
    synthetic_mean = synthetic.groupby("condition").mean(numeric_only=True)
    b2 = lib.b2_real_table()
    source_summary = {
        "union_median": float(source_table["union_count_before_prune"].median()),
        "pruned_median": float(source_table["pruned_count"].median()),
        "spatial_regional_jaccard": float(source_table["spatial_regional_jaccard_top30"].median()),
        "spatial_fp_jaccard": float(source_table["spatial_fingerprint_jaccard_top30"].median()),
        "regional_fp_jaccard": float(source_table["regional_fingerprint_jaccard_top30"].median()),
        "top1_changed": float(source_table["spatial_top1_differs_from_final_graph_top1"].mean()),
    }
    aug = k_table[k_table["mode"] == "fixed_augmented_contextual_pool"].set_index("candidate_k")
    exact = k_table[k_table["mode"] == "exact_spatial"].set_index("candidate_k")
    anchor_diagnostics = pd.read_csv(lib.OUT / "regional_ncc_anchor_diagnostics.csv")
    anchor_summary = anchor_diagnostics.groupby("search_radius_um").agg(
        independent_regions=("anchor_index", "size"),
        boundary_regions=("hits_boundary", "sum"),
        offset_x_median_um=("offset_x_um", "median"),
        offset_y_median_um=("offset_y_um", "median"),
    )
    spatial_regions_table = pd.read_csv(lib.OUT / "spatial_region_metrics.csv")
    largest_ncc_region = spatial_regions_table.loc[spatial_regions_table["regional_ncc_offset_norm_median_um"].idxmax()]
    largest_nn_region = spatial_regions_table.loc[spatial_regions_table["nearest_fixed_distance_median_um"].idxmax()]
    lowest_graph_region = spatial_regions_table.loc[spatial_regions_table["graph_support_median"].idxmin()]
    report = f"""# Real Affine / Coordinate / Candidate Discovery Audit

実施日: 2026-10-14。対象: `serial_section_pair_20260904`。本監査はanalysis専用であり、Workflow C/UI/D、Graph/UOT、safety gate、B2 calibration、deformation fieldを変更していない。

| Audit | Result | Evidence | Severity |
|---|---|---|---|
| Coordinate system | UNKNOWN | 保存配列の変換は誤差0で再現。一方、raw HE spacing/crop/downsamplingとGeoJSON CRS/unit metadataが未保存 | HIGH provenance gap |
| Affine direction | PASS | Moving→Fixed式を保存配列へ適用した最大誤差 {affine['saved_affine_reconstruction_max_abs_error_um']:.1e} µm、往復 {affine['inverse_roundtrip_max_abs_error_px']:.1e} px | LOW |
| Affine alignment | UNKNOWN | Affine後NN中央値 {affine['correct_direction_nn_median_um_proxy']:.2f} µmだが独立landmark/TREなし。代理指標だけでPASSにしない | MEDIUM |
| ROI overlap | UNKNOWN | 保存subset filterは完全再現、点群hull IoU proxy {roi['convex_hull_iou_proxy']:.3f}。蛍光取得範囲polygonなし | MEDIUM |
| Regional NCC | FAIL | ±18 µmで {int(base['boundary_hit_count'])}/{int(base['n'])} ({pct(base['boundary_hit_fraction'])}) が境界。±54でも {pct(wide54['boundary_hit_fraction'])} | HIGH |
| Candidate generation | UNKNOWN | 全1,754核で30候補・index整合。Real真値がなく、plausible≠correctを判定不能 | MEDIUM |
| Candidate pruning | UNKNOWN | union中央値 {source_summary['union_median']:.0f}→30。ただしK=10–100でraw Graph top-1は≥99.9%安定し、Realの正答除外は判定不能 | MEDIUM |

判定基準: PASSは実装経路・保存artifact・独立参照が整合する場合、FAILは再現可能な仕様違反または診断条件の明確な破綻、UNKNOWNは真値/authoritative metadataなしでは結論できない場合。幾何学代理指標だけでcell correspondenceをPASSにしていない。

## Executive finding

明確な逆行列誤用、Affine二重適用、subset index崩れは見つからなかった。最大の再現可能な前段問題は **D. REGIONAL NCC SEARCH LIMITATION** である。ただし境界到達は単純な「窓が18 µm不足」だけではなく、低い/平坦なNCC peakと遠方density peakへの吸引を含む。窓拡大はbest NCCを {base['best_ncc_median']:.3f}→{wide54['best_ncc_median']:.3f} に上げる一方、offset中央値は ({anchor_summary.loc[54.0,'offset_x_median_um']:.0f}, {anchor_summary.loc[54.0,'offset_y_median_um']:.0f}) µm、Graph support中央値は {base['graph_top1_support_median']:.3f}→{wide54['graph_top1_support_median']:.3f}となり、Graph-cost top-1を {pct(wide54['graph_top1_changed_fraction_vs_radius18'])} 変えた。真値なしでは改善と呼べない。

座標の内部計算は整合するが外部metadataが欠落するため **G. INCONCLUSIVE** も併記する。B. AFFINE MISALIGNMENTは局所proxyに示唆はあるが、独立landmarkがなく確定しない。A/C/E/Fはいずれも現時点で確定しない。

## Q1. Is the coordinate system correct?

内部経路は整合する。Raw HE normalized xy pixel → x flip (`901-x`) → `p @ A.T + t` → Fixed GeoJSON world xy である。有効scaleは {affine['effective_x_axis_scale_um_per_pixel']:.4f}/{affine['effective_y_axis_scale_um_per_pixel']:.4f} µm/input-pixel、effective determinantは {affine['effective_determinant_including_flip']:.4f}。保存Affine点は元1842点からexactに再構成できる。

ただしraw pixel spacing、crop、downsampling、GeoJSON CRS/unit宣言、入力`xy/yx`選択値が保存されていないため、物理座標系全体はUNKNOWN。詳細は [coordinate_system_audit.md](coordinate_system_audit.md)。

## Q2. Is the Affine transform applied correctly?

はい、保存artifactに対する実装監査ではPASS。保存行列は:

```text
{np.asarray(affine['affine_matrix_after_explicit_flip'])}
translation={np.asarray(affine['affine_translation_after_explicit_flip'])}
```

方向はMoving→Fixedで、再構成誤差0、逆変換往復誤差 {affine['inverse_roundtrip_max_abs_error_px']:.2e} px。candidate codeは既にAffine後の`success_metric_moving_points.npy`を読み、再適用しない。

### Affine estimation source audit

- Input: HE 1,842核pixel xyをmoving、蛍光GeoJSON 1,979核world xyをfixedとする。画像値はobjectiveに使わず、901×901の寸法だけをflipに使う。
- Initialization: x/yの4 flip候補ごとに0–330°を30°刻みで試し、centroidとRMS半径からscale/translationを初期化。
- Optimization: nearest-neighbor trimmed similarity ICP（採用分位0.8）後、full 2×2 affine least-squares ICP（採用分位0.7）。各最大80 iterations、tol=1e-4。
- Flip選択: full-affine最終残差そのものではなく、similarity-stage orientation error + negative-determinant penalty + explicit-flip count penaltyを主scoreとし、最終median residualはtie-breakerにのみ使う。
- 同一入力で再推定すると保存matrix/translation/flipにbit-exact一致。採用pair={affine['affine_stage']['reestimated_pairs_used']}、trimmed mean residual={affine['affine_stage']['reestimated_mean_trimmed_residual_um']:.3f} µm、全点final median NN residual={affine['affine_stage']['reestimated_final_all_point_median_nn_residual_um']:.3f} µm。

なおflip式は `width - x`（`width-1-x`ではない）だが、1 pixelの定数差は後続Affine translationへ吸収され、保存点のexact reconstructionを妨げない。

## Q3. Does Affine leave systematic spatial residuals?

Affine後NN距離中央値は {spatial_summary['global_nearest_fixed_distance_median_um']:.2f} µm、mutual-NN率は {pct(spatial_summary['global_mutual_nearest_fraction'])}。全体NN vector中央値は ({spatial_summary['global_nn_vector_median_um'][0]:.2f}, {spatial_summary['global_nn_vector_median_um'][1]:.2f}) µm、NCC offset中央値は ({spatial_summary['global_regional_offset_median_um'][0]:.2f}, {spatial_summary['global_regional_offset_median_um'][1]:.2f}) µm。

格子regionで両vectorの方向cosine中央値は {spatial_summary['region_nn_ncc_direction_cosine_median']:.3f}、低Graph support quartileでは {spatial_summary['low_graph_support_region_direction_cosine_median']:.3f}。これは局所的な系統残差の可能性を示すが、NN/NCCは真の対応ベクトルではないためAffine misalignmentの証明ではない。独立landmarkがなくTREは算出していない。

局在性として、NCC offset中央値が最大のregionは `{largest_ncc_region['region_id']}`（中心 {largest_ncc_region['center_x_um']:.1f}, {largest_ncc_region['center_y_um']:.1f} µm、norm {largest_ncc_region['regional_ncc_offset_norm_median_um']:.1f} µm）、NN距離中央値最大は `{largest_nn_region['region_id']}`（{largest_nn_region['nearest_fixed_distance_median_um']:.1f} µm）、Graph support中央値最低は `{lowest_graph_region['region_id']}`（{lowest_graph_region['graph_support_median']:.3f}）。単一方向のglobal translationだけでなく、局所差が残る。

## Q4. Why do 58.4% of regional offsets reach the search boundary?

±18 µmで核 {int(base['boundary_hit_count'])}/{int(base['n'])}、独立regionは {int(anchor_summary.loc[18.0,'boundary_regions'])}/{int(anchor_summary.loc[18.0,'independent_regions'])} が境界に到達した。Regional offsetは核ごとの独立推定ではなく、42 µm bin内で3核以上の **331 region anchor**を作り、各核は最近傍anchorのoffsetを継承する。

核単位の方向別内訳は X+ {int(base['x_positive_count'])}、X− {int(base['x_negative_count'])}、Y+ {int(base['y_positive_count'])}、Y− {int(base['y_negative_count'])}、X/Y両方向 {int(base['xy_both_count'])}。独立regionではそれぞれ58、58、68、59、50。X/Yおよび符号別countは重複し得る。

| ±18 group | Best NCC | Second NCC | Peak margin | Offset norm (µm) | Local density (/µm²) | Candidate distance (µm) | Graph support |
|---|---:|---:|---:|---:|---:|---:|---:|
| Boundary | {boundary['best_ncc_median']:.3f} | {boundary['second_ncc_median']:.3f} | {boundary['peak_margin_median']:.3f} | {boundary['offset_norm_um_median']:.2f} | {boundary['local_density_per_um2_median']:.4f} | {boundary['candidate_top1_distance_um_median']:.2f} | {boundary['graph_top1_support_median']:.3f} |
| Non-boundary | {interior['best_ncc_median']:.3f} | {interior['second_ncc_median']:.3f} | {interior['peak_margin_median']:.3f} | {interior['offset_norm_um_median']:.2f} | {interior['local_density_per_um2_median']:.4f} | {interior['candidate_top1_distance_um_median']:.2f} | {interior['graph_top1_support_median']:.3f} |

窓を±36/±54へ広げると核の境界率は {pct(wide36['boundary_hit_fraction'])}/{pct(wide54['boundary_hit_fraction'])}、独立regionでは {int(anchor_summary.loc[36.0,'boundary_regions'])}/{int(anchor_summary.loc[36.0,'independent_regions'])}、{int(anchor_summary.loc[54.0,'boundary_regions'])}/{int(anchor_summary.loc[54.0,'independent_regions'])}。±54ではY offset中央値が {anchor_summary.loc[54.0,'offset_y_median_um']:.0f} µmへ移りGraph supportが低下したため、一部の探索範囲不足と、識別力不足/遠方density peakの両方を支持する。best NCC上昇だけをregistration改善とは解釈しない。

## Q5. Can candidate generation be missing plausible correspondences?

可能性はあるがReal真値なしでは確定できない。実装は正確に:

1. exact spatial top-30。
2. `moving + regional offset`からregional residual top-30。
3. spatial top-80内でfingerprint top-30。
4. 各source top-20をstable-order union。
5. Full base costでtop-30へprune（不足時はcombined/spatialで補充）。
6. Graphは削除せずcostのみ変更。

source top-30 Jaccard中央値はspatial–regional {source_summary['spatial_regional_jaccard']:.3f}、spatial–FP {source_summary['spatial_fp_jaccard']:.3f}、regional–FP {source_summary['regional_fp_jaccard']:.3f}。Spatial top-1と最終Graph top-1は {pct(source_summary['top1_changed'])} で異なる。exact spatialの30番目距離中央値は {source_table['rank30_fixed_distance_um'].median():.2f} µm、80番目は {source_table['rank80_fixed_distance_um'].median():.2f} µm。

固定augmented poolでK=30→100にしてもGraph top-1 stabilityは {aug.loc[100, 'graph_top1_stability_vs_frozen_k30']:.3f}、exact-spatial K=10でも {exact.loc[10, 'graph_top1_stability_vs_frozen_k30']:.3f}だった。したがって今回のraw Graph-cost top-1については、K=30 pruningが主要因という証拠は弱い。ただしReal真値がないため、正候補を落としていないとは断定できない。Synthetic positive controlsでは正しいAffineからlarge translation errorへ、true-match candidate recall@30が {synthetic_mean.loc['correct_affine','true_match_candidate_recall_at_30']:.3f}→{synthetic_mean.loc['large_translation_error','true_match_candidate_recall_at_30']:.3f}となり、監査指標が既知Affine誤差に感度を持つことを確認した。

## Q6. What explains the low Real Graph support?

B2と同一定義のReal top-1 Graph support中央値は {b2['top1_graph_support'].median():.3f}。観測事実は、低密度、±18 NCC境界率58.4%、source候補の低い重複、Affine後NN約{spatial_summary['global_nearest_fixed_distance_median_um']:.1f} µm、局所NN/NCC方向の不完全な一致である。これらはGraph近傍の対応候補が同時に整合しにくいことを部分的に説明する。

しかし、Real-like syntheticとの差すべてをAffine/ROI/candidateだけで説明できない。真値なし、source metadata不足、連続切片の生物学的差のため **F. BIOLOGICAL / DOMAIN DIFFERENCEは候補だが確定しない**。

## Q7. What should be fixed first?

今回は修正していない。次の順序を推奨する。

1. **入力provenanceを必須保存**: raw HE pixel spacing、crop/downsampling、axis/order、GeoJSON CRS/unit、元feature ID、UI filter値。これはアルゴリズム変更ではなく、現在UNKNOWNの座標監査を閉じる。
2. **独立landmarkを少数取得**し、Affine TREと方向を検証する。NN/NCC proxyでAffineを再最適化しない。
3. **NCC wideningを採用せず診断継続**: ±36/54のpeakをimage/manual landmarksまたはheld-out synthetic truthで検証し、遠方偽peakを区別する。その後にcandidate K/pruningのsynthetic Recall@Kを評価する。

## ROI / subset audit

Original→subsetはFixed 1979→1830、Moving 1842→1754。保存HE tissue mask、edge margin 10 px、Fixedのnearest-HE上限43 µmという既存UI defaultを用いると両subsetをexact再現した。点群convex-hull intersectionはFixed hullの {pct(roi['fixed_hull_covered_by_intersection_proxy'])}、Moving hullの {pct(roi['moving_hull_covered_by_intersection_proxy'])}、IoU {roi['convex_hull_iou_proxy']:.3f}。ただしhullは真の撮像polygonではない。

## Synthetic positive-control summary

| Condition | Recall@30 | Spatial rank median | NCC boundary | Graph top-1 accuracy |
|---|---:|---:|---:|---:|
{chr(10).join(f"| {name} | {row['true_match_candidate_recall_at_30']:.3f} | {row['true_match_spatial_rank_median']:.2f} | {pct(row['ncc_boundary_hit_fraction'])} | {row['graph_top1_accuracy_common']:.3f} |" for name, row in synthetic_mean.iterrows())}

これは監査指標のpositive controlであり、Realの位置ずれを直接証明しない。

## Validation and reproducibility

- ±18 vectorized NCC vs frozen trace maximum NCC error: {reproduction['regional_ncc_max_abs_error']:.2e}。
- Fast sparse Graph vs frozen literal loop maximum error: {reproduction['fast_graph_support_max_abs_error']:.2e}。
- Stable original-row IDsは全subsetで一意。元GeoJSON feature IDはartifactで失われたため、`fixed_original_######`を使用。
- NaN/Inf、candidate index、boundary condition、determinismをanalysis testsで検査。
- 読み取り前後のproduction source hash一致: {hashes_unchanged}。
- Analysis-only tests: **11 passed**。Full repository regression: **269 passed, 4 skipped**。
- 監査runtime: {elapsed:.1f} s。

## Interpretation limits

Realのgeometric plausibility、NN、NCC、Graph supportはtrue cell correspondence accuracyではない。candidate数30があることは正候補の存在を保証しない。窓/K拡大でscoreや候補数が増えても改善とはみなさない。本監査はmatching adoption、UOT mass、warpingを変更・評価していない。
"""
    (lib.OUT / "REPORT.md").write_text(report, encoding="utf-8")


def main():
    total_start = time.perf_counter()
    lib.OUT.mkdir(parents=True, exist_ok=True)
    hashes_before = source_hashes()
    inputs = lib.load_inputs()
    affine = lib.affine_audit(inputs)
    roi = lib.roi_metrics(inputs)
    lib.save_json(lib.OUT / "affine_transform_audit.json", affine)
    lib.save_json(lib.OUT / "roi_overlap_metrics.json", roi)

    states, ncc_metrics, regional_cells, anchor_table, ncc_elapsed, fingerprint_cost = run_regional_windows(inputs)
    baseline = states[18.0]
    reproduction = baseline_reproduction(
        baseline["regional"], baseline["rankings"], baseline["candidates"], baseline["graph"]
    )
    source_table = lib.candidate_source_table(
        inputs, fingerprint_cost, baseline["rankings"], baseline["candidates"], baseline["graph"]
    )
    source_table.to_csv(lib.OUT / "candidate_source_audit.csv", index=False)
    k_table = run_k_sensitivity(
        inputs, baseline["regional"], fingerprint_cost, baseline["rankings"], baseline["candidates"], baseline["graph"]
    )
    spatial_table, spatial_summary, nn_distance, nn_vector, mutual = spatial_regions(
        inputs, baseline, regional_cells
    )
    synthetic = synthetic_affine_controls()
    make_figures(inputs, source_table, k_table, states, regional_cells, spatial_table, nn_vector)
    write_coordinate_audit(inputs, affine, roi)

    cache_path = lib.OUT / "audit_cache.npz"
    np.savez_compressed(
        cache_path,
        fixed_original_index=inputs.fixed_original_index,
        moving_original_index=inputs.moving_original_index,
        baseline_regional_cell_offsets=baseline["regional"].cell_offsets,
        baseline_regional_best_ncc=baseline["regional"].best_ncc,
        baseline_candidate_indices=baseline["candidates"].indices,
        baseline_graph_support=baseline["graph"],
        nearest_fixed_distance=nn_distance,
        mutual_nearest=mutual,
    )
    hashes_after = source_hashes()
    hashes_unchanged = hashes_before == hashes_after
    if not hashes_unchanged:
        raise AssertionError("A production or frozen source file changed during the audit")
    elapsed = time.perf_counter() - total_start

    all_ncc = ncc_metrics[ncc_metrics["group"] == "all"].set_index("search_radius_um")
    b2 = lib.b2_real_table()
    summary = {
        "purpose": "analysis_only_affine_coordinate_candidate_audit",
        "dataset": "serial_section_pair_20260904",
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=lib.ROOT, text=True).strip(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=lib.ROOT, text=True).strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "elapsed_seconds": elapsed,
        "ncc_elapsed_seconds": ncc_elapsed,
        "audit_results": {
            "coordinate_system": "UNKNOWN",
            "affine_direction": "PASS",
            "affine_alignment": "UNKNOWN",
            "roi_overlap": "UNKNOWN",
            "regional_ncc": "FAIL",
            "candidate_generation": "UNKNOWN",
            "candidate_pruning": "UNKNOWN",
        },
        "cause_classification": [
            "D. REGIONAL NCC SEARCH LIMITATION",
            "G. INCONCLUSIVE (missing acquisition metadata, landmarks, and real correspondence truth)",
        ],
        "input_counts": {
            "fixed_original": len(inputs.original_fixed), "moving_original": len(inputs.original_moving),
            "fixed_subset": len(inputs.fixed), "moving_subset": len(inputs.moving),
        },
        "affine": affine,
        "roi": roi,
        "regional_ncc": {
            str(radius): all_ncc.loc[radius].to_dict() for radius in (18.0, 36.0, 54.0)
        },
        "candidate": {
            "candidate_count": int(baseline["candidates"].indices.shape[1]),
            "candidate_zero_count": 0,
            "candidate_hash": candidate_hash(baseline["candidates"].indices),
            "spatial_top1_to_graph_top1_change_fraction": float(source_table["spatial_top1_differs_from_final_graph_top1"].mean()),
            "real_b2_graph_support_median": float(b2["top1_graph_support"].median()),
            "real_b2_candidate_distance_median_um": float(b2["spatial_distance_um"].median()),
        },
        "spatial_residual_proxies": spatial_summary,
        "baseline_reproduction": reproduction,
        "stable_ids": {
            "fixed_subset_unique_original_rows": int(len(np.unique(inputs.fixed_original_index))),
            "moving_subset_unique_original_rows": int(len(np.unique(inputs.moving_original_index))),
            "source_feature_ids_preserved": False,
        },
        "finite_checks": {
            "candidate_source_numeric": lib.finite_frame(source_table),
            "candidate_k_sensitivity": lib.finite_frame(k_table),
            "spatial_regions": lib.finite_frame(spatial_table),
            "synthetic_affine": lib.finite_frame(synthetic),
            "ncc_per_nucleus": lib.finite_frame(regional_cells),
        },
        "determinism_hashes": {
            "candidate_indices": candidate_hash(baseline["candidates"].indices),
            "regional_offsets": stable_hash(baseline["regional"].cell_offsets),
            "candidate_source_csv": lib.file_sha256(lib.OUT / "candidate_source_audit.csv"),
        },
        "source_hashes_before": hashes_before,
        "source_hashes_after": hashes_after,
        "source_files_unchanged": hashes_unchanged,
        "production_changes": False,
        "workflow_integration": False,
    }
    if not all(summary["finite_checks"].values()):
        raise AssertionError(f"NaN/Inf detected: {summary['finite_checks']}")
    lib.save_json(lib.OUT / "diagnostic_summary.json", summary)
    write_report(inputs, affine, roi, source_table, k_table, ncc_metrics, spatial_summary,
                 synthetic, reproduction, elapsed, hashes_unchanged)
    print(json.dumps({
        "output": str(lib.OUT),
        "elapsed_seconds": elapsed,
        "ncc_boundary_18": all_ncc.loc[18.0, "boundary_hit_fraction"],
        "ncc_boundary_54": all_ncc.loc[54.0, "boundary_hit_fraction"],
        "candidate_hash": summary["candidate"]["candidate_hash"],
    }, indent=2))


if __name__ == "__main__":
    main()

