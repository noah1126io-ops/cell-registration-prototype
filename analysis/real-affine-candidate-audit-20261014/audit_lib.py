"""Analysis-only helpers for the real affine/candidate audit.

Nothing in this module is imported by production Workflow C.  The functions
deliberately reuse the frozen contextual matching implementation and preserve
its equations while exposing intermediate values needed by the audit.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.linalg import polar
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import cdist, pdist


ROOT = Path("/mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu")
OUT = ROOT / "analysis/real-affine-candidate-audit-20261014"
INPUT = ROOT / "sweeps/real-grid27-20260908-v2/input"
RUN = ROOT / "runs/20260904-161247-698fe6c9"
CONTEXT = ROOT / "analysis/contextual-graph-uot-20261008"
DIAGNOSTICS = ROOT / "analysis/contextual-graph-uot-diagnostics-20261008"
ADOPTION = ROOT / "analysis/graph-matching-adoption-separation-20261014"

# These modules are the frozen analysis implementation, not production code.
sys.path.insert(0, str(CONTEXT))
from candidate_generation import (  # noqa: E402
    CandidateSet,
    build_candidate_rankings,
    build_candidate_set,
)
from config import CANDIDATES, DIFFICULTIES, GRAPH, REGIONAL  # noqa: E402
from graph_consistency import graph_support_scores, mutual_knn_graph  # noqa: E402
from local_fingerprint import compute_fingerprint_pair, pairwise_fingerprint_cost  # noqa: E402
from regional_pattern import (  # noqa: E402
    RegionalMatch,
    _anchors,
    _bounds,
    _channels,
    _sample_patch,
)
from synthetic_correspondence import (  # noqa: E402
    assert_identity_alignment,
    generate_correspondence_sample,
)
from src.pointset_registration import estimate_affine_with_y_flip  # noqa: E402


@dataclass(frozen=True)
class Inputs:
    original_fixed: np.ndarray
    fixed: np.ndarray
    original_moving: np.ndarray
    moving: np.ndarray
    metadata: dict
    tissue_mask: np.ndarray
    he_image: np.ndarray
    affine: dict
    fixed_original_index: np.ndarray
    moving_original_index: np.ndarray


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_points(name: str, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError(f"{name} must be finite (n, 2) points")
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError(f"{name} contains duplicate coordinates, so row IDs are not stable")
    return points


def exact_subset_indices(original: np.ndarray, subset: np.ndarray) -> np.ndarray:
    lookup = {tuple(point): index for index, point in enumerate(np.asarray(original))}
    indices = np.asarray([lookup.get(tuple(point), -1) for point in np.asarray(subset)], dtype=int)
    if np.any(indices < 0) or len(np.unique(indices)) != len(indices):
        raise AssertionError("Subset is not a unique exact-row subset of the original array")
    if not np.array_equal(original[indices], subset):
        raise AssertionError("Subset row reconstruction failed")
    return indices


def load_inputs() -> Inputs:
    original_fixed = validate_points(
        "original fixed", np.load(INPUT / "original_fixed_points.npy", allow_pickle=False)
    )
    fixed = validate_points(
        "fixed subset", np.load(INPUT / "success_metric_fixed_points.npy", allow_pickle=False)
    )
    original_moving = validate_points(
        "original moving", np.load(INPUT / "original_moving_points.npy", allow_pickle=False)
    )
    affine_moving = validate_points(
        "affine moving", np.load(INPUT / "moving_points.npy", allow_pickle=False)
    )
    moving = validate_points(
        "moving subset", np.load(INPUT / "success_metric_moving_points.npy", allow_pickle=False)
    )
    if original_fixed.shape != (1979, 2) or fixed.shape != (1830, 2):
        raise AssertionError("Unexpected fixed original/subset counts")
    if original_moving.shape != (1842, 2) or moving.shape != (1754, 2):
        raise AssertionError("Unexpected moving original/subset counts")
    config = json.loads((RUN / "config.json").read_text(encoding="utf-8"))
    metadata = json.loads((INPUT / "metadata.json").read_text(encoding="utf-8"))
    tissue_mask = np.load(INPUT / "tissue_mask.npy", allow_pickle=False)
    he_image = np.load(INPUT / "he_image.npy", allow_pickle=False)
    if tissue_mask.shape != he_image.shape[:2] or tissue_mask.dtype != np.bool_:
        raise AssertionError("Warped HE image and tissue mask are inconsistent")
    return Inputs(
        original_fixed=original_fixed,
        fixed=fixed,
        original_moving=original_moving,
        moving=moving,
        metadata=metadata,
        tissue_mask=tissue_mask,
        he_image=he_image,
        affine=config["affine"],
        fixed_original_index=exact_subset_indices(original_fixed, fixed),
        moving_original_index=exact_subset_indices(affine_moving, moving),
    )


def apply_saved_affine(raw: np.ndarray, affine: dict) -> np.ndarray:
    oriented = np.asarray(raw, dtype=float).copy()
    if affine["flip_x"]:
        oriented[:, 0] = float(affine["image_width"]) - oriented[:, 0]
    if affine["flip_y"]:
        oriented[:, 1] = float(affine["image_height"]) - oriented[:, 1]
    matrix = np.asarray(affine["matrix"], dtype=float)
    translation = np.asarray(affine["translation"], dtype=float)
    return oriented @ matrix.T + translation


def invert_saved_affine(world: np.ndarray, affine: dict) -> np.ndarray:
    matrix = np.asarray(affine["matrix"], dtype=float)
    translation = np.asarray(affine["translation"], dtype=float)
    oriented = (np.asarray(world, dtype=float) - translation) @ np.linalg.inv(matrix).T
    if affine["flip_x"]:
        oriented[:, 0] = float(affine["image_width"]) - oriented[:, 0]
    if affine["flip_y"]:
        oriented[:, 1] = float(affine["image_height"]) - oriented[:, 1]
    return oriented


def affine_audit(inputs: Inputs) -> dict:
    affine_all = np.load(INPUT / "moving_points.npy", allow_pickle=False)
    rebuilt = apply_saved_affine(inputs.original_moving, inputs.affine)
    roundtrip = invert_saved_affine(rebuilt, inputs.affine)
    matrix = np.asarray(inputs.affine["matrix"], dtype=float)
    translation = np.asarray(inputs.affine["translation"], dtype=float)
    flip_matrix = np.diag(
        [-1.0 if inputs.affine["flip_x"] else 1.0, -1.0 if inputs.affine["flip_y"] else 1.0]
    )
    flip_offset = np.asarray(
        [float(inputs.affine["image_width"]) if inputs.affine["flip_x"] else 0.0,
         float(inputs.affine["image_height"]) if inputs.affine["flip_y"] else 0.0]
    )
    composite_matrix = matrix @ flip_matrix
    composite_translation = translation + matrix @ flip_offset
    rotation, stretch = polar(matrix)
    angle = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    singular = np.linalg.svd(matrix, compute_uv=False)
    wrong = invert_saved_affine(inputs.moving, inputs.affine)
    correct_nn = cKDTree(inputs.fixed).query(inputs.moving, k=1)[0]
    wrong_nn = cKDTree(inputs.fixed).query(wrong, k=1)[0]
    reestimated = estimate_affine_with_y_flip(
        inputs.original_moving,
        inputs.original_fixed,
        image_height_px=float(inputs.affine["image_height"]),
        image_width_px=float(inputs.affine["image_width"]),
        similarity_trim_quantile=0.8,
        affine_trim_quantile=0.7,
        max_iter=80,
    )
    return {
        "source_run": str(RUN),
        "transform_direction": "normalized HE pixel xy -> fluorescence GeoJSON world xy (um)",
        "application_order": "optional x/y image-axis flip, then row-vector points @ affine_matrix.T + translation",
        "estimation_inputs": {
            "moving": "1,842 normalized HE nucleus centers in image-pixel xy",
            "fixed": "1,979 fluorescence GeoJSON nucleus centers in application world-um xy",
            "image_values_used_in_objective": False,
            "image_dimensions_used_for_flip": [float(inputs.affine["image_width"]), float(inputs.affine["image_height"])],
        },
        "estimation_initialization": "Each of 4 x/y flip candidates; within each, rotations 0..330 degrees in 30-degree steps with centroid/RMS-radius scale initialization.",
        "similarity_stage": {
            "method": "nearest-neighbor trimmed similarity ICP followed by Umeyama refit",
            "trim_quantile": 0.8,
            "max_iterations": 80,
            "convergence_tolerance": 1e-4,
        },
        "affine_stage": {
            "method": "nearest-neighbor trimmed ICP with full 2x2 affine least-squares refit",
            "trim_quantile": 0.7,
            "max_iterations": 80,
            "convergence_tolerance": 1e-4,
            "reestimated_pairs_used": int(reestimated.n_pairs),
            "reestimated_mean_trimmed_residual_um": float(reestimated.mean_residual),
            "reestimated_final_all_point_median_nn_residual_um": float(reestimated.median_residual),
        },
        "flip_candidate_selection": "Similarity-stage orientation error + target-span penalty when fitted affine determinant < 0 + 0.001 per explicit flip; final affine median residual is only a tie-breaker.",
        "reestimation_exactly_matches_saved_matrix": bool(np.array_equal(reestimated.affine_matrix, matrix)),
        "reestimation_exactly_matches_saved_translation": bool(np.array_equal(reestimated.translation, translation)),
        "reestimated_flip_matches_saved": bool(
            reestimated.flip_x == inputs.affine["flip_x"] and reestimated.flip_y == inputs.affine["flip_y"]
        ),
        "affine_matrix_after_explicit_flip": matrix,
        "affine_translation_after_explicit_flip": translation,
        "flip_x": bool(inputs.affine["flip_x"]),
        "flip_y": bool(inputs.affine["flip_y"]),
        "image_width_px": float(inputs.affine["image_width"]),
        "image_height_px": float(inputs.affine["image_height"]),
        "effective_raw_pixel_to_world_matrix": composite_matrix,
        "effective_raw_pixel_to_world_translation": composite_translation,
        "determinant_after_explicit_flip": float(np.linalg.det(matrix)),
        "effective_determinant_including_flip": float(np.linalg.det(composite_matrix)),
        "singular_scales_um_per_input_pixel": singular,
        "effective_x_axis_scale_um_per_pixel": float(np.linalg.norm(composite_matrix[:, 0])),
        "effective_y_axis_scale_um_per_pixel": float(np.linalg.norm(composite_matrix[:, 1])),
        "polar_rotation_degrees_after_flip": angle,
        "symmetric_stretch_matrix": stretch,
        "shear_indicator_off_diagonal": float(stretch[0, 1]),
        "saved_affine_reconstruction_max_abs_error_um": float(np.max(np.abs(rebuilt - affine_all))),
        "inverse_roundtrip_max_abs_error_px": float(np.max(np.abs(roundtrip - inputs.original_moving))),
        "correct_direction_nn_median_um_proxy": float(np.median(correct_nn)),
        "inverse_direction_nn_median_mixed_units_proxy": float(np.median(wrong_nn)),
        "independent_landmarks_available": False,
        "tre_um": None,
        "raw_he_pixel_spacing_metadata": "UNKNOWN",
        "raw_he_crop_offset_metadata": "UNKNOWN",
        "raw_he_downsampling_metadata": "UNKNOWN",
        "geojson_crs_and_unit_metadata": "UNKNOWN (world-um is an application convention, not retained source CRS metadata)",
    }


def world_to_pixels(points: np.ndarray, metadata: dict) -> np.ndarray:
    pixel = float(metadata["output_pixel_size_um"])
    if metadata["output_origin"] == "upper-right":
        cols = (float(metadata["col0_world_x"]) - points[:, 0]) / pixel
    else:
        cols = (points[:, 0] - float(metadata["col0_world_x"])) / pixel
    if metadata["output_origin"] in {"upper-left", "upper-right"}:
        rows = (points[:, 1] - float(metadata["row0_world_y"])) / pixel
    else:
        rows = (float(metadata["row0_world_y"]) - points[:, 1]) / pixel
    return np.column_stack([cols, rows])


def tissue_features(points: np.ndarray, inputs: Inputs) -> pd.DataFrame:
    pixels = world_to_pixels(np.asarray(points), inputs.metadata)
    height, width = inputs.tissue_mask.shape
    cols, rows = pixels[:, 0], pixels[:, 1]
    inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    row_i = np.clip(np.rint(rows).astype(int), 0, height - 1)
    col_i = np.clip(np.rint(cols).astype(int), 0, width - 1)
    tissue = np.zeros(len(points), dtype=bool)
    tissue[inside] = inputs.tissue_mask[row_i[inside], col_i[inside]]
    inside_distance = distance_transform_edt(inputs.tissue_mask)
    outside_distance = distance_transform_edt(~inputs.tissue_mask)
    dist_inside = np.zeros(len(points), dtype=float)
    dist_outside = np.full(len(points), np.inf)
    dist_inside[inside] = inside_distance[row_i[inside], col_i[inside]]
    dist_outside[inside] = outside_distance[row_i[inside], col_i[inside]]
    image_edge = np.minimum.reduce([cols, width - 1 - cols, rows, height - 1 - rows])
    return pd.DataFrame({
        "col": cols,
        "row": rows,
        "inside_image": inside,
        "inside_tissue": tissue,
        "distance_inside_tissue_px": dist_inside,
        "distance_to_tissue_px": dist_outside,
        "distance_to_image_edge_px": image_edge,
    })


def reconstructed_subset_masks(inputs: Inputs, edge_margin_px: float = 10.0, near_um: float = 43.0):
    affine_all = np.load(INPUT / "moving_points.npy", allow_pickle=False)
    fixed_features = tissue_features(inputs.original_fixed, inputs)
    moving_features = tissue_features(affine_all, inputs)
    near = cKDTree(affine_all).query(inputs.original_fixed, k=1)[0] <= near_um
    fixed_valid = (
        fixed_features["inside_image"].to_numpy(bool)
        & fixed_features["inside_tissue"].to_numpy(bool)
        & (fixed_features["distance_to_image_edge_px"].to_numpy(float) >= edge_margin_px)
        & (fixed_features["distance_inside_tissue_px"].to_numpy(float) >= edge_margin_px)
        & near
    )
    moving_valid = (
        moving_features["inside_image"].to_numpy(bool)
        & moving_features["inside_tissue"].to_numpy(bool)
        & (moving_features["distance_to_image_edge_px"].to_numpy(float) >= edge_margin_px)
        & (moving_features["distance_inside_tissue_px"].to_numpy(float) >= edge_margin_px)
    )
    expected_fixed = np.zeros(len(inputs.original_fixed), dtype=bool)
    expected_fixed[inputs.fixed_original_index] = True
    expected_moving = np.zeros(len(affine_all), dtype=bool)
    expected_moving[inputs.moving_original_index] = True
    return fixed_valid, moving_valid, expected_fixed, expected_moving, fixed_features, moving_features


def convex_intersection(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman intersection of two CCW convex polygons."""
    output = [np.asarray(point, dtype=float) for point in subject]
    for edge_index in range(len(clip)):
        a = clip[edge_index]
        b = clip[(edge_index + 1) % len(clip)]
        source = output
        output = []
        if not source:
            break

        def inside(point):
            return np.cross(b - a, point - a) >= -1e-12

        def intersection(first, second):
            direction = second - first
            edge = b - a
            denominator = np.cross(direction, edge)
            if abs(denominator) < 1e-15:
                return second
            t = np.cross(a - first, edge) / denominator
            return first + t * direction

        previous = source[-1]
        for current in source:
            if inside(current):
                if not inside(previous):
                    output.append(intersection(previous, current))
                output.append(current)
            elif inside(previous):
                output.append(intersection(previous, current))
            previous = current
    return np.asarray(output)


def polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    return float(0.5 * abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) - np.dot(points[:, 1], np.roll(points[:, 0], 1))))


def roi_metrics(inputs: Inputs) -> dict:
    fv, mv, ef, em, ff, mf = reconstructed_subset_masks(inputs)
    fixed_hull = inputs.fixed[ConvexHull(inputs.fixed).vertices]
    moving_hull = inputs.moving[ConvexHull(inputs.moving).vertices]
    intersection = convex_intersection(fixed_hull, moving_hull)
    fixed_area = polygon_area(fixed_hull)
    moving_area = polygon_area(moving_hull)
    overlap_area = polygon_area(intersection)
    tissue_area = float(np.count_nonzero(inputs.tissue_mask)) * float(inputs.metadata["output_pixel_size_um"]) ** 2
    eroded_common = distance_transform_edt(inputs.tissue_mask) >= 10.0
    eroded_area = float(np.count_nonzero(eroded_common)) * float(inputs.metadata["output_pixel_size_um"]) ** 2
    return {
        "original_counts": {"fixed": len(inputs.original_fixed), "moving": len(inputs.original_moving)},
        "subset_counts": {"fixed": len(inputs.fixed), "moving": len(inputs.moving)},
        "excluded_counts": {"fixed": int(len(inputs.original_fixed) - len(inputs.fixed)), "moving": int(len(inputs.original_moving) - len(inputs.moving))},
        "reconstructed_filter_assumptions": {"edge_margin_px": 10.0, "fixed_nearest_he_max_um": 43.0},
        "saved_fixed_subset_exactly_reproduced": bool(np.array_equal(fv, ef)),
        "saved_moving_subset_exactly_reproduced": bool(np.array_equal(mv, em)),
        "warped_he_valid_tissue_area_um2": tissue_area,
        "eroded_10um_tissue_area_um2": eroded_area,
        "fixed_subset_convex_hull_area_um2": fixed_area,
        "moving_subset_convex_hull_area_um2": moving_area,
        "convex_hull_intersection_area_um2_proxy": overlap_area,
        "convex_hull_iou_proxy": overlap_area / max(fixed_area + moving_area - overlap_area, 1e-12),
        "fixed_hull_covered_by_intersection_proxy": overlap_area / max(fixed_area, 1e-12),
        "moving_hull_covered_by_intersection_proxy": overlap_area / max(moving_area, 1e-12),
        "authoritative_fluorescence_acquisition_polygon_available": False,
        "roi_interpretation": "HE-derived tissue mask is authoritative for saved subset filtering; hull metrics are geometric proxies, not acquisition-FOV proof.",
        "fixed_feature_counts": {
            "outside_image": int((~ff["inside_image"]).sum()),
            "outside_tissue_or_background": int((ff["inside_image"] & ~ff["inside_tissue"]).sum()),
            "within_10px_tissue_boundary": int((ff["inside_tissue"] & (ff["distance_inside_tissue_px"] < 10)).sum()),
        },
        "moving_feature_counts": {
            "outside_image": int((~mf["inside_image"]).sum()),
            "outside_tissue_or_background": int((mf["inside_image"] & ~mf["inside_tissue"]).sum()),
            "within_10px_tissue_boundary": int((mf["inside_tissue"] & (mf["distance_inside_tissue_px"] < 10)).sum()),
        },
    }


def regional_match_vectorized(fixed: np.ndarray, moving: np.ndarray, radius_um: float) -> RegionalMatch:
    """Exact vectorization of frozen ``match_regional_patterns`` for one radius."""
    fixed = np.asarray(fixed, dtype=float)
    moving = np.asarray(moving, dtype=float)
    pixel = float(REGIONAL["pixel_size_um"])
    patch_radius = float(REGIONAL["patch_radius_um"])
    padding = patch_radius + float(radius_um) + 4.0
    bounds = _bounds(fixed, moving, padding)
    fixed_channels = _channels(fixed, bounds)
    moving_channels = _channels(moving, bounds)
    anchors = _anchors(moving)
    step = float(REGIONAL["search_step_um"])
    offsets_1d = np.arange(-radius_um, radius_um + 0.5 * step, step)
    offsets = np.asarray([(x, y) for y in offsets_1d for x in offsets_1d], dtype=float)
    axis = np.arange(-patch_radius, patch_radius + 0.5 * pixel, pixel)
    xx, yy = np.meshgrid(axis, axis)
    n_channels = len(fixed_channels)

    best_offsets, best_scores, second_scores = [], [], []
    margins, ratios, all_top_offsets, all_top_scores = [], [], [], []
    for anchor in anchors:
        moving_patch = _sample_patch(moving_channels, anchor, bounds)
        left_centered = moving_patch - np.mean(moving_patch, axis=(1, 2), keepdims=True)
        left_norm = np.linalg.norm(left_centered.reshape(n_channels, -1), axis=1)
        cols = (anchor[0] + offsets[:, 0, None, None] + xx[None] - bounds[0]) / pixel
        rows = (anchor[1] + offsets[:, 1, None, None] + yy[None] - bounds[1]) / pixel
        channel_scores = []
        for channel_index, channel in enumerate(fixed_channels):
            patches = map_coordinates(channel, [rows, cols], order=1, mode="constant", cval=0.0)
            centered = patches - np.mean(patches, axis=(1, 2), keepdims=True)
            denominator = left_norm[channel_index] * np.linalg.norm(centered.reshape(len(offsets), -1), axis=1)
            numerator = np.sum(centered * left_centered[channel_index], axis=(1, 2))
            channel_scores.append(
                np.divide(numerator, denominator, out=np.full(len(offsets), -1.0), where=denominator > 1e-12)
            )
        scores = np.mean(channel_scores, axis=0)
        order = np.argsort(scores)[::-1]
        best_index = int(order[0])
        best_offset = offsets[best_index]
        separated = [
            int(index) for index in order[1:]
            if np.max(np.abs(offsets[index] - best_offset)) > 1.01 * step
        ]
        second_index = separated[0] if separated else int(order[1])
        top_indices = [best_index, *separated[:4]]
        while len(top_indices) < 5:
            top_indices.append(int(order[len(top_indices)]))
        best, second = float(scores[best_index]), float(scores[second_index])
        best_offsets.append(best_offset)
        best_scores.append(best)
        second_scores.append(second)
        margins.append(best - second)
        ratios.append((best + 1.001) / max(second + 1.001, 1e-6))
        all_top_offsets.append(offsets[top_indices])
        all_top_scores.append(scores[top_indices])

    best_offsets_array = np.asarray(best_offsets)
    best_scores_array = np.asarray(best_scores)
    second_scores_array = np.asarray(second_scores)
    margins_array = np.asarray(margins)
    cell_anchor = cKDTree(anchors).query(moving, k=1)[1].astype(int)
    confidence = np.clip(0.5 * (best_scores_array + 1.0) + 2.0 * np.maximum(margins_array, 0.0), 0.0, 1.0)
    return RegionalMatch(
        anchor_centers=anchors,
        best_offsets=best_offsets_array,
        best_ncc=best_scores_array,
        second_ncc=second_scores_array,
        margin=margins_array,
        peak_ratio=np.asarray(ratios),
        top_offsets=np.asarray(all_top_offsets),
        top_scores=np.asarray(all_top_scores),
        cell_anchor_index=cell_anchor,
        cell_offsets=best_offsets_array[cell_anchor],
        cell_confidence=confidence[cell_anchor],
        bounds=bounds,
    )


def selected_candidate_set(rankings, fingerprint_cost, regional, indices: np.ndarray) -> CandidateSet:
    indices = np.asarray(indices, dtype=int)
    position_distance = np.take_along_axis(rankings.spatial_distance_um, indices, axis=1)
    region_residual = np.take_along_axis(rankings.region_residual_um, indices, axis=1)
    position_cost = position_distance**2 / (2.0 * float(CANDIDATES["position_scale_um"]) ** 2)
    selected_fp = np.take_along_axis(fingerprint_cost, indices, axis=1)
    region_cost = region_residual**2 / (2.0 * float(CANDIDATES["region_residual_scale_um"]) ** 2)
    region_cost += 0.25 * (1.0 - regional.cell_confidence[:, None])
    weights = CANDIDATES["weights_full"]
    base = weights["position"] * position_cost + weights["fingerprint"] * selected_fp + weights["region"] * region_cost
    return CandidateSet(
        indices=indices,
        position_cost=position_cost,
        fingerprint_cost=selected_fp,
        region_cost=region_cost,
        base_cost=base,
        spatial_distance_um=position_distance,
        region_residual_um=region_residual,
        fingerprint_similarity=np.exp(-0.5 * np.clip(selected_fp, 0.0, 20.0)),
        region_similarity=np.exp(-0.5 * np.clip(region_cost, 0.0, 20.0)),
    )


def graph_support_scores_fast(fixed: np.ndarray, moving: np.ndarray, candidates: CandidateSet) -> np.ndarray:
    """Algebraically equivalent sparse lookup version of frozen graph support."""
    moving_graph = mutual_knn_graph(moving)
    fixed_graph = mutual_knn_graph(fixed)
    row_slots = [{int(value): slot for slot, value in enumerate(row)} for row in candidates.indices]
    output = np.zeros(candidates.indices.shape, dtype=float)
    length_tolerance = float(GRAPH["length_tolerance_um"])
    angle_tolerance = float(GRAPH["angle_tolerance_radians"])
    for moving_index, moving_neighbors in enumerate(moving_graph):
        if not len(moving_neighbors):
            continue
        for slot, fixed_index in enumerate(candidates.indices[moving_index]):
            fixed_neighbors = fixed_graph[int(fixed_index)]
            if not len(fixed_neighbors):
                continue
            neighbor_scores = []
            for moving_neighbor in moving_neighbors:
                moving_vector = moving[int(moving_neighbor)] - moving[moving_index]
                moving_length = max(float(np.linalg.norm(moving_vector)), 1e-6)
                best = 0.0
                lookup = row_slots[int(moving_neighbor)]
                for fixed_neighbor in fixed_neighbors:
                    neighbor_slot = lookup.get(int(fixed_neighbor))
                    if neighbor_slot is None:
                        continue
                    fixed_vector = fixed[int(fixed_neighbor)] - fixed[int(fixed_index)]
                    fixed_length = max(float(np.linalg.norm(fixed_vector)), 1e-6)
                    length_delta = abs(fixed_length - moving_length)
                    cosine = np.clip(float(np.dot(moving_vector, fixed_vector) / (moving_length * fixed_length)), -1.0, 1.0)
                    angle_delta = float(np.arccos(cosine))
                    geometry = np.exp(-0.5 * (length_delta / length_tolerance) ** 2)
                    geometry *= np.exp(-0.5 * (angle_delta / angle_tolerance) ** 2)
                    compatibility = np.exp(-0.35 * min(float(candidates.base_cost[int(moving_neighbor), neighbor_slot]), 10.0))
                    best = max(best, float(geometry * compatibility))
                neighbor_scores.append(best)
            values = np.asarray(neighbor_scores, dtype=float)
            if values.size:
                strong_fraction = float(np.mean(values >= 0.30))
                robust_mean = float(np.mean(np.sort(values)[-max(2, int(np.ceil(0.6 * len(values)))):]))
                output[moving_index, slot] = 0.5 * strong_fraction + 0.5 * robust_mean
    return np.clip(output, 0.0, 1.0)


def build_matching_state(fixed: np.ndarray, moving: np.ndarray, regional: RegionalMatch):
    fingerprints = compute_fingerprint_pair(fixed, moving)
    fingerprint_cost = pairwise_fingerprint_cost(fingerprints)
    rankings = build_candidate_rankings(fixed, moving, fingerprint_cost, regional)
    candidates = build_candidate_set(rankings, fingerprint_cost, regional, include_region=True)
    graph = graph_support_scores_fast(fixed, moving, candidates)
    return fingerprint_cost, rankings, candidates, graph


def local_density(points: np.ndarray, query: np.ndarray, radius: float = 34.0) -> np.ndarray:
    tree = cKDTree(points)
    counts = np.asarray([len(values) for values in tree.query_ball_point(query, radius)], dtype=float)
    return counts / (np.pi * radius**2)


def qsummary(values: np.ndarray, prefix: str) -> dict:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {f"{prefix}_p05": np.nan, f"{prefix}_median": np.nan, f"{prefix}_p95": np.nan}
    return {
        f"{prefix}_p05": float(np.percentile(finite, 5)),
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_p95": float(np.percentile(finite, 95)),
    }


def boundary_flags(offsets: np.ndarray, radius: float) -> dict[str, np.ndarray]:
    tol = 1e-9
    xp = offsets[:, 0] >= radius - tol
    xn = offsets[:, 0] <= -radius + tol
    yp = offsets[:, 1] >= radius - tol
    yn = offsets[:, 1] <= -radius + tol
    x = xp | xn
    y = yp | yn
    return {"x_positive": xp, "x_negative": xn, "y_positive": yp, "y_negative": yn, "x_any": x, "y_any": y, "xy_both": x & y, "any": x | y}


def regional_coherence(regional: RegionalMatch) -> float:
    if len(regional.anchor_centers) < 2:
        return np.nan
    neighbors = cKDTree(regional.anchor_centers).query(regional.anchor_centers, k=min(5, len(regional.anchor_centers)))[1][:, 1:]
    differences = regional.best_offsets[:, None, :] - regional.best_offsets[neighbors]
    return float(np.median(np.linalg.norm(differences, axis=2)))


def source_union_rows(rankings) -> list[list[int]]:
    result = []
    source_k = int(CANDIDATES["source_k"])
    for row in range(len(rankings.spatial)):
        result.append(list(dict.fromkeys(np.concatenate([
            rankings.spatial[row, :source_k], rankings.regional[row, :source_k], rankings.fingerprint[row, :source_k]
        ]).tolist())))
    return result


def candidate_source_table(inputs: Inputs, fingerprint_cost, rankings, candidates, graph) -> pd.DataFrame:
    unions = source_union_rows(rankings)
    source_k = int(CANDIDATES["source_k"])
    graph_cost = candidates.base_cost + float(CANDIDATES["graph_weight"]) * (1.0 - graph)
    graph_top = candidates.indices[np.arange(len(inputs.moving)), np.argmin(graph_cost, axis=1)]
    tissue = tissue_features(inputs.moving, inputs)
    rows = []
    for i in range(len(inputs.moving)):
        spatial = rankings.spatial[i, :30]
        regional = rankings.regional[i, :30]
        fingerprint = rankings.fingerprint[i, :30]
        combined = rankings.combined[i, :30]
        final = candidates.indices[i]
        final_set = set(map(int, final))
        union = unions[i]
        union_set = set(map(int, union))
        spatial20, regional20, fingerprint20 = map(
            lambda values: set(map(int, values)),
            (spatial[:source_k], regional[:source_k], fingerprint[:source_k]),
        )
        candidate_points = inputs.fixed[final]
        rows.append({
            "moving_subset_index": i,
            "moving_original_index": int(inputs.moving_original_index[i]),
            "moving_stable_id": f"moving_original_{inputs.moving_original_index[i]:06d}",
            "moving_x_um": inputs.moving[i, 0],
            "moving_y_um": inputs.moving[i, 1],
            "spatial_candidate_ids": ";".join(f"fixed_original_{inputs.fixed_original_index[j]:06d}" for j in spatial),
            "regional_candidate_ids": ";".join(f"fixed_original_{inputs.fixed_original_index[j]:06d}" for j in regional),
            "fingerprint_candidate_ids": ";".join(f"fixed_original_{inputs.fixed_original_index[j]:06d}" for j in fingerprint),
            "combined_candidate_ids": ";".join(f"fixed_original_{inputs.fixed_original_index[j]:06d}" for j in combined),
            "final_candidate_ids": ";".join(f"fixed_original_{inputs.fixed_original_index[j]:06d}" for j in final),
            "pruned_union_candidate_ids": ";".join(f"fixed_original_{inputs.fixed_original_index[j]:06d}" for j in union if j not in final_set),
            "union_count_before_prune": len(union),
            "pruned_count": len(union_set - final_set),
            "retained_spatial_source20": len(final_set & spatial20),
            "retained_regional_source20": len(final_set & regional20),
            "retained_fingerprint_source20": len(final_set & fingerprint20),
            "spatial_regional_jaccard_top30": len(set(spatial) & set(regional)) / len(set(spatial) | set(regional)),
            "spatial_fingerprint_jaccard_top30": len(set(spatial) & set(fingerprint)) / len(set(spatial) | set(fingerprint)),
            "regional_fingerprint_jaccard_top30": len(set(regional) & set(fingerprint)) / len(set(regional) | set(fingerprint)),
            "spatial_top1_fixed_original_index": int(inputs.fixed_original_index[spatial[0]]),
            "regional_top1_fixed_original_index": int(inputs.fixed_original_index[regional[0]]),
            "fingerprint_top1_fixed_original_index": int(inputs.fixed_original_index[fingerprint[0]]),
            "combined_top1_fixed_original_index": int(inputs.fixed_original_index[combined[0]]),
            "final_base_top1_fixed_original_index": int(inputs.fixed_original_index[final[np.argmin(candidates.base_cost[i])]]),
            "final_graph_top1_fixed_original_index": int(inputs.fixed_original_index[graph_top[i]]),
            "spatial_top1_differs_from_final_graph_top1": bool(spatial[0] != graph_top[i]),
            "nearest_fixed_distance_um": rankings.spatial_distance_um[i, rankings.spatial[i, 0]],
            "rank10_fixed_distance_um": rankings.spatial_distance_um[i, rankings.spatial[i, 9]],
            "rank30_fixed_distance_um": rankings.spatial_distance_um[i, rankings.spatial[i, 29]],
            "rank80_fixed_distance_um": float(np.partition(rankings.spatial_distance_um[i], 79)[79]),
            "final_candidate_max_distance_um": float(np.max(candidates.spatial_distance_um[i])),
            "final_candidate_spread_um": float(np.max(pdist(candidate_points))) if len(candidate_points) > 1 else 0.0,
            "distance_inside_tissue_px": tissue.iloc[i]["distance_inside_tissue_px"],
            "distance_to_image_edge_px": tissue.iloc[i]["distance_to_image_edge_px"],
        })
    return pd.DataFrame(rows)


def fixed_augmented_pool(rankings, fingerprint_cost, regional, max_k: int = 100):
    spatial_order = np.argsort(rankings.spatial_distance_um, axis=1)
    regional_order = np.argsort(rankings.region_residual_um, axis=1)
    # Fingerprint is intentionally restricted to its frozen spatial top-80 pool.
    spatial80 = spatial_order[:, : min(80, spatial_order.shape[1])]
    fp80 = np.take_along_axis(fingerprint_cost, spatial80, axis=1)
    fingerprint_order = np.take_along_axis(spatial80, np.argsort(fp80, axis=1), axis=1)
    rows = []
    for i in range(len(spatial_order)):
        pool = np.asarray(list(dict.fromkeys(np.concatenate([
            spatial_order[i, :100], regional_order[i, :100], fingerprint_order[i, :80]
        ]).tolist())), dtype=int)
        pos = rankings.spatial_distance_um[i, pool] ** 2 / (2 * float(CANDIDATES["position_scale_um"]) ** 2)
        reg = rankings.region_residual_um[i, pool] ** 2 / (2 * float(CANDIDATES["region_residual_scale_um"]) ** 2)
        reg += 0.25 * (1.0 - regional.cell_confidence[i])
        weights = CANDIDATES["weights_full"]
        cost = weights["position"] * pos + weights["fingerprint"] * fingerprint_cost[i, pool] + weights["region"] * reg
        rows.append(pool[np.argsort(cost)[:max_k]])
    return np.asarray(rows), spatial_order, regional_order, fingerprint_order


def b2_real_table() -> pd.DataFrame:
    table = pd.read_csv(ADOPTION / "real_exploratory_per_nucleus.csv")
    result = table[table["method"] == "Separated Matching / Adoption"].copy()
    result = result.sort_values("moving_index")
    if not np.array_equal(result["moving_index"].to_numpy(), np.arange(len(result))):
        raise AssertionError("B2 real table is not aligned to moving rows")
    return result


def finite_frame(frame: pd.DataFrame, excluded: tuple[str, ...] = ()) -> bool:
    numeric = frame.select_dtypes(include=[np.number]).drop(columns=list(excluded), errors="ignore")
    return bool(np.isfinite(numeric.to_numpy()).all())

