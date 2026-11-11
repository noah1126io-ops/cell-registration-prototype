"""Claude-like local regional matching using Gaussian/DoG density NCC."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree

from config import REGIONAL


@dataclass(frozen=True)
class RegionalMatch:
    anchor_centers: np.ndarray
    best_offsets: np.ndarray
    best_ncc: np.ndarray
    second_ncc: np.ndarray
    margin: np.ndarray
    peak_ratio: np.ndarray
    top_offsets: np.ndarray
    top_scores: np.ndarray
    cell_anchor_index: np.ndarray
    cell_offsets: np.ndarray
    cell_confidence: np.ndarray
    bounds: tuple[float, float, float, float]


def _bounds(fixed: np.ndarray, moving: np.ndarray, padding: float) -> tuple[float, float, float, float]:
    pooled = np.vstack([fixed, moving])
    return (
        float(np.min(pooled[:, 0]) - padding),
        float(np.min(pooled[:, 1]) - padding),
        float(np.max(pooled[:, 0]) + padding),
        float(np.max(pooled[:, 1]) + padding),
    )


def _impulses(points: np.ndarray, bounds, pixel_size: float) -> np.ndarray:
    min_x, min_y, max_x, max_y = bounds
    width = int(np.ceil((max_x - min_x) / pixel_size)) + 1
    height = int(np.ceil((max_y - min_y) / pixel_size)) + 1
    cols = np.rint((points[:, 0] - min_x) / pixel_size).astype(int)
    rows = np.rint((points[:, 1] - min_y) / pixel_size).astype(int)
    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    output = np.zeros((height, width), dtype=float)
    np.add.at(output, (rows[valid], cols[valid]), 1.0)
    return output


def _channels(points: np.ndarray, bounds) -> np.ndarray:
    pixel = float(REGIONAL["pixel_size_um"])
    impulses = _impulses(points, bounds, pixel)
    gaussian = [
        gaussian_filter(impulses, float(sigma) / pixel, mode="constant")
        for sigma in REGIONAL["gaussian_sigmas_um"]
    ]
    dog_low, dog_high = REGIONAL["dog_sigmas_um"]
    dog = gaussian_filter(impulses, dog_low / pixel, mode="constant") - gaussian_filter(
        impulses, dog_high / pixel, mode="constant"
    )
    return np.stack([*gaussian, dog], axis=0)


def _sample_patch(channels: np.ndarray, center: np.ndarray, bounds) -> np.ndarray:
    pixel = float(REGIONAL["pixel_size_um"])
    radius = float(REGIONAL["patch_radius_um"])
    axis = np.arange(-radius, radius + 0.5 * pixel, pixel)
    xx, yy = np.meshgrid(axis, axis)
    cols = (center[0] + xx - bounds[0]) / pixel
    rows = (center[1] + yy - bounds[1]) / pixel
    patches = [
        map_coordinates(channel, [rows, cols], order=1, mode="constant", cval=0.0)
        for channel in channels
    ]
    return np.stack(patches, axis=0)


def _ncc(first: np.ndarray, second: np.ndarray) -> float:
    scores = []
    for left, right in zip(first, second, strict=True):
        left_centered = left - float(np.mean(left))
        right_centered = right - float(np.mean(right))
        denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
        scores.append(float(np.sum(left_centered * right_centered) / denominator) if denominator > 1e-12 else -1.0)
    return float(np.mean(scores))


def _anchors(moving: np.ndarray) -> np.ndarray:
    spacing = float(REGIONAL["region_spacing_um"])
    origin = np.min(moving, axis=0)
    bins = np.floor((moving - origin) / spacing).astype(int)
    groups: dict[tuple[int, int], list[int]] = {}
    for index, key in enumerate(map(tuple, bins)):
        groups.setdefault(key, []).append(index)
    anchors = [np.mean(moving[indices], axis=0) for indices in groups.values() if len(indices) >= 3]
    if not anchors:
        anchors = [np.mean(moving, axis=0)]
    return np.asarray(anchors, dtype=float)


def match_regional_patterns(fixed: np.ndarray, moving: np.ndarray) -> RegionalMatch:
    fixed = np.asarray(fixed, dtype=float)
    moving = np.asarray(moving, dtype=float)
    padding = float(REGIONAL["patch_radius_um"] + REGIONAL["search_radius_um"] + 4.0)
    bounds = _bounds(fixed, moving, padding)
    fixed_channels = _channels(fixed, bounds)
    moving_channels = _channels(moving, bounds)
    anchors = _anchors(moving)
    radius = float(REGIONAL["search_radius_um"])
    step = float(REGIONAL["search_step_um"])
    offsets_1d = np.arange(-radius, radius + 0.5 * step, step)
    offsets = np.asarray([(x, y) for y in offsets_1d for x in offsets_1d], dtype=float)

    best_offsets = []
    best_scores = []
    second_scores = []
    margins = []
    ratios = []
    all_top_offsets = []
    all_top_scores = []
    for anchor in anchors:
        moving_patch = _sample_patch(moving_channels, anchor, bounds)
        scores = np.asarray(
            [_ncc(moving_patch, _sample_patch(fixed_channels, anchor + offset, bounds)) for offset in offsets]
        )
        order = np.argsort(scores)[::-1]
        best_index = int(order[0])
        best_offset = offsets[best_index]
        separated = [
            int(index)
            for index in order[1:]
            if np.max(np.abs(offsets[index] - best_offset)) > 1.01 * step
        ]
        second_index = separated[0] if separated else int(order[1])
        top_indices = [best_index, *separated[:4]]
        while len(top_indices) < 5:
            top_indices.append(int(order[len(top_indices)]))
        best = float(scores[best_index])
        second = float(scores[second_index])
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
    # Confidence stays interpretable: positive NCC margin and positive NCC both
    # contribute, but raw components are retained separately.
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

