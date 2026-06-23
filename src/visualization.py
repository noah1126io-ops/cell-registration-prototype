from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib import pyplot as plt


def colorize_label_image(label_image: np.ndarray) -> np.ndarray:
    """Convert an integer label image to an RGB preview."""
    labels = np.asarray(label_image)
    if labels.ndim != 2:
        raise ValueError("Label image preview expects a 2D integer mask.")

    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    positive = labels > 0
    if not np.any(positive):
        return rgb

    normalized = (labels.astype(np.uint64) * 2654435761 % 256).astype(np.uint8)
    colors = (colormaps["tab20"](normalized / 255.0)[..., :3] * 255).astype(np.uint8)
    rgb[positive] = colors[positive]
    return rgb


def _to_grayscale_preview(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3:
        image = image[..., :3].mean(axis=2)

    image = image.astype(np.float32)
    min_value = float(np.min(image))
    max_value = float(np.max(image))
    if max_value == min_value:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_value) / (max_value - min_value)


def visualize_cell_matches(
    fixed_image,
    fixed_features: pd.DataFrame,
    moving_features: pd.DataFrame,
    matches: pd.DataFrame,
    max_pairs: int,
):
    """Visualize fixed/transformed moving centroids and matched pair links."""
    if fixed_features is None or moving_features is None or matches is None:
        raise ValueError("fixed_features, moving_features, and matches are required.")

    max_pairs = max(0, int(max_pairs))
    display_matches = matches.head(max_pairs)

    fig, ax = plt.subplots(figsize=(8, 8))
    has_background = fixed_image is not None
    if has_background:
        background = _to_grayscale_preview(fixed_image)
        ax.imshow(background, cmap="gray")

    fixed_points = display_matches[
        ["fixed_centroid_x", "fixed_centroid_y", "matched_status"]
    ].dropna(subset=["fixed_centroid_x", "fixed_centroid_y"])
    moving_points = display_matches[
        ["moving_centroid_x", "moving_centroid_y", "matched_status"]
    ].dropna(subset=["moving_centroid_x", "moving_centroid_y"])

    if not fixed_points.empty:
        ax.scatter(
            fixed_points["fixed_centroid_x"],
            fixed_points["fixed_centroid_y"],
            s=12,
            c="#00d1ff",
            marker="o",
            linewidths=0,
            alpha=0.8,
            label="fixed",
        )

    if not moving_points.empty:
        ax.scatter(
            moving_points["moving_centroid_x"],
            moving_points["moving_centroid_y"],
            s=12,
            c="#ffb000",
            marker="x",
            linewidths=0.8,
            alpha=0.8,
            label="transformed moving",
        )

    matched = display_matches[display_matches["matched_status"] == "matched"]
    low_confidence = display_matches[display_matches["matched_status"] == "low_confidence"]

    for _, row in matched.iterrows():
        ax.plot(
            [row["fixed_centroid_x"], row["moving_centroid_x"]],
            [row["fixed_centroid_y"], row["moving_centroid_y"]],
            color="#54e346",
            linewidth=0.8,
            alpha=0.75,
        )

    if not low_confidence.empty:
        ax.scatter(
            low_confidence["fixed_centroid_x"],
            low_confidence["fixed_centroid_y"],
            s=24,
            facecolors="none",
            edgecolors="#ff4d6d",
            marker="o",
            linewidths=1.0,
            alpha=0.9,
            label="low confidence fixed",
        )
        ax.scatter(
            low_confidence["moving_centroid_x"],
            low_confidence["moving_centroid_y"],
            s=24,
            facecolors="none",
            edgecolors="#ff4d6d",
            marker="s",
            linewidths=1.0,
            alpha=0.9,
            label="low confidence moving",
        )

    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")
    if not has_background:
        ax.invert_yaxis()
    ax.set_title(f"Cell match overlay ({len(display_matches)} displayed rows)")
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    return fig


def visualize_point_sets(
    fixed_features: pd.DataFrame,
    moving_features: pd.DataFrame,
    *,
    title: str,
    background_image=None,
    invert_x_axis: bool = False,
    invert_y_axis: bool | None = None,
):
    """Visualize fixed and moving point sets on an optional image background."""
    if fixed_features is None or moving_features is None:
        raise ValueError("fixed_features and moving_features are required.")

    fig, ax = plt.subplots(figsize=(8, 8))
    has_background = background_image is not None
    if has_background:
        ax.imshow(_to_grayscale_preview(background_image), cmap="gray")

    if not fixed_features.empty:
        ax.scatter(
            fixed_features["centroid_x"],
            fixed_features["centroid_y"],
            s=14,
            c="#00d1ff",
            marker="o",
            linewidths=0,
            alpha=0.85,
            label="fixed",
        )

    if not moving_features.empty:
        ax.scatter(
            moving_features["centroid_x"],
            moving_features["centroid_y"],
            s=14,
            c="#ffb000",
            marker="x",
            linewidths=0.8,
            alpha=0.85,
            label="moving",
        )

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    if invert_x_axis:
        ax.invert_xaxis()
    if invert_y_axis is None:
        invert_y_axis = not has_background
    if invert_y_axis:
        ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    return fig


def visualize_warped_he_point_overlay(
    warped_he_image,
    geojson_pixels: np.ndarray,
    he_pixels: np.ndarray,
    *,
    title: str,
    max_points: int = 3000,
):
    """Overlay GeoJSON and transformed HE points on a warped HE image."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(_to_grayscale_preview(warped_he_image), cmap="gray")

    max_points = max(1, int(max_points))
    if len(geojson_pixels):
        geojson_pixels = np.asarray(geojson_pixels, dtype=float)[:max_points]
        ax.scatter(
            geojson_pixels[:, 0],
            geojson_pixels[:, 1],
            s=12,
            c="#00d1ff",
            marker="o",
            linewidths=0,
            alpha=0.65,
            label="GeoJSON nuclei",
        )

    if len(he_pixels):
        he_pixels = np.asarray(he_pixels, dtype=float)[:max_points]
        ax.scatter(
            he_pixels[:, 0],
            he_pixels[:, 1],
            s=14,
            c="#ffb000",
            marker="x",
            linewidths=0.8,
            alpha=0.8,
            label="warped HE nuclei",
        )

    ax.set_title(title)
    ax.set_xlim(0, warped_he_image.shape[1])
    ax.set_ylim(warped_he_image.shape[0], 0)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    return fig


def visualize_translation_anchors(anchors: pd.DataFrame, *, title: str):
    """Show accepted and rejected local-translation anchors."""
    fig, ax = plt.subplots(figsize=(8, 8))
    accepted = anchors[anchors["accepted"]]
    rejected = anchors[~anchors["accepted"]]

    if not rejected.empty:
        ax.scatter(rejected["anchor_x"], rejected["anchor_y"], s=12, c="#999999", alpha=0.45, label="rejected")
        ax.quiver(
            rejected["anchor_x"],
            rejected["anchor_y"],
            rejected["dx"],
            rejected["dy"],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="#999999",
            alpha=0.35,
            width=0.002,
        )
    if not accepted.empty:
        ax.scatter(accepted["anchor_x"], accepted["anchor_y"], s=18, c="#00d1ff", alpha=0.8, label="accepted")
        ax.quiver(
            accepted["anchor_x"],
            accepted["anchor_y"],
            accepted["dx"],
            accepted["dy"],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="#ffb000",
            alpha=0.9,
            width=0.003,
        )

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    return fig


def visualize_displacement_field(grid_x, grid_y, displacement_x, displacement_y, *, title: str, stride: int = 2):
    """Show a sampled displacement vector field."""
    fig, ax = plt.subplots(figsize=(8, 8))
    stride = max(1, int(stride))
    magnitude = np.sqrt(displacement_x**2 + displacement_y**2)
    image = ax.imshow(
        magnitude,
        cmap="magma",
        extent=[float(np.min(grid_x)), float(np.max(grid_x)), float(np.max(grid_y)), float(np.min(grid_y))],
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="displacement")
    ax.quiver(
        grid_x[::stride, ::stride],
        grid_y[::stride, ::stride],
        displacement_x[::stride, ::stride],
        displacement_y[::stride, ::stride],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="white",
        alpha=0.8,
        width=0.003,
    )
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def visualize_distance_histogram(before_distances, after_distances, *, title: str):
    """Plot before/after nearest-neighbor distance histograms."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(before_distances, bins=40, alpha=0.55, label="before", color="#999999")
    ax.hist(after_distances, bins=40, alpha=0.55, label="after", color="#00d1ff")
    ax.set_title(title)
    ax.set_xlabel("nearest distance")
    ax.set_ylabel("count")
    ax.legend(frameon=True)
    fig.tight_layout()
    return fig


def visualize_anchor_correlation_heatmap(anchors: pd.DataFrame, *, title: str):
    """Scatter heatmap of local translation anchor correlations."""
    fig, ax = plt.subplots(figsize=(8, 8))
    values = anchors["correlation"].to_numpy(dtype=float)
    scatter = ax.scatter(
        anchors["anchor_x"],
        anchors["anchor_y"],
        c=values,
        s=22,
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="correlation")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# TODO: Add checkerboards and registration quality summary plots.
