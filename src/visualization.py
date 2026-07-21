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


def visualize_density_flow_point_comparison(
    fixed_points: np.ndarray,
    affine_points: np.ndarray,
    attempted_points: np.ndarray,
    applied_points: np.ndarray,
    *,
    title: str = "Density-flow point registration states",
    max_points: int = 5000,
    invert_x_axis: bool = False,
    invert_y_axis: bool = False,
):
    """Compare fixed, affine, attempted, and safety-gated HE point states."""
    point_sets = [
        (np.asarray(fixed_points, dtype=float), "#00d1ff", "o", "fixed GeoJSON", 16, 0.75),
        (np.asarray(affine_points, dtype=float), "#64748b", "+", "affine HE", 18, 0.55),
        (np.asarray(attempted_points, dtype=float), "#d946ef", "x", "attempted density-flow HE", 18, 0.7),
        (np.asarray(applied_points, dtype=float), "#ffb000", "o", "applied HE", 12, 0.65),
    ]
    fig, ax = plt.subplots(figsize=(9, 8))
    limit = max(1, int(max_points))
    for points, color, marker, label, size, alpha in point_sets:
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"{label} points must have shape (n, 2).")
        display_points = points[:limit]
        if len(display_points):
            ax.scatter(
                display_points[:, 0],
                display_points[:, 1],
                s=size,
                c=color,
                marker=marker,
                linewidths=0.8,
                alpha=alpha,
                label=label,
            )
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    if invert_x_axis:
        ax.invert_xaxis()
    if invert_y_axis:
        ax.invert_yaxis()
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    return fig


def visualize_warped_he_point_overlay(
    warped_he_image,
    geojson_pixels: np.ndarray,
    he_pixels: np.ndarray,
    *,
    title: str,
    max_points: int = 3000,
    geojson_classifications=None,
    show_excluded_geojson: bool = True,
    show_edge_candidate_geojson: bool = True,
    boundary_pin_pixels: np.ndarray | None = None,
):
    """Overlay GeoJSON and transformed HE points on a warped HE image."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(_to_grayscale_preview(warped_he_image), cmap="gray")

    max_points = max(1, int(max_points))
    if len(geojson_pixels):
        geojson_pixels = np.asarray(geojson_pixels, dtype=float)[:max_points]
        if geojson_classifications is None:
            ax.scatter(
                geojson_pixels[:, 0],
                geojson_pixels[:, 1],
                s=12,
                c="#00d1ff",
                marker="o",
                linewidths=0,
                alpha=0.65,
                label="valid GeoJSON",
            )
        else:
            classifications = np.asarray(geojson_classifications, dtype=object)[:max_points]
            styles = {
                "valid": ("#00d1ff", "valid GeoJSON", 14, 0.75),
                "edge_candidate": ("#8b5cf6", "edge candidate GeoJSON", 14, 0.75),
                "excluded": ("#d1d5db", "excluded GeoJSON", 10, 0.25),
            }
            for class_name, (color, label, size, alpha) in styles.items():
                if class_name == "excluded" and not show_excluded_geojson:
                    continue
                if class_name == "edge_candidate" and not show_edge_candidate_geojson:
                    continue
                mask = classifications == class_name
                if np.any(mask):
                    ax.scatter(
                        geojson_pixels[mask, 0],
                        geojson_pixels[mask, 1],
                        s=size,
                        c=color,
                        marker="o",
                        linewidths=0,
                        alpha=alpha,
                        label=label,
                    )

    if boundary_pin_pixels is not None and len(boundary_pin_pixels):
        boundary_pin_pixels = np.asarray(boundary_pin_pixels, dtype=float)[:max_points]
        ax.scatter(
            boundary_pin_pixels[:, 0],
            boundary_pin_pixels[:, 1],
            s=18,
            c="#ffffff",
            marker="+",
            linewidths=0.8,
            alpha=0.7,
            label="boundary pin anchors",
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


def visualize_warped_pixel_point_scatter(
    geojson_pixels: np.ndarray,
    he_pixels: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    title: str,
    max_points: int = 3000,
):
    """Show GeoJSON and registered HE points in warped-image pixel coordinates."""
    fig, ax = plt.subplots(figsize=(8, 8))
    max_points = max(1, int(max_points))
    geojson_pixels = np.asarray(geojson_pixels, dtype=float)[:max_points]
    he_pixels = np.asarray(he_pixels, dtype=float)[:max_points]

    if len(geojson_pixels):
        ax.scatter(
            geojson_pixels[:, 0],
            geojson_pixels[:, 1],
            s=12,
            c="#00d1ff",
            marker="o",
            linewidths=0,
            alpha=0.65,
            label="GeoJSON pixels",
        )
    if len(he_pixels):
        ax.scatter(
            he_pixels[:, 0],
            he_pixels[:, 1],
            s=14,
            c="#ffb000",
            marker="x",
            linewidths=0.8,
            alpha=0.8,
            label="registered HE pixels",
        )

    ax.set_title(title)
    ax.set_xlim(0, image_width)
    ax.set_ylim(image_height, 0)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    return fig


def visualize_geojson_classification_overlay(
    warped_he_image,
    geojson_pixels: np.ndarray,
    classifications,
    *,
    title: str,
    max_points: int = 5000,
):
    """Overlay GeoJSON classification states on the warped HE image."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(_to_grayscale_preview(warped_he_image), cmap="gray")
    geojson_pixels = np.asarray(geojson_pixels, dtype=float)[:max_points]
    classifications = np.asarray(classifications, dtype=object)[:max_points]
    styles = {
        "valid": ("#00d1ff", "valid", 16, 0.8),
        "edge_candidate": ("#ffe45e", "edge_candidate", 16, 0.85),
        "excluded": ("#ff4d6d", "excluded", 12, 0.45),
    }
    for class_name, (color, label, size, alpha) in styles.items():
        mask = classifications == class_name
        if np.any(mask):
            ax.scatter(
                geojson_pixels[mask, 0],
                geojson_pixels[mask, 1],
                s=size,
                c=color,
                marker="o",
                linewidths=0,
                alpha=alpha,
                label=label,
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
    if anchors is None or anchors.empty or not {"anchor_x", "anchor_y"}.issubset(anchors.columns):
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        return fig

    anchors = anchors.copy()
    if "accepted" not in anchors:
        anchors["accepted"] = True
    if "dx" not in anchors:
        anchors["dx"] = 0.0
    if "dy" not in anchors:
        anchors["dy"] = 0.0
    if "shift_magnitude" not in anchors:
        anchors["shift_magnitude"] = np.sqrt(
            anchors["dx"].to_numpy(dtype=float) ** 2 + anchors["dy"].to_numpy(dtype=float) ** 2
        )

    anchors["accepted"] = anchors["accepted"].fillna(False).astype(bool)
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


def visualize_jacobian_heatmap(grid_x, grid_y, jacobian, *, title: str):
    """Show expansion/compression/fold-over from a displacement-field Jacobian."""
    fig, ax = plt.subplots(figsize=(8, 8))
    jacobian = np.asarray(jacobian, dtype=float)
    image = ax.imshow(
        jacobian,
        cmap="coolwarm",
        vmin=0.0,
        vmax=max(2.0, float(np.nanpercentile(jacobian, 99)) if np.isfinite(jacobian).any() else 2.0),
        extent=[float(np.min(grid_x)), float(np.max(grid_x)), float(np.max(grid_y)), float(np.min(grid_y))],
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Jacobian")
    fold = jacobian <= 0
    if np.any(fold):
        ax.contour(grid_x, grid_y, fold.astype(float), levels=[0.5], colors="black", linewidths=1.0)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def visualize_warp_grid_overlay(
    warped_he_image,
    before_lines: list[np.ndarray],
    after_lines: list[np.ndarray],
    *,
    title: str,
):
    """Overlay a regular grid before and after fine warp in warped-image pixel coordinates."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(_to_grayscale_preview(warped_he_image), cmap="gray")
    for line in before_lines:
        line = np.asarray(line, dtype=float)
        ax.plot(line[:, 0], line[:, 1], color="#00d1ff", linewidth=0.6, alpha=0.55)
    for line in after_lines:
        line = np.asarray(line, dtype=float)
        ax.plot(line[:, 0], line[:, 1], color="#ffb000", linewidth=0.8, alpha=0.75)
    ax.set_title(title)
    ax.set_xlim(0, warped_he_image.shape[1])
    ax.set_ylim(warped_he_image.shape[0], 0)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


def visualize_local_residual_map(
    fixed_points: np.ndarray,
    moving_points: np.ndarray,
    *,
    title: str,
    max_points: int = 5000,
):
    """Plot moving points colored by nearest fixed-point residual distance."""
    fig, ax = plt.subplots(figsize=(8, 8))
    fixed_points = np.asarray(fixed_points, dtype=float)
    moving_points = np.asarray(moving_points, dtype=float)
    max_points = max(1, int(max_points))
    if len(fixed_points) == 0 or len(moving_points) == 0:
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        return fig

    from scipy.spatial import cKDTree

    moving_display = moving_points[:max_points]
    distances, _ = cKDTree(fixed_points).query(moving_display, k=1)
    ax.scatter(
        fixed_points[:max_points, 0],
        fixed_points[:max_points, 1],
        s=8,
        c="#00d1ff",
        alpha=0.25,
        linewidths=0,
        label="GeoJSON fixed",
    )
    scatter = ax.scatter(
        moving_display[:, 0],
        moving_display[:, 1],
        c=distances,
        s=14,
        cmap="magma",
        alpha=0.85,
        marker="x",
        label="HE residual",
    )
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="nearest residual")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    return fig


def visualize_distance_histogram(
    before_distances,
    after_distances,
    *,
    title="Nearest-neighbor distance before/after fine alignment",
    attempted_distances=None,
):
    """Plot before/after nearest-neighbor distance histograms."""
    fig, ax = plt.subplots(figsize=(8, 4))
    if attempted_distances is None:
        ax.hist(before_distances, bins=40, alpha=0.55, label="before", color="#999999")
        ax.hist(after_distances, bins=40, alpha=0.55, label="after", color="#00d1ff")
    else:
        ax.hist(before_distances, bins=40, alpha=0.45, label="Affine", color="#999999")
        ax.hist(attempted_distances, bins=40, alpha=0.45, label="Attempted fine", color="#ffb000")
        ax.hist(after_distances, bins=40, alpha=0.45, label="Applied fine", color="#00d1ff")
    ax.set_title(title)
    ax.set_xlabel("nearest distance")
    ax.set_ylabel("count")
    ax.legend(frameon=True)
    fig.tight_layout()
    return fig


def visualize_anchor_correlation_heatmap(anchors: pd.DataFrame, *, title: str):
    """Scatter heatmap of local translation anchor correlations."""
    fig, ax = plt.subplots(figsize=(8, 8))
    if anchors is None or anchors.empty or not {"anchor_x", "anchor_y"}.issubset(anchors.columns):
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        return fig

    anchors = anchors.copy()
    if "correlation" in anchors:
        values = anchors["correlation"].to_numpy(dtype=float)
    elif "best_correlation" in anchors:
        values = anchors["best_correlation"].to_numpy(dtype=float)
    elif "confidence" in anchors:
        values = anchors["confidence"].to_numpy(dtype=float)
    else:
        values = np.full(len(anchors), np.nan, dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        values = np.zeros(len(anchors), dtype=float)
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
