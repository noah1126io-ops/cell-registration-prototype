from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.spatial import cKDTree

from src.density import create_density_map
from src.export import array_to_png_bytes, figure_to_png_bytes
from src.features import extract_cell_features, point_features_to_cell_features
from src.geojson_utils import load_geojson_centroids
from src.io_utils import read_uploaded_image, read_uploaded_mask
from src.matching import match_cells
from src.point_io import load_csv_points, load_npy_centers
from src.pointset_registration import (
    cluster_anchor_fine_warp,
    estimate_affine_with_y_flip,
    FineWarpResult,
    fine_center_snap_warp,
    local_translation_fine_warp,
    matched_nuclei_rbf_fine_warp,
    point_distance_metrics,
    point_nearest_distances,
    warp_he_image_to_world,
    world_points_to_warped_image_pixels,
)
from src.registration import (
    estimate_affine_transform,
    transform_cell_centroids,
    warp_image,
    warp_mask,
)
from src.visualization import (
    colorize_label_image,
    visualize_cell_matches,
    visualize_anchor_correlation_heatmap,
    visualize_displacement_field,
    visualize_distance_histogram,
    visualize_geojson_classification_overlay,
    visualize_jacobian_heatmap,
    visualize_local_residual_map,
    visualize_point_sets,
    visualize_translation_anchors,
    visualize_warp_grid_overlay,
    visualize_warped_he_point_overlay,
    visualize_warped_pixel_point_scatter,
)


st.set_page_config(
    page_title="Cell Registration Prototype",
    layout="wide",
)


def show_uploaded_image(title: str, uploaded_file, *, is_mask: bool = False):
    st.subheader(title)

    if uploaded_file is None:
        st.info("Upload a file to preview it.")
        return None

    try:
        if is_mask:
            image = read_uploaded_mask(uploaded_file)
            preview = colorize_label_image(image)
            st.image(preview, caption=f"{uploaded_file.name} | shape={image.shape} | dtype={image.dtype}")
            st.caption(f"Label range: {int(image.min())} - {int(image.max())}")
        else:
            image = read_uploaded_image(uploaded_file)
            st.image(image, caption=f"{uploaded_file.name} | shape={image.shape} | dtype={image.dtype}")
        return image
    except Exception as exc:  # pragma: no cover - Streamlit UI feedback
        st.error(f"Could not read {uploaded_file.name}: {exc}")
        return None


def density_map_to_png(density_map: np.ndarray) -> bytes:
    return array_to_png_bytes(np.asarray(density_map, dtype=np.float32))


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


def create_overlay(fixed_image: np.ndarray, moving_image: np.ndarray) -> np.ndarray:
    fixed = _to_grayscale_preview(fixed_image)
    moving = _to_grayscale_preview(moving_image)

    height = min(fixed.shape[0], moving.shape[0])
    width = min(fixed.shape[1], moving.shape[1])
    fixed = fixed[:height, :width]
    moving = moving[:height, :width]

    overlay = np.zeros((height, width, 3), dtype=np.float32)
    overlay[..., 0] = moving
    overlay[..., 1] = fixed
    overlay[..., 2] = fixed
    return np.clip(overlay, 0.0, 1.0)


def show_feature_table(title: str, mask, image, filename: str):
    st.subheader(title)

    if mask is None:
        st.info("Upload a mask to extract cell features.")
        return None

    try:
        features = extract_cell_features(mask, image=image)
    except ValueError as exc:
        st.warning(f"{exc} Extracting features without intensity values.")
        features = extract_cell_features(mask)

    st.dataframe(features, use_container_width=True)
    st.download_button(
        "Download CSV",
        data=features.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=f"download-{filename}",
    )
    return features


def show_density_map(title: str, features, image_shape, sigma: float, filename: str):
    st.subheader(title)

    if features is None or image_shape is None:
        st.info("Upload a mask to create a density map.")
        return None

    density_map = create_density_map(features, image_shape, sigma=sigma)
    st.image(
        density_map,
        clamp=True,
        caption=f"{filename} | shape={density_map.shape} | sigma={sigma}",
    )
    st.download_button(
        "Download PNG",
        data=density_map_to_png(density_map),
        file_name=filename,
        mime="image/png",
        key=f"download-{filename}",
    )
    return density_map


def transformation_summary_to_json(registration_result) -> bytes:
    summary = {
        "registration_model": "affine",
        "moving_to_fixed": True,
        "success": registration_result.success,
        "message": registration_result.message,
        "ecc_score": registration_result.ecc_score,
        "affine_transform": registration_result.affine_matrix.tolist(),
        "fallback": not registration_result.success,
        "notes": "Affine only. Non-rigid registration is not implemented in this prototype stage.",
    }
    return json.dumps(summary, indent=2).encode("utf-8")


def show_registration_result(
    fixed_image,
    moving_image,
    fixed_mask,
    moving_mask,
    fixed_density_map,
    moving_density_map,
    moving_features,
):
    st.subheader("Affine registration")

    if fixed_density_map is None or moving_density_map is None:
        st.info("Upload both masks to estimate affine registration from density maps.")
        return moving_image, moving_mask, moving_features, None

    result = estimate_affine_transform(fixed_density_map, moving_density_map)
    if not result.success:
        st.warning(result.message)

    output_shape = fixed_density_map.shape
    warped_moving_image = None if moving_image is None else warp_image(moving_image, result.affine_matrix, output_shape)
    warped_moving_mask = None if moving_mask is None else warp_mask(moving_mask, result.affine_matrix, output_shape)
    warped_moving_density = warp_image(moving_density_map, result.affine_matrix, output_shape)
    transformed_moving_features = (
        None if moving_features is None else transform_cell_centroids(moving_features, result.affine_matrix)
    )

    density_before, density_after = st.columns(2)
    with density_before:
        st.image(create_overlay(fixed_density_map, moving_density_map), caption="Before registration density overlay")
    with density_after:
        st.image(create_overlay(fixed_density_map, warped_moving_density), caption="After registration density overlay")

    if fixed_image is not None and moving_image is not None and warped_moving_image is not None:
        before_left, after_right = st.columns(2)
        with before_left:
            st.image(create_overlay(fixed_image, moving_image), caption="Before registration overlay")
        with after_right:
            st.image(create_overlay(fixed_image, warped_moving_image), caption="After registration overlay")

    preview_left, preview_right = st.columns(2)
    with preview_left:
        if warped_moving_image is not None:
            st.image(warped_moving_image, caption="Warped moving image")
    with preview_right:
        if warped_moving_mask is not None:
            st.image(colorize_label_image(warped_moving_mask), caption="Warped moving mask")

    if transformed_moving_features is not None:
        st.subheader("Transformed moving cell centroids")
        st.dataframe(
            transformed_moving_features[["cell_id", "centroid_x", "centroid_y"]],
            use_container_width=True,
        )

    st.download_button(
        "Download transformation summary",
        data=transformation_summary_to_json(result),
        file_name="transformation_summary.json",
        mime="application/json",
        key="download-transformation-summary.json",
    )

    return warped_moving_image, warped_moving_mask, transformed_moving_features, result


def show_cell_correspondence(
    fixed_features,
    moving_features,
    *,
    max_distance: float,
    min_area_ratio: float,
    max_area_ratio: float,
    max_score: float,
    filename: str = "cell_correspondence.csv",
) -> None:
    st.subheader("Cell correspondence candidates")

    if fixed_features is None or moving_features is None:
        st.info("Upload both fixed and moving masks to estimate correspondence candidates.")
        return None

    try:
        matches = match_cells(
            fixed_features,
            moving_features,
            max_distance=max_distance,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            max_score=max_score,
        )
    except ValueError as exc:
        st.warning(str(exc))
        return None
    st.dataframe(matches, use_container_width=True)
    st.download_button(
        "Download CSV",
        data=matches.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=f"download-{filename}",
    )
    return matches


def show_match_visualization(
    fixed_image,
    fixed_features,
    moving_features,
    matches,
    max_pairs_to_display: int,
) -> None:
    st.subheader("Matched pair visualization")

    if fixed_features is None or moving_features is None or matches is None:
        st.info("Create correspondence results to visualize matches.")
        return

    try:
        figure = visualize_cell_matches(
            fixed_image,
            fixed_features,
            moving_features,
            matches,
            max_pairs=max_pairs_to_display,
        )
    except ValueError as exc:
        st.warning(str(exc))
        return

    st.pyplot(figure, clear_figure=False)
    st.download_button(
        "Download matched overlay",
        data=figure_to_png_bytes(figure),
        file_name="matched_cells_overlay.png",
        mime="image/png",
        key="download-matched-cells-overlay.png",
    )


def load_uploaded_points(uploaded_file, *, coordinate_order: str):
    if uploaded_file is None:
        return None

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".npy":
        return load_npy_centers(
            uploaded_file,
            point_source=suffix.lstrip("."),
            coordinate_order=coordinate_order,
        )
    if suffix == ".csv":
        return load_csv_points(uploaded_file, point_source=suffix.lstrip("."))
    raise ValueError("Point files must be .npy or .csv.")


def infer_canvas_shape(*feature_tables, background_image=None, margin: int = 50) -> tuple[int, int]:
    if background_image is not None:
        return tuple(background_image.shape[:2])

    max_x = 0.0
    max_y = 0.0
    for features in feature_tables:
        if features is None or features.empty:
            continue
        max_x = max(max_x, float(features["centroid_x"].max()))
        max_y = max(max_y, float(features["centroid_y"].max()))

    width = max(64, int(np.ceil(max_x)) + margin)
    height = max(64, int(np.ceil(max_y)) + margin)
    return height, width


def show_point_table(title: str, points, filename: str):
    st.subheader(title)
    if points is None:
        st.info("Upload a point file to preview it.")
        return None
    st.dataframe(points, use_container_width=True)
    st.download_button(
        "Download normalized points",
        data=points.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=f"download-{filename}",
    )
    return point_features_to_cell_features(points)


def show_point_registration_workflow(
    *,
    density_sigma: float,
    max_distance: float,
    min_area_ratio: float,
    max_area_ratio: float,
    max_score: float,
    max_pairs_to_display: int,
) -> None:
    st.header("Workflow A: Point registration")
    st.caption("Register and match precomputed point tables. Images are optional QC backgrounds.")

    point_left, point_right = st.columns(2)
    with point_left:
        fixed_points_file = st.file_uploader("Fixed points file", type=["npy", "csv"])
        fixed_coordinate_order = st.selectbox("Fixed .npy coordinate order", ["xy", "yx"], key="fixed-point-order")
    with point_right:
        moving_points_file = st.file_uploader("Moving points file", type=["npy", "csv"])
        moving_coordinate_order = st.selectbox("Moving .npy coordinate order", ["xy", "yx"], key="moving-point-order")

    bg_left, bg_right = st.columns(2)
    with bg_left:
        fixed_background_file = st.file_uploader("Optional fixed background image", type=["png", "jpg", "jpeg", "tif", "tiff"])
        fixed_background = show_uploaded_image("Fixed background", fixed_background_file) if fixed_background_file else None
    with bg_right:
        moving_background_file = st.file_uploader("Optional moving background image", type=["png", "jpg", "jpeg", "tif", "tiff"])
        moving_background = show_uploaded_image("Moving background", moving_background_file) if moving_background_file else None

    try:
        fixed_points = load_uploaded_points(fixed_points_file, coordinate_order=fixed_coordinate_order)
        moving_points = load_uploaded_points(moving_points_file, coordinate_order=moving_coordinate_order)
    except ValueError as exc:
        st.warning(str(exc))
        return

    table_left, table_right = st.columns(2)
    with table_left:
        fixed_features = show_point_table("Fixed point table", fixed_points, "fixed_points_normalized.csv")
    with table_right:
        moving_features = show_point_table("Moving point table", moving_points, "moving_points_normalized.csv")

    if fixed_features is None or moving_features is None:
        st.info("Upload both fixed and moving point files to run registration.")
        return

    canvas_shape = infer_canvas_shape(fixed_features, moving_features, background_image=fixed_background)
    fixed_density_map = create_density_map(fixed_features, canvas_shape, sigma=density_sigma)
    moving_density_map = create_density_map(moving_features, canvas_shape, sigma=density_sigma)

    st.divider()
    density_left, density_right = st.columns(2)
    with density_left:
        st.image(fixed_density_map, clamp=True, caption=f"Fixed density map | shape={fixed_density_map.shape}")
        st.download_button(
            "Download fixed density PNG",
            data=density_map_to_png(fixed_density_map),
            file_name="fixed_density_map.png",
            mime="image/png",
            key="download-fixed-density-point.png",
        )
    with density_right:
        st.image(moving_density_map, clamp=True, caption=f"Moving density map | shape={moving_density_map.shape}")
        st.download_button(
            "Download moving density PNG",
            data=density_map_to_png(moving_density_map),
            file_name="moving_density_map.png",
            mime="image/png",
            key="download-moving-density-point.png",
        )

    result = estimate_affine_transform(fixed_density_map, moving_density_map)
    if not result.success:
        st.warning(result.message)

    transformed_moving_features = transform_cell_centroids(moving_features, result.affine_matrix)
    warped_moving_background = None
    if moving_background is not None:
        warped_moving_background = warp_image(moving_background, result.affine_matrix, canvas_shape)

    st.divider()
    before_after_left, before_after_right = st.columns(2)
    with before_after_left:
        before_figure = visualize_point_sets(
            fixed_features,
            moving_features,
            title="Before registration",
            background_image=fixed_background,
        )
        st.pyplot(before_figure, clear_figure=False)
    with before_after_right:
        after_figure = visualize_point_sets(
            fixed_features,
            transformed_moving_features,
            title="After registration",
            background_image=fixed_background,
        )
        st.pyplot(after_figure, clear_figure=False)

    matches = show_cell_correspondence(
        fixed_features,
        transformed_moving_features,
        max_distance=max_distance,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        max_score=max_score,
        filename="matched_points.csv",
    )

    show_match_visualization(
        fixed_background,
        fixed_features,
        transformed_moving_features,
        matches,
        max_pairs_to_display=max_pairs_to_display,
    )

    st.download_button(
        "Download transform summary",
        data=transformation_summary_to_json(result),
        file_name="transform_summary.json",
        mime="application/json",
        key="download-point-transform-summary.json",
    )
    if warped_moving_background is not None:
        st.image(warped_moving_background, caption="Warped moving background")
        st.download_button(
            "Download warped moving image",
            data=array_to_png_bytes(warped_moving_background),
            file_name="warped_moving_background.png",
            mime="image/png",
            key="download-warped-moving-background.png",
        )


def show_mask_to_mask_workflow(
    *,
    density_sigma: float,
    max_distance: float,
    min_area_ratio: float,
    max_area_ratio: float,
    max_score: float,
    max_pairs_to_display: int,
) -> None:
    st.header("Workflow B: Mask-derived point registration")
    st.caption("Extract points/features from existing label masks, then run the shared registration and matching flow.")

    st.subheader("Input files")
    file_types = ["png", "jpg", "jpeg", "tif", "tiff"]

    fixed_image = st.file_uploader("Fixed image (optional QC background)", type=file_types, key="mask-fixed-image")
    moving_image = st.file_uploader("Moving image (optional QC background)", type=file_types, key="mask-moving-image")
    fixed_mask = st.file_uploader("Fixed mask", type=file_types, key="mask-fixed-mask")
    moving_mask = st.file_uploader("Moving mask", type=file_types, key="mask-moving-mask")

    image_left, image_right = st.columns(2)
    with image_left:
        fixed_image_array = show_uploaded_image("Fixed image", fixed_image)
    with image_right:
        moving_image_array = show_uploaded_image("Moving image", moving_image)

    mask_left, mask_right = st.columns(2)
    with mask_left:
        fixed_mask_array = show_uploaded_image("Fixed mask", fixed_mask, is_mask=True)
    with mask_right:
        moving_mask_array = show_uploaded_image("Moving mask", moving_mask, is_mask=True)

    st.divider()
    feature_left, feature_right = st.columns(2)
    with feature_left:
        fixed_features = show_feature_table(
            "Fixed cell features",
            fixed_mask_array,
            fixed_image_array,
            "fixed_cell_features.csv",
        )
    with feature_right:
        moving_features = show_feature_table(
            "Moving cell features",
            moving_mask_array,
            moving_image_array,
            "moving_cell_features.csv",
        )

    st.divider()
    density_left, density_right = st.columns(2)
    with density_left:
        fixed_density_map = show_density_map(
            "Fixed density map",
            fixed_features,
            None if fixed_mask_array is None else fixed_mask_array.shape,
            density_sigma,
            "fixed_density_map.png",
        )
    with density_right:
        moving_density_map = show_density_map(
            "Moving density map",
            moving_features,
            None if moving_mask_array is None else moving_mask_array.shape,
            density_sigma,
            "moving_density_map.png",
        )

    st.divider()
    _, _, transformed_moving_features, _ = show_registration_result(
        fixed_image_array,
        moving_image_array,
        fixed_mask_array,
        moving_mask_array,
        fixed_density_map,
        moving_density_map,
        moving_features,
    )

    st.divider()
    matches = show_cell_correspondence(
        fixed_features,
        transformed_moving_features,
        max_distance=max_distance,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        max_score=max_score,
    )

    st.divider()
    show_match_visualization(
        fixed_image_array,
        fixed_features,
        transformed_moving_features,
        matches,
        max_pairs_to_display=max_pairs_to_display,
    )

    # TODO: Add segmentation source selection while preserving label-mask input.
    # TODO: Add optional Cellpose adapter without making Cellpose the only source.
    # TODO: Add StarDist nuclei centers .npy upload.
    # TODO: Add GeoJSON nuclei upload.
    # TODO: Add point-set registration mode.
    # TODO: Add GeoJSON nuclei polygon QC support for HE-GeoJSON alignment mode.
    # TODO: Add HE nuclei .npy input support for precomputed StarDist centers.
    # TODO: Keep HE-GeoJSON alignment as a special-coordinate workflow separate from mask-derived point registration.
    # TODO: Add world-um coordinate transform utilities for GeoJSON alignment.
    # TODO: Add Y-flip handling for fluorescence GeoJSON world coordinates.
    # TODO: Add Jacobian quality check for fine center-snap warp validation.
    # TODO: Add multi-scale density map presets for registration experiments.
    # TODO: Add non-rigid registration after affine registration is validated.
    # TODO: Add batch export of matched cells, transforms, and preview images.


def _points_from_table(points: pd.DataFrame) -> np.ndarray:
    return points[["centroid_x", "centroid_y"]].to_numpy(dtype=float)


def _cell_features_from_points(points: pd.DataFrame):
    return point_features_to_cell_features(points)


def _identity_fine_result(affine_points: np.ndarray, fixed_points: np.ndarray, grid_spacing: float, message: str) -> FineWarpResult:
    min_x, min_y = np.min(np.vstack([fixed_points, affine_points]), axis=0) - 30.0
    max_x, max_y = np.max(np.vstack([fixed_points, affine_points]), axis=0) + 30.0
    xs = np.arange(min_x, max_x + grid_spacing, grid_spacing)
    ys = np.arange(min_y, max_y + grid_spacing, grid_spacing)
    grid_x, grid_y = np.meshgrid(xs, ys)
    zeros = np.zeros_like(grid_x, dtype=float)
    metrics = point_distance_metrics(fixed_points, affine_points)
    return FineWarpResult(
        transformed_points=affine_points.copy(),
        grid_x=grid_x,
        grid_y=grid_y,
        displacement_x=zeros,
        displacement_y=zeros,
        attempted_transformed_points=affine_points.copy(),
        attempted_displacement_x=zeros,
        attempted_displacement_y=zeros,
        bounds=(float(min_x), float(min_y), float(max_x), float(max_y)),
        grid_spacing=float(grid_spacing),
        jacobian_min=1.0,
        jacobian_max=1.0,
        max_displacement=0.0,
        n_candidate_pairs=0,
        n_pairs=0,
        n_filtered_pairs=0,
        median_pair_distance_before=metrics["median_distance"],
        median_pair_distance_after=metrics["median_distance"],
        success=False,
        message=message,
        attempted_metrics=metrics,
        applied_metrics=metrics,
        rejection_reason=message,
        applied=False,
        anchors=None,
        metrics={"before": metrics, "attempted": metrics, "applied": metrics},
    )


def _coordinate_summary_row(label: str, points: np.ndarray) -> dict:
    points = np.asarray(points, dtype=float)
    finite = points[np.all(np.isfinite(points), axis=1)] if points.size else np.empty((0, 2), dtype=float)
    if finite.size == 0:
        return {
            "point_set": label,
            "n_points": 0,
            "min_x": np.nan,
            "max_x": np.nan,
            "min_y": np.nan,
            "max_y": np.nan,
            "width": np.nan,
            "height": np.nan,
            "mean_x": np.nan,
            "mean_y": np.nan,
            "median_x": np.nan,
            "median_y": np.nan,
        }
    min_x = float(np.min(finite[:, 0]))
    max_x = float(np.max(finite[:, 0]))
    min_y = float(np.min(finite[:, 1]))
    max_y = float(np.max(finite[:, 1]))
    return {
        "point_set": label,
        "n_points": int(len(finite)),
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "width": max_x - min_x,
        "height": max_y - min_y,
        "mean_x": float(np.mean(finite[:, 0])),
        "mean_y": float(np.mean(finite[:, 1])),
        "median_x": float(np.median(finite[:, 0])),
        "median_y": float(np.median(finite[:, 1])),
    }


def _pixel_coordinate_summary_row(label: str, pixels: np.ndarray, *, image_width: int, image_height: int) -> dict:
    pixels = np.asarray(pixels, dtype=float)
    finite = pixels[np.all(np.isfinite(pixels), axis=1)] if pixels.size else np.empty((0, 2), dtype=float)
    if finite.size == 0:
        return {
            "point_set": label,
            "min_col": np.nan,
            "max_col": np.nan,
            "min_row": np.nan,
            "max_row": np.nan,
            "mean_col": np.nan,
            "mean_row": np.nan,
            "n_outside_image": 0,
            "fraction_outside_image": np.nan,
        }
    outside = (
        (finite[:, 0] < 0)
        | (finite[:, 0] >= float(image_width))
        | (finite[:, 1] < 0)
        | (finite[:, 1] >= float(image_height))
    )
    return {
        "point_set": label,
        "min_col": float(np.min(finite[:, 0])),
        "max_col": float(np.max(finite[:, 0])),
        "min_row": float(np.min(finite[:, 1])),
        "max_row": float(np.max(finite[:, 1])),
        "mean_col": float(np.mean(finite[:, 0])),
        "mean_row": float(np.mean(finite[:, 1])),
        "n_outside_image": int(np.count_nonzero(outside)),
        "fraction_outside_image": float(np.mean(outside)),
    }


def _field_jacobian(displacement_x: np.ndarray, displacement_y: np.ndarray, grid_spacing: float) -> np.ndarray:
    dfx_dy, dfx_dx = np.gradient(displacement_x, grid_spacing, grid_spacing)
    dfy_dy, dfy_dx = np.gradient(displacement_y, grid_spacing, grid_spacing)
    return (1.0 + dfx_dx) * (1.0 + dfy_dy) - dfx_dy * dfy_dx


def _jacobian_summary_row(label: str, jacobian: np.ndarray) -> dict:
    jacobian = np.asarray(jacobian, dtype=float)
    finite = jacobian[np.isfinite(jacobian)]
    if finite.size == 0:
        return {
            "field": label,
            "jacobian_min": np.nan,
            "jacobian_max": np.nan,
            "jacobian_median": np.nan,
            "fraction_expansion_gt_1": np.nan,
            "fraction_compression_lt_1": np.nan,
            "fraction_foldover_le_0": np.nan,
        }
    return {
        "field": label,
        "jacobian_min": float(np.min(finite)),
        "jacobian_max": float(np.max(finite)),
        "jacobian_median": float(np.median(finite)),
        "fraction_expansion_gt_1": float(np.mean(finite > 1.0)),
        "fraction_compression_lt_1": float(np.mean((finite > 0.0) & (finite < 1.0))),
        "fraction_foldover_le_0": float(np.mean(finite <= 0.0)),
    }


def _tissue_mask_from_image(image: np.ndarray, threshold: float) -> np.ndarray:
    gray = _to_grayscale_preview(image)
    threshold = float(np.clip(threshold, 0.0, 1.0))
    return gray > threshold


def _point_tissue_validity(
    points: np.ndarray,
    warp_metadata: dict,
    tissue_mask: np.ndarray,
    *,
    edge_margin: float,
    moving_points: np.ndarray | None = None,
    max_nearest_he_distance: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    pixels = world_points_to_warped_image_pixels(points, warp_metadata)
    height, width = tissue_mask.shape[:2]
    cols = pixels[:, 0]
    rows = pixels[:, 1]
    edge_margin = float(max(edge_margin, 0.0))
    inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    row_i = np.clip(np.rint(rows).astype(int), 0, max(height - 1, 0))
    col_i = np.clip(np.rint(cols).astype(int), 0, max(width - 1, 0))
    tissue = np.zeros(len(points), dtype=bool)
    if len(points):
        tissue[inside] = tissue_mask[row_i[inside], col_i[inside]]

    tissue_distance_inside = distance_transform_edt(tissue_mask)
    tissue_distance_outside = distance_transform_edt(~tissue_mask)
    dist_inside = np.zeros(len(points), dtype=float)
    dist_to_tissue = np.full(len(points), np.inf, dtype=float)
    if len(points):
        dist_inside[inside] = tissue_distance_inside[row_i[inside], col_i[inside]]
        dist_to_tissue[inside] = tissue_distance_outside[row_i[inside], col_i[inside]]

    near_he = np.ones(len(points), dtype=bool)
    nearest_he_distance = np.full(len(points), np.nan, dtype=float)
    if moving_points is not None and max_nearest_he_distance is not None and len(moving_points):
        nearest_he_distance, _ = cKDTree(moving_points).query(points, k=1)
        near_he = nearest_he_distance <= float(max_nearest_he_distance)

    stable_image_edge = (
        (cols >= edge_margin)
        & (cols < width - edge_margin)
        & (rows >= edge_margin)
        & (rows < height - edge_margin)
    )
    valid = inside & tissue & stable_image_edge & (dist_inside >= edge_margin) & near_he
    edge_candidate = (
        inside
        & near_he
        & ~valid
        & (
            (tissue & (dist_inside < edge_margin))
            | (~tissue & (dist_to_tissue <= edge_margin))
            | ~stable_image_edge
        )
    )
    classification = np.full(len(points), "excluded", dtype=object)
    classification[edge_candidate] = "edge_candidate"
    classification[valid] = "valid"
    weights = np.zeros(len(points), dtype=float)
    weights[valid] = 1.0
    weights[edge_candidate] = 0.3

    reason = classification.copy()
    reason[~inside] = "outside_image"
    reason[inside & ~tissue & (dist_to_tissue > edge_margin)] = "background_or_far_from_tissue"
    reason[inside & tissue & ~stable_image_edge] = "near_image_edge"
    reason[inside & ~near_he] = "no_nearby_he_points"
    diagnostics = pd.DataFrame(
        {
            "centroid_x": points[:, 0],
            "centroid_y": points[:, 1],
            "col": cols,
            "row": rows,
            "classification": classification,
            "target_weight": weights,
            "valid_for_fine_warp": classification != "excluded",
            "distance_inside_tissue_px": dist_inside,
            "distance_to_tissue_px": dist_to_tissue,
            "nearest_he_distance_um": nearest_he_distance,
            "exclusion_reason": reason,
        }
    )
    return classification != "excluded", weights, classification, diagnostics


def _warped_pixels_to_world_points(pixels: np.ndarray, warp_metadata: dict) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=float)
    pixel_size = float(warp_metadata["output_pixel_size_um"])
    output_origin = warp_metadata.get("output_origin", "upper-left")
    col0_world_x = float(warp_metadata["col0_world_x"])
    row0_world_y = float(warp_metadata["row0_world_y"])
    if output_origin == "upper-right":
        x = col0_world_x - pixels[:, 0] * pixel_size
    else:
        x = col0_world_x + pixels[:, 0] * pixel_size
    if output_origin in {"upper-left", "upper-right"}:
        y = row0_world_y + pixels[:, 1] * pixel_size
    else:
        y = row0_world_y - pixels[:, 1] * pixel_size
    return np.column_stack([x, y])


def _boundary_anchor_points_from_tissue_mask(
    tissue_mask: np.ndarray,
    warp_metadata: dict,
    *,
    spacing_px: float,
    include_image_border: bool = True,
) -> np.ndarray:
    height, width = tissue_mask.shape[:2]
    spacing_px = max(float(spacing_px), 1.0)
    pixels = []
    if include_image_border:
        xs = np.arange(0, width, spacing_px)
        ys = np.arange(0, height, spacing_px)
        pixels.extend([(x, 0.0) for x in xs])
        pixels.extend([(x, float(height - 1)) for x in xs])
        pixels.extend([(0.0, y) for y in ys])
        pixels.extend([(float(width - 1), y) for y in ys])

    if np.any(tissue_mask):
        eroded = binary_erosion(tissue_mask, iterations=1, border_value=0)
        boundary = tissue_mask & ~eroded
        rows, cols = np.where(boundary)
        if len(rows):
            stride = max(1, int(round(spacing_px)))
            order = np.arange(0, len(rows), stride)
            pixels.extend([(float(cols[i]), float(rows[i])) for i in order])

    if not pixels:
        return np.empty((0, 2), dtype=float)
    pixels = np.unique(np.asarray(pixels, dtype=float), axis=0)
    return _warped_pixels_to_world_points(pixels, warp_metadata)


def _warp_grid_lines_world(bounds, spacing: float, samples_per_line: int = 80) -> list[np.ndarray]:
    min_x, min_y, max_x, max_y = bounds
    xs = np.arange(min_x, max_x + spacing, spacing)
    ys = np.arange(min_y, max_y + spacing, spacing)
    line_ys = np.linspace(min_y, max_y, samples_per_line)
    line_xs = np.linspace(min_x, max_x, samples_per_line)
    lines = []
    for x in xs:
        lines.append(np.column_stack([np.full_like(line_ys, x), line_ys]))
    for y in ys:
        lines.append(np.column_stack([line_xs, np.full_like(line_xs, y)]))
    return lines


def _warp_grid_lines_pixels(bounds, spacing: float, warp_metadata: dict, fine_result=None) -> list[np.ndarray]:
    lines = []
    for line in _warp_grid_lines_world(bounds, spacing):
        world_line = line
        if fine_result is not None:
            from src.pointset_registration import _sample_field

            sampled = _sample_field(
                world_line,
                fine_result.displacement_x,
                fine_result.displacement_y,
                fine_result.bounds,
                fine_result.grid_spacing,
            )
            world_line = world_line + sampled
        lines.append(world_points_to_warped_image_pixels(world_line, warp_metadata))
    return lines


def _he_geojson_summary_to_json(affine_result, fine_result, parameters: dict, warp_metadata: dict | None = None) -> bytes:
    fine_applied_value = getattr(fine_result, "applied", None)
    fine_applied = fine_result.success if fine_applied_value is None else bool(fine_applied_value)
    rejection_reason = getattr(fine_result, "rejection_reason", None) or ""
    attempted_metrics = getattr(fine_result, "attempted_metrics", None)
    applied_metrics = getattr(fine_result, "applied_metrics", None)
    summary = {
        "workflow": "HE-GeoJSON alignment",
        "coordinate_system": "GeoJSON world-um",
        "affine": {
            "model": "xy_flip_optional_affine_icp",
            "flip_x": affine_result.flip_x,
            "flip_y": affine_result.flip_y,
            "image_width_px": affine_result.image_width,
            "image_height_px": affine_result.image_height,
            "affine_matrix": affine_result.affine_matrix.tolist(),
            "translation": affine_result.translation.tolist(),
            "mean_residual_um": affine_result.mean_residual,
            "median_residual_um": affine_result.median_residual,
            "n_pairs": affine_result.n_pairs,
            "success": affine_result.success,
            "message": affine_result.message,
        },
        "fine_center_snap": {
            "model": "confidence_weighted_smooth_displacement_field",
            "bounds_um": fine_result.bounds,
            "grid_spacing_um": fine_result.grid_spacing,
            "jacobian_min": fine_result.jacobian_min,
            "jacobian_max": fine_result.jacobian_max,
            "max_displacement_um": fine_result.max_displacement,
            "n_candidate_pairs": fine_result.n_candidate_pairs,
            "n_pairs": fine_result.n_pairs,
            "n_filtered_pairs": fine_result.n_filtered_pairs,
            "median_pair_distance_before_um": fine_result.median_pair_distance_before,
            "median_pair_distance_after_um": fine_result.median_pair_distance_after,
            "applied": fine_applied,
            "rejection_reason": rejection_reason,
            "attempted_metrics": attempted_metrics,
            "applied_metrics": applied_metrics,
            "distance_metrics": fine_result.metrics,
            "success": fine_result.success,
            "message": fine_result.message,
        },
        "warped_he_image": warp_metadata,
        "parameters": parameters,
        "notes": "Research prototype only. Raster HE warp export is an MVP for QC and should be visually checked.",
    }
    return json.dumps(summary, indent=2).encode("utf-8")


def show_he_geojson_preparation() -> None:
    st.header("Workflow C: HE-GeoJSON alignment")
    st.caption("Special-coordinate workflow for HE nuclei points and fluorescence GeoJSON in world-um space.")

    input_left, input_right = st.columns(2)
    with input_left:
        he_centers_file = st.file_uploader("HE nuclei centers .npy", type=["npy"], key="he-nuclei-npy")
        he_coordinate_order = st.selectbox("HE .npy coordinate order", ["xy", "yx"], key="he-nuclei-order")
        he_image_file = st.file_uploader(
            "Optional HE image for y-flip height / QC background",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="he-qc-image",
        )
    with input_right:
        geojson_file = st.file_uploader(
            "Fluorescence nuclei GeoJSON",
            type=["geojson", "json"],
            key="fluorescence-geojson",
        )

    st.subheader("Point-set registration parameters")
    fine_alignment_method = st.selectbox(
        "Fine alignment method",
        ["cluster-anchor", "matched nuclei RBF", "local translation field", "center-snap", "off"],
        index=0,
        help="Cluster-anchor searches local HE cluster translations against fixed GeoJSON clusters; GeoJSON points are never moved.",
    )
    local_preset = st.selectbox(
        "Local translation preset",
        ["conservative", "balanced", "aggressive", "debug"],
        index=1,
        help="Preset defaults for local translation field. Debug is permissive and useful for diagnosis.",
    )
    local_presets = {
        "conservative": {
            "density_sigma": 3.0,
            "grid_spacing": 60.0,
            "patch_radius": 25.0,
            "search_radius": 12.0,
            "min_correlation": 0.25,
            "max_shift": 20.0,
            "smoothing": 10.0,
            "min_accepted_anchors": 8,
        },
        "balanced": {
            "density_sigma": 2.0,
            "grid_spacing": 35.0,
            "patch_radius": 18.0,
            "search_radius": 25.0,
            "min_correlation": 0.15,
            "max_shift": 35.0,
            "smoothing": 3.0,
            "min_accepted_anchors": 5,
        },
        "aggressive": {
            "density_sigma": 1.5,
            "grid_spacing": 25.0,
            "patch_radius": 12.0,
            "search_radius": 35.0,
            "min_correlation": 0.10,
            "max_shift": 45.0,
            "smoothing": 0.0,
            "min_accepted_anchors": 3,
        },
        "debug": {
            "density_sigma": 2.0,
            "grid_spacing": 30.0,
            "patch_radius": 15.0,
            "search_radius": 35.0,
            "min_correlation": 0.05,
            "max_shift": 50.0,
            "smoothing": 0.0,
            "min_accepted_anchors": 1,
        },
    }
    local_default = local_presets[local_preset]
    param_left, param_mid, param_right = st.columns(3)
    with param_left:
        similarity_trim = st.slider("Similarity ICP trim quantile", 0.1, 1.0, 0.8, 0.05)
        affine_trim = st.slider("Affine ICP trim quantile", 0.1, 1.0, 0.7, 0.05)
    with param_mid:
        match_radius = st.number_input("Fine match radius (um)", min_value=0.1, value=10.0, step=1.0)
        fine_bandwidth = st.number_input("Fine warp bandwidth (um)", min_value=0.1, value=12.0, step=1.0)
    with param_right:
        grid_spacing = st.number_input("Fine warp grid spacing (um)", min_value=0.1, value=6.0, step=1.0)
        ridge = st.number_input("Fine warp ridge", min_value=0.0, value=0.3, step=0.1)
    st.subheader("Robust fine-snap filtering")
    robust_left, robust_mid, robust_right = st.columns(3)
    with robust_left:
        min_pair_confidence = st.slider("Min pair confidence", 0.0, 1.0, 0.05, 0.01)
        coherence_radius = st.number_input("Local coherence radius (um)", min_value=0.1, value=30.0, step=5.0)
    with robust_mid:
        max_local_deviation = st.number_input("Max local deviation (um)", min_value=0.1, value=8.0, step=1.0)
        snap_strength = st.slider("Fine snap strength", 0.1, 2.5, 1.25, 0.05)
    with robust_right:
        max_snap_displacement = st.number_input("Max snap displacement (um)", min_value=0.1, value=25.0, step=5.0)
        min_matched_anchor_pairs = st.number_input("Min matched nuclei anchors", min_value=1, value=6, step=1)
    st.subheader("Local translation field")
    local_left, local_mid, local_right = st.columns(3)
    with local_left:
        local_density_sigma = st.number_input(
            "Density sigma (um)", min_value=0.1, value=local_default["density_sigma"], step=0.5
        )
        local_density_pixel_size = st.number_input("Density pixel size (um)", min_value=0.1, value=1.0, step=0.5)
        local_point_weight = st.number_input("Point weight", min_value=0.1, value=1.0, step=0.5)
    with local_mid:
        local_grid_spacing = st.number_input(
            "Local grid spacing (um)", min_value=1.0, value=local_default["grid_spacing"], step=5.0
        )
        local_patch_radius = st.number_input(
            "Patch radius (um)", min_value=1.0, value=local_default["patch_radius"], step=5.0
        )
        local_search_radius = st.number_input(
            "Search radius (um)", min_value=1.0, value=local_default["search_radius"], step=2.0
        )
    with local_right:
        local_min_correlation = st.slider(
            "Min local correlation", 0.0, 1.0, local_default["min_correlation"], 0.05
        )
        local_max_shift = st.number_input(
            "Max local shift (um)", min_value=1.0, value=local_default["max_shift"], step=5.0
        )
        local_min_anchors = st.number_input(
            "Min accepted anchors", min_value=1, value=local_default["min_accepted_anchors"], step=1
        )
    local_outlier_percentile = st.slider("Anchor outlier percentile", 50.0, 100.0, 95.0, 1.0)
    local_neighbor_radius = st.number_input("Neighbor consistency radius (um)", min_value=1.0, value=120.0, step=10.0)
    local_smoothing = st.number_input("RBF smoothing", min_value=0.0, value=local_default["smoothing"], step=1.0)
    local_kernel = st.selectbox("RBF kernel", ["thin_plate_spline", "linear", "cubic", "quintic"], index=0)
    local_neighbors = st.number_input("RBF neighbors", min_value=0, value=50, step=5)
    st.subheader("Cluster-anchor fine warp")
    cluster_left, cluster_mid, cluster_right = st.columns(3)
    with cluster_left:
        cluster_grid_spacing = st.number_input("Cluster grid spacing (um)", min_value=1.0, value=35.0, step=5.0)
        cluster_patch_radius = st.number_input("Cluster patch radius (um)", min_value=1.0, value=18.0, step=2.0)
        cluster_search_radius = st.number_input("Cluster search radius (um)", min_value=1.0, value=25.0, step=2.0)
    with cluster_mid:
        cluster_search_step = st.number_input("Cluster search step (um)", min_value=0.1, value=2.5, step=0.5)
        min_points_per_cluster = st.number_input("Min points per cluster", min_value=1, value=5, step=1)
        cluster_match_threshold = st.number_input("Cluster match threshold (um)", min_value=0.1, value=5.0, step=0.5)
    with cluster_right:
        cluster_min_improvement = st.number_input("Min cluster improvement (um)", min_value=0.0, value=1.0, step=0.5)
        cluster_max_shift = st.number_input("Max cluster shift (um)", min_value=1.0, value=35.0, step=5.0)
        cluster_min_anchors = st.number_input("Min cluster anchors", min_value=1, value=5, step=1)
    bspline_left, bspline_right = st.columns(2)
    with bspline_left:
        cluster_interpolation = st.selectbox("Cluster warp interpolation", ["rbf", "b-spline"], index=0)
        control_grid_spacing = st.number_input("B-spline control grid spacing (um)", min_value=1.0, value=35.0, step=5.0)
        local_support_radius = st.number_input("Local support radius (um)", min_value=1.0, value=120.0, step=10.0)
    with bspline_right:
        cluster_regularization = st.number_input("B-spline regularization", min_value=0.0, value=3.0, step=1.0)
        tissue_mask_threshold = st.slider("Tissue mask threshold", 0.0, 1.0, 0.05, 0.01)
    edge_margin = st.number_input("Fine-warp tissue edge margin (px)", min_value=0.0, value=10.0, step=2.0)
    pin_left, pin_right = st.columns(2)
    with pin_left:
        enable_boundary_pinning = st.checkbox("Enable boundary pinning", value=True)
        boundary_anchor_spacing = st.number_input("Boundary anchor spacing (px)", min_value=2.0, value=40.0, step=5.0)
    with pin_right:
        boundary_anchor_weight = st.number_input("Boundary anchor weight", min_value=0.01, value=3.0, step=0.5)
        jacobian_max_threshold = st.number_input("Jacobian max threshold", min_value=1.0, value=4.0, step=0.5)
    registration_display_origin = st.selectbox(
        "Registration QC display origin",
        ["lower-left", "upper-left", "upper-right"],
        index=0,
        help="Controls only the scatter/QC registration plots. Use lower-left if the registration plot appears upside-down relative to the uploaded HE image.",
    )
    warped_he_output_origin = st.selectbox(
        "Warped HE output origin",
        ["lower-left", "upper-left", "upper-right"],
        index=0,
        help="Controls only the exported warped HE image orientation. Use lower-left if the warped HE output appears upside-down.",
    )
    st.info("Warped HE output origin controls exported image orientation, not registration quality.")
    flip_mode = st.selectbox(
        "HE coordinate flip candidates",
        ["auto", "none", "x", "y", "x+y"],
        index=0,
        help="Use y when HE image coordinates are top-left-origin and the GeoJSON world coordinates are bottom-left-origin.",
    )
    warped_he_pixel_size = st.number_input(
        "Warped HE output pixel size (um)",
        min_value=0.1,
        value=1.0,
        step=0.5,
        help="Smaller values create larger PNG files.",
    )
    max_warped_overlay_points = st.number_input(
        "Max warped HE overlay points",
        min_value=100,
        max_value=20000,
        value=3000,
        step=500,
    )

    he_image = show_uploaded_image("Optional HE image", he_image_file) if he_image_file else None

    try:
        he_points = (
            None
            if he_centers_file is None
            else load_npy_centers(
                he_centers_file,
                point_source="he_npy",
                coordinate_order=he_coordinate_order,
            )
        )
        geojson_points = (
            None
            if geojson_file is None
            else load_geojson_centroids(geojson_file, point_source="fluorescence_geojson")
        )
    except ValueError as exc:
        st.warning(str(exc))
        return

    table_left, table_right = st.columns(2)
    with table_left:
        st.subheader("HE nuclei centers")
        if he_points is None:
            st.info("Upload HE nuclei .npy centers.")
        else:
            st.dataframe(he_points.head(500), use_container_width=True)
            st.download_button(
                "Download normalized HE centers",
                data=he_points.to_csv(index=False).encode("utf-8"),
                file_name="he_nuclei_centers_normalized.csv",
                mime="text/csv",
            )
    with table_right:
        st.subheader("GeoJSON nuclei centroids")
        if geojson_points is None:
            st.info("Upload fluorescence nuclei GeoJSON.")
        else:
            st.dataframe(geojson_points.head(500), use_container_width=True)
            st.download_button(
                "Download GeoJSON centroids",
                data=geojson_points.to_csv(index=False).encode("utf-8"),
                file_name="geojson_nuclei_centroids.csv",
                mime="text/csv",
            )

    if he_points is None or geojson_points is None:
        st.info("Upload both HE .npy centers and GeoJSON centroids to run Workflow C.")
        return

    if he_image is not None:
        image_height_px = float(he_image.shape[0])
        image_width_px = float(he_image.shape[1])
    else:
        image_height_px = float(np.ceil(he_points["centroid_y"].max()))
        image_width_px = float(np.ceil(he_points["centroid_x"].max()))
        st.warning(
            "No HE image was uploaded. Flip candidates use the maximum HE x/y coordinates as image size estimates."
        )

    he_array = _points_from_table(he_points)
    geojson_array = _points_from_table(geojson_points)
    affine_tissue_image = None
    affine_tissue_metadata = None

    try:
        flip_candidates = {
            "auto": None,
            "none": ((False, False),),
            "x": ((True, False),),
            "y": ((False, True),),
            "x+y": ((True, True),),
        }[flip_mode]
        affine_result = estimate_affine_with_y_flip(
            he_array,
            geojson_array,
            image_height_px=image_height_px,
            image_width_px=image_width_px,
            flip_candidates=flip_candidates,
            similarity_trim_quantile=similarity_trim,
            affine_trim_quantile=affine_trim,
        )
        fine_target_array = geojson_array
        fine_target_weights = np.ones(len(geojson_array), dtype=float)
        geojson_classification = np.full(len(geojson_array), "valid", dtype=object)
        boundary_anchor_points = None
        tissue_validity_table = None
        tissue_mask = None
        if he_image is not None:
            tissue_bounds_points = np.vstack([geojson_array, affine_result.transformed_points])
            tissue_min_x, tissue_min_y = np.min(tissue_bounds_points, axis=0) - 30.0
            tissue_max_x, tissue_max_y = np.max(tissue_bounds_points, axis=0) + 30.0
            affine_tissue_image, affine_tissue_metadata = warp_he_image_to_world(
                he_image,
                affine_result,
                None,
                output_pixel_size_um=warped_he_pixel_size,
                bounds=(float(tissue_min_x), float(tissue_min_y), float(tissue_max_x), float(tissue_max_y)),
                output_origin=warped_he_output_origin,
            )
            tissue_mask = _tissue_mask_from_image(affine_tissue_image, tissue_mask_threshold)
            valid_geojson_mask, geojson_weights, geojson_classification, tissue_validity_table = _point_tissue_validity(
                geojson_array,
                affine_tissue_metadata,
                tissue_mask,
                edge_margin=edge_margin,
                moving_points=affine_result.transformed_points,
                max_nearest_he_distance=cluster_patch_radius + cluster_search_radius,
            )
            fine_target_array = geojson_array[valid_geojson_mask]
            fine_target_weights = geojson_weights[valid_geojson_mask]
            if enable_boundary_pinning:
                boundary_anchor_points = _boundary_anchor_points_from_tissue_mask(
                    tissue_mask,
                    affine_tissue_metadata,
                    spacing_px=boundary_anchor_spacing,
                )
            if len(fine_target_array) < 3:
                st.warning(
                    "Fewer than 3 GeoJSON points are valid inside the affine HE tissue mask. "
                    "Fine warp will likely be rejected."
                )
        if len(fine_target_array) == 0:
            fine_result = _identity_fine_result(
                affine_result.transformed_points,
                geojson_array,
                grid_spacing,
                "No GeoJSON points are valid inside the affine HE tissue mask; fine warp was not applied.",
            )
        elif fine_alignment_method == "cluster-anchor":
            fine_result = cluster_anchor_fine_warp(
                fine_target_array,
                affine_result.transformed_points,
                fixed_point_weights=fine_target_weights,
                boundary_anchor_points=boundary_anchor_points,
                boundary_anchor_weight=boundary_anchor_weight,
                grid_spacing=control_grid_spacing if cluster_interpolation == "b-spline" else cluster_grid_spacing,
                patch_radius=cluster_patch_radius,
                search_radius=cluster_search_radius,
                search_step=cluster_search_step,
                min_points_per_cluster=int(min_points_per_cluster),
                match_threshold=cluster_match_threshold,
                min_improvement=cluster_min_improvement,
                max_shift=cluster_max_shift,
                outlier_percentile=local_outlier_percentile,
                neighbor_consistency_radius=local_neighbor_radius,
                smoothing=local_smoothing,
                kernel=local_kernel,
                neighbors=int(local_neighbors) if local_neighbors > 0 else 0,
                interpolation_method=cluster_interpolation,
                regularization=cluster_regularization,
                local_support_radius=local_support_radius,
                min_accepted_anchors=int(cluster_min_anchors),
                jacobian_max_threshold=jacobian_max_threshold,
            )
        elif fine_alignment_method == "matched nuclei RBF":
            fine_result = matched_nuclei_rbf_fine_warp(
                affine_result.transformed_points,
                fine_target_array,
                match_radius=match_radius,
                grid_spacing=grid_spacing,
                coherence_radius=coherence_radius,
                max_local_deviation=max_local_deviation,
                min_pair_confidence=min_pair_confidence,
                max_displacement=max_snap_displacement,
                smoothing=local_smoothing,
                kernel=local_kernel,
                neighbors=int(local_neighbors) if local_neighbors > 0 else 0,
                min_pairs=int(min_matched_anchor_pairs),
            )
        elif fine_alignment_method == "local translation field":
            fine_result = local_translation_fine_warp(
                fine_target_array,
                affine_result.transformed_points,
                density_sigma=local_density_sigma,
                density_pixel_size=local_density_pixel_size,
                point_weight=local_point_weight,
                grid_spacing=local_grid_spacing,
                patch_radius=local_patch_radius,
                search_radius=local_search_radius,
                min_correlation=local_min_correlation,
                max_shift=local_max_shift,
                outlier_percentile=local_outlier_percentile,
                neighbor_consistency_radius=local_neighbor_radius,
                smoothing=local_smoothing,
                kernel=local_kernel,
                neighbors=int(local_neighbors) if local_neighbors > 0 else 0,
                min_accepted_anchors=int(local_min_anchors),
            )
        elif fine_alignment_method == "center-snap":
            fine_result = fine_center_snap_warp(
                affine_result.transformed_points,
                fine_target_array,
                match_radius=match_radius,
                grid_spacing=grid_spacing,
                bandwidth=fine_bandwidth,
                ridge=ridge,
                coherence_radius=coherence_radius,
                max_local_deviation=max_local_deviation,
                min_pair_confidence=min_pair_confidence,
                snap_strength=snap_strength,
                max_snap_displacement=max_snap_displacement,
            )
        else:
            min_x, min_y = np.min(np.vstack([geojson_array, affine_result.transformed_points]), axis=0) - 30.0
            max_x, max_y = np.max(np.vstack([geojson_array, affine_result.transformed_points]), axis=0) + 30.0
            xs = np.arange(min_x, max_x + grid_spacing, grid_spacing)
            ys = np.arange(min_y, max_y + grid_spacing, grid_spacing)
            grid_x, grid_y = np.meshgrid(xs, ys)
            zeros = np.zeros_like(grid_x, dtype=float)
            affine_metrics = {"before": {}, "after": {}}
            fine_result = FineWarpResult(
                transformed_points=affine_result.transformed_points.copy(),
                grid_x=grid_x,
                grid_y=grid_y,
                displacement_x=zeros,
                displacement_y=zeros,
                attempted_transformed_points=affine_result.transformed_points.copy(),
                attempted_displacement_x=zeros,
                attempted_displacement_y=zeros,
                bounds=(float(min_x), float(min_y), float(max_x), float(max_y)),
                grid_spacing=float(grid_spacing),
                jacobian_min=1.0,
                jacobian_max=1.0,
                max_displacement=0.0,
                n_candidate_pairs=0,
                n_pairs=0,
                n_filtered_pairs=0,
                median_pair_distance_before=affine_result.median_residual,
                median_pair_distance_after=affine_result.median_residual,
                success=True,
                message="Fine alignment disabled.",
                attempted_metrics=None,
                applied_metrics=None,
                rejection_reason="Fine alignment disabled.",
                applied=False,
                anchors=None,
                metrics=affine_metrics,
            )
    except ValueError as exc:
        st.warning(str(exc))
        return

    if not fine_result.success:
        st.warning(fine_result.message)
    if fine_result.jacobian_min <= 0:
        st.warning("Jacobian minimum is <= 0. The fine warp may contain local fold-over.")

    transformed_affine_points = he_points.copy()
    transformed_affine_points["centroid_x"] = affine_result.transformed_points[:, 0]
    transformed_affine_points["centroid_y"] = affine_result.transformed_points[:, 1]
    transformed_affine_points["source"] = "he_affine_world_um"

    transformed_fine_points = he_points.copy()
    transformed_fine_points["centroid_x"] = fine_result.transformed_points[:, 0]
    transformed_fine_points["centroid_y"] = fine_result.transformed_points[:, 1]
    transformed_fine_points["source"] = "he_fine_world_um"

    attempted_points = (
        fine_result.attempted_transformed_points
        if getattr(fine_result, "attempted_transformed_points", None) is not None
        else fine_result.transformed_points
    )
    attempted_displacement_x = (
        fine_result.attempted_displacement_x
        if getattr(fine_result, "attempted_displacement_x", None) is not None
        else fine_result.displacement_x
    )
    attempted_displacement_y = (
        fine_result.attempted_displacement_y
        if getattr(fine_result, "attempted_displacement_y", None) is not None
        else fine_result.displacement_y
    )
    attempted_metrics_result = getattr(fine_result, "attempted_metrics", None)
    applied_metrics_result = getattr(fine_result, "applied_metrics", None)
    rejection_reason = getattr(fine_result, "rejection_reason", None) or ""
    fine_applied_value = getattr(fine_result, "applied", None)
    fine_applied = fine_result.success if fine_applied_value is None else bool(fine_applied_value)

    transformed_attempted_points = he_points.copy()
    transformed_attempted_points["centroid_x"] = attempted_points[:, 0]
    transformed_attempted_points["centroid_y"] = attempted_points[:, 1]
    transformed_attempted_points["source"] = "he_attempted_fine_world_um"

    st.subheader("Coordinate diagnostics")
    st.caption(
        "Use this table to separate raw point ranges, registration output ranges, and possible GeoJSON ROI/range mismatch."
    )
    coordinate_diagnostics = pd.DataFrame(
        [
            _coordinate_summary_row("HE raw nuclei points", he_array),
            _coordinate_summary_row("GeoJSON centroids", geojson_array),
            _coordinate_summary_row("GeoJSON valid fine-warp targets", fine_target_array),
            _coordinate_summary_row("HE affine world points", affine_result.transformed_points),
            _coordinate_summary_row("HE fine world points", fine_result.transformed_points),
        ]
    )
    st.dataframe(coordinate_diagnostics, use_container_width=True)
    if tissue_validity_table is not None:
        st.subheader("Tissue-validity filtering")
        validity_summary = (
            tissue_validity_table["exclusion_reason"]
            .value_counts(dropna=False)
            .rename_axis("status")
            .reset_index(name="n_points")
        )
        st.dataframe(validity_summary, use_container_width=True)
        st.download_button(
            "Download GeoJSON tissue-validity CSV",
            data=tissue_validity_table.to_csv(index=False).encode("utf-8"),
            file_name="geojson_tissue_validity.csv",
            mime="text/csv",
        )
        if affine_tissue_image is not None and affine_tissue_metadata is not None:
            tissue_col, class_col = st.columns(2)
            with tissue_col:
                st.image(
                    tissue_mask.astype(np.uint8) * 255,
                    caption="Tissue mask preview from affine-only warped HE image",
                )
            with class_col:
                classification_pixels = world_points_to_warped_image_pixels(geojson_array, affine_tissue_metadata)
                classification_figure = visualize_geojson_classification_overlay(
                    affine_tissue_image,
                    classification_pixels,
                    geojson_classification,
                    title="GeoJSON tissue-aware classification",
                    max_points=max_warped_overlay_points,
                )
                st.pyplot(classification_figure, clear_figure=False)
                st.download_button(
                    "Download GeoJSON classification overlay PNG",
                    data=figure_to_png_bytes(classification_figure),
                    file_name="geojson_tissue_classification_overlay.png",
                    mime="image/png",
                )

    st.subheader("Registration QC")
    metric_a, metric_b, metric_c, metric_d, metric_e = st.columns(5)
    metric_a.metric("X-flip", str(affine_result.flip_x))
    metric_b.metric("Y-flip", str(affine_result.flip_y))
    metric_c.metric("Affine median", f"{affine_result.median_residual:.2f} um")
    metric_d.metric("Fine median", f"{fine_result.median_pair_distance_after:.2f} um")
    metric_e.metric("Jacobian min", f"{fine_result.jacobian_min:.3f}")
    st.caption(
        f"Fine snap pairs: {fine_result.n_pairs} used / {fine_result.n_candidate_pairs} candidates "
        f"({fine_result.n_filtered_pairs} filtered out)"
    )

    metric_fixed_points = fine_target_array if len(fine_target_array) else geojson_array
    before_metrics = point_distance_metrics(metric_fixed_points, affine_result.transformed_points)
    attempted_metrics = attempted_metrics_result or point_distance_metrics(
        metric_fixed_points,
        attempted_points,
    )
    applied_metrics = applied_metrics_result or point_distance_metrics(metric_fixed_points, fine_result.transformed_points)
    status = "applied" if fine_applied else ("disabled" if fine_alignment_method == "off" else "rejected")
    accepted_anchor_count = fine_result.n_pairs
    total_anchor_count = fine_result.n_candidate_pairs
    rejected_anchor_count = fine_result.n_filtered_pairs
    if fine_result.anchors is not None and "shift_magnitude" in fine_result.anchors:
        accepted_anchor_magnitudes = fine_result.anchors.loc[fine_result.anchors["accepted"], "shift_magnitude"]
        median_shift = float(accepted_anchor_magnitudes.median()) if not accepted_anchor_magnitudes.empty else 0.0
        p95_shift = float(accepted_anchor_magnitudes.quantile(0.95)) if not accepted_anchor_magnitudes.empty else 0.0
    else:
        attempted_magnitude = np.sqrt(attempted_displacement_x**2 + attempted_displacement_y**2)
        median_shift = float(np.median(attempted_magnitude))
        p95_shift = float(np.percentile(attempted_magnitude, 95))

    st.subheader("Fine alignment diagnostics")
    diag_a, diag_b, diag_c, diag_d = st.columns(4)
    diag_a.metric("Fine method", fine_alignment_method)
    diag_b.metric("Fine status", status)
    diag_c.metric("Accepted anchors", f"{accepted_anchor_count}/{total_anchor_count}")
    diag_d.metric("Rejected anchors", rejected_anchor_count)
    if rejection_reason:
        st.warning(f"Rejection reason: {rejection_reason}")
    diag2_a, diag2_b, diag2_c, diag2_d = st.columns(4)
    diag2_a.metric("Median shift", f"{median_shift:.2f} um")
    diag2_b.metric("Shift p95", f"{p95_shift:.2f} um")
    diag2_c.metric("Max displacement", f"{fine_result.max_displacement:.2f} um")
    diag2_d.metric("Jacobian min", f"{fine_result.jacobian_min:.3f}")

    def _metric_rows_for_group(group_name: str, fixed_subset: np.ndarray) -> list[dict]:
        if len(fixed_subset) == 0:
            return []
        rows = []
        for stage, moving_subset in [
            ("affine", affine_result.transformed_points),
            ("attempted_fine", attempted_points),
            ("applied_fine", fine_result.transformed_points),
        ]:
            values = point_distance_metrics(fixed_subset, moving_subset)
            rows.append(
                {
                    "point_group": group_name,
                    "stage": stage,
                    "median": values["median_distance"],
                    "mean": values["mean_distance"],
                    "within_3": values["within_3"],
                    "within_5": values["within_5"],
                    "within_10": values["within_10"],
                }
            )
        return rows

    metric_rows = []
    metric_rows.extend(_metric_rows_for_group("all_geojson", geojson_array))
    metric_rows.extend(_metric_rows_for_group("valid_geojson", geojson_array[geojson_classification == "valid"]))
    metric_rows.extend(
        _metric_rows_for_group("edge_candidate_geojson", geojson_array[geojson_classification == "edge_candidate"])
    )
    st.dataframe(pd.DataFrame(metric_rows), use_container_width=True)

    plot_left, plot_mid, plot_right = st.columns(3)
    geojson_features = _cell_features_from_points(geojson_points)
    affine_features = _cell_features_from_points(transformed_affine_points)
    attempted_features = _cell_features_from_points(transformed_attempted_points)
    fine_features = _cell_features_from_points(transformed_fine_points)
    invert_x_axis = registration_display_origin == "upper-right"
    invert_y_axis = registration_display_origin in {"upper-right", "upper-left"}
    with plot_left:
        affine_figure = visualize_point_sets(
            geojson_features,
            affine_features,
            title="Affine HE centers vs GeoJSON centroids (world-um)",
            invert_x_axis=invert_x_axis,
            invert_y_axis=invert_y_axis,
        )
        st.pyplot(affine_figure, clear_figure=False)
        st.download_button(
            "Download affine scatter PNG",
            data=figure_to_png_bytes(affine_figure),
            file_name="he_geojson_affine_scatter.png",
            mime="image/png",
        )
    with plot_mid:
        attempted_figure = visualize_point_sets(
            geojson_features,
            attempted_features,
            title=f"Attempted fine HE centers ({fine_alignment_method})",
            invert_x_axis=invert_x_axis,
            invert_y_axis=invert_y_axis,
        )
        st.pyplot(attempted_figure, clear_figure=False)
        st.download_button(
            "Download attempted fine scatter PNG",
            data=figure_to_png_bytes(attempted_figure),
            file_name="he_geojson_attempted_fine_scatter.png",
            mime="image/png",
        )
    with plot_right:
        fine_figure = visualize_point_sets(
            geojson_features,
            fine_features,
            title=f"Applied fine HE centers ({fine_alignment_method})",
            invert_x_axis=invert_x_axis,
            invert_y_axis=invert_y_axis,
        )
        st.pyplot(fine_figure, clear_figure=False)
        st.download_button(
            "Download fine scatter PNG",
            data=figure_to_png_bytes(fine_figure),
            file_name="he_geojson_fine_scatter.png",
            mime="image/png",
        )

    before_distances = point_nearest_distances(metric_fixed_points, affine_result.transformed_points)
    attempted_distances = point_nearest_distances(metric_fixed_points, attempted_points)
    after_distances = point_nearest_distances(metric_fixed_points, fine_result.transformed_points)
    hist_figure = visualize_distance_histogram(
        before_distances,
        after_distances,
        title="Nearest-neighbor distance before/after fine alignment",
        attempted_distances=attempted_distances,
    )
    st.pyplot(hist_figure, clear_figure=False)
    st.download_button(
        "Download distance histogram PNG",
        data=figure_to_png_bytes(hist_figure),
        file_name="fine_alignment_distance_histogram.png",
        mime="image/png",
    )

    residual_left, residual_right = st.columns(2)
    with residual_left:
        attempted_residual_figure = visualize_local_residual_map(
            geojson_array,
            attempted_points,
            title="Attempted fine local residual map",
            max_points=max_warped_overlay_points,
        )
        st.pyplot(attempted_residual_figure, clear_figure=False)
        st.download_button(
            "Download attempted residual map PNG",
            data=figure_to_png_bytes(attempted_residual_figure),
            file_name="attempted_fine_residual_map.png",
            mime="image/png",
        )
    with residual_right:
        applied_residual_figure = visualize_local_residual_map(
            geojson_array,
            fine_result.transformed_points,
            title="Applied fine local residual map",
            max_points=max_warped_overlay_points,
        )
        st.pyplot(applied_residual_figure, clear_figure=False)
        st.download_button(
            "Download applied residual map PNG",
            data=figure_to_png_bytes(applied_residual_figure),
            file_name="applied_fine_residual_map.png",
            mime="image/png",
        )

    if fine_result.anchors is not None:
        st.subheader("Fine warp displacement anchors")
        st.dataframe(fine_result.anchors, use_container_width=True)
        anchor_filename = (
            "cluster_anchor_candidates.csv"
            if fine_alignment_method == "cluster-anchor"
            else "fine_warp_displacement_anchors.csv"
        )
        st.download_button(
            "Download fine warp anchors CSV",
            data=fine_result.anchors.to_csv(index=False).encode("utf-8"),
            file_name=anchor_filename,
            mime="text/csv",
        )

        anchor_left, anchor_right = st.columns(2)
        with anchor_left:
            anchor_figure = visualize_translation_anchors(
                fine_result.anchors,
                title="Accepted and rejected fine warp anchors",
            )
            st.pyplot(anchor_figure, clear_figure=False)
            st.download_button(
                "Download anchor vectors PNG",
                data=figure_to_png_bytes(anchor_figure),
                file_name="fine_warp_anchor_vectors.png",
                mime="image/png",
            )
        with anchor_right:
            corr_figure = visualize_anchor_correlation_heatmap(
                fine_result.anchors,
                title="Fine warp anchor correlation",
            )
            st.pyplot(corr_figure, clear_figure=False)
            st.download_button(
                "Download correlation heatmap PNG",
                data=figure_to_png_bytes(corr_figure),
                file_name="fine_warp_correlation_heatmap.png",
                mime="image/png",
            )

    field_left, field_right = st.columns(2)
    with field_left:
        attempted_field_figure = visualize_displacement_field(
            fine_result.grid_x,
            fine_result.grid_y,
            attempted_displacement_x,
            attempted_displacement_y,
            title="Attempted fine displacement field",
        )
        st.pyplot(attempted_field_figure, clear_figure=False)
        st.download_button(
            "Download attempted displacement field PNG",
            data=figure_to_png_bytes(attempted_field_figure),
            file_name="attempted_fine_displacement_vector_field.png",
            mime="image/png",
        )
    with field_right:
        applied_field_figure = visualize_displacement_field(
            fine_result.grid_x,
            fine_result.grid_y,
            fine_result.displacement_x,
            fine_result.displacement_y,
            title="Applied fine displacement field",
        )
        st.pyplot(applied_field_figure, clear_figure=False)
        st.download_button(
            "Download applied displacement field PNG",
            data=figure_to_png_bytes(applied_field_figure),
            file_name="applied_fine_displacement_vector_field.png",
            mime="image/png",
        )

    attempted_jacobian = _field_jacobian(attempted_displacement_x, attempted_displacement_y, fine_result.grid_spacing)
    applied_jacobian = _field_jacobian(fine_result.displacement_x, fine_result.displacement_y, fine_result.grid_spacing)
    st.subheader("Jacobian QC")
    st.dataframe(
        pd.DataFrame(
            [
                _jacobian_summary_row("attempted_fine", attempted_jacobian),
                _jacobian_summary_row("applied_fine", applied_jacobian),
            ]
        ),
        use_container_width=True,
    )
    jac_left, jac_right = st.columns(2)
    with jac_left:
        attempted_jacobian_figure = visualize_jacobian_heatmap(
            fine_result.grid_x,
            fine_result.grid_y,
            attempted_jacobian,
            title="Attempted fine Jacobian heatmap",
        )
        st.pyplot(attempted_jacobian_figure, clear_figure=False)
        st.download_button(
            "Download attempted Jacobian heatmap PNG",
            data=figure_to_png_bytes(attempted_jacobian_figure),
            file_name="attempted_fine_jacobian_heatmap.png",
            mime="image/png",
        )
    with jac_right:
        applied_jacobian_figure = visualize_jacobian_heatmap(
            fine_result.grid_x,
            fine_result.grid_y,
            applied_jacobian,
            title="Applied fine Jacobian heatmap",
        )
        st.pyplot(applied_jacobian_figure, clear_figure=False)
        st.download_button(
            "Download applied Jacobian heatmap PNG",
            data=figure_to_png_bytes(applied_jacobian_figure),
            file_name="applied_fine_jacobian_heatmap.png",
            mime="image/png",
        )

    warped_he_image = None
    warped_he_metadata = None
    attempted_warped_he_image = None
    attempted_warped_he_metadata = None
    if he_image is not None:
        try:
            warped_he_image, warped_he_metadata = warp_he_image_to_world(
                he_image,
                affine_result,
                fine_result,
                output_pixel_size_um=warped_he_pixel_size,
                output_origin=warped_he_output_origin,
            )
        except ValueError as exc:
            st.warning(f"Could not warp HE image: {exc}")
        if not fine_applied and getattr(fine_result, "attempted_displacement_x", None) is not None:
            try:
                attempted_preview_result = replace(
                    fine_result,
                    transformed_points=attempted_points,
                    displacement_x=attempted_displacement_x,
                    displacement_y=attempted_displacement_y,
                    success=True,
                    applied=True,
                )
                attempted_warped_he_image, attempted_warped_he_metadata = warp_he_image_to_world(
                    he_image,
                    affine_result,
                    attempted_preview_result,
                    output_pixel_size_um=warped_he_pixel_size,
                    output_origin=warped_he_output_origin,
                )
            except ValueError as exc:
                st.warning(f"Could not render attempted warp preview: {exc}")

    if warped_he_image is not None:
        st.subheader("Warped HE image")
        st.info("Warped HE output origin controls exported image orientation, not registration quality.")
        if fine_applied:
            st.success("Warped HE image uses: affine + applied fine warp")
        else:
            st.warning("Fine warp was rejected or disabled; warped HE image is affine-only.")
            if attempted_warped_he_image is not None:
                st.info("Attempted warp preview shows the rejected candidate deformation for QC only.")
        warped_image_height, warped_image_width = warped_he_image.shape[:2]
        geojson_pixels = world_points_to_warped_image_pixels(geojson_array, warped_he_metadata)
        he_pixels = world_points_to_warped_image_pixels(fine_result.transformed_points, warped_he_metadata)

        st.subheader("Warped pixel-coordinate diagnostics")
        pixel_diagnostics = pd.DataFrame(
            [
                _pixel_coordinate_summary_row(
                    "GeoJSON pixels",
                    geojson_pixels,
                    image_width=warped_image_width,
                    image_height=warped_image_height,
                ),
                _pixel_coordinate_summary_row(
                    "Registered HE pixels",
                    he_pixels,
                    image_width=warped_image_width,
                    image_height=warped_image_height,
                ),
            ]
        )
        st.dataframe(pixel_diagnostics, use_container_width=True)

        pixel_scatter_figure = visualize_warped_pixel_point_scatter(
            geojson_pixels,
            he_pixels,
            image_width=warped_image_width,
            image_height=warped_image_height,
            title="Warped pixel coordinates without HE image background",
            max_points=max_warped_overlay_points,
        )
        st.pyplot(pixel_scatter_figure, clear_figure=False)
        st.download_button(
            "Download warped pixel scatter PNG",
            data=figure_to_png_bytes(pixel_scatter_figure),
            file_name="warped_pixel_points_scatter.png",
            mime="image/png",
        )

        image_only_col, overlay_col = st.columns(2)
        with image_only_col:
            st.image(
                warped_he_image,
                caption=(
                    "Warped HE image without points "
                    f"| pixel_size={warped_he_metadata['output_pixel_size_um']} um"
                ),
            )
        st.download_button(
            "Download warped HE image PNG",
            data=array_to_png_bytes(warped_he_image),
            file_name="warped_he_world_um.png",
            mime="image/png",
        )
        if attempted_warped_he_image is not None:
            attempted_geojson_pixels = world_points_to_warped_image_pixels(geojson_array, attempted_warped_he_metadata)
            attempted_he_pixels = world_points_to_warped_image_pixels(attempted_points, attempted_warped_he_metadata)
            st.image(
                attempted_warped_he_image,
                caption="Attempted HE warp preview from rejected candidate field",
            )
            st.download_button(
                "Download attempted warped HE preview PNG",
                data=array_to_png_bytes(attempted_warped_he_image),
                file_name="attempted_warped_he_preview.png",
                mime="image/png",
            )
            attempted_overlay_figure = visualize_warped_he_point_overlay(
                attempted_warped_he_image,
                attempted_geojson_pixels,
                attempted_he_pixels,
                title="Attempted HE warp preview with GeoJSON and HE nuclei overlay",
                max_points=max_warped_overlay_points,
            )
            st.pyplot(attempted_overlay_figure, clear_figure=False)
            st.download_button(
                "Download attempted warped HE overlay PNG",
                data=figure_to_png_bytes(attempted_overlay_figure),
                file_name="attempted_warped_he_points_overlay.png",
                mime="image/png",
            )

            before_grid_lines = _warp_grid_lines_pixels(
                fine_result.bounds,
                max(cluster_grid_spacing, 1.0),
                attempted_warped_he_metadata,
                fine_result=None,
            )
            attempted_grid_result = replace(
                fine_result,
                displacement_x=attempted_displacement_x,
                displacement_y=attempted_displacement_y,
            )
            attempted_grid_lines = _warp_grid_lines_pixels(
                fine_result.bounds,
                max(cluster_grid_spacing, 1.0),
                attempted_warped_he_metadata,
                fine_result=attempted_grid_result,
            )
            attempted_grid_figure = visualize_warp_grid_overlay(
                attempted_warped_he_image,
                before_grid_lines,
                attempted_grid_lines,
                title="Attempted warp grid QC: cyan before / orange after",
            )
            st.pyplot(attempted_grid_figure, clear_figure=False)
            st.download_button(
                "Download attempted warp grid QC PNG",
                data=figure_to_png_bytes(attempted_grid_figure),
                file_name="attempted_warp_grid_qc.png",
                mime="image/png",
            )

        with overlay_col:
            warped_overlay_figure = visualize_warped_he_point_overlay(
                warped_he_image,
                geojson_pixels,
                he_pixels,
                title="Warped HE image with registered nuclei overlay",
                max_points=max_warped_overlay_points,
            )
            st.pyplot(warped_overlay_figure, clear_figure=False)
        st.download_button(
            "Download warped HE overlay PNG",
            data=figure_to_png_bytes(warped_overlay_figure),
            file_name="warped_he_registered_points_overlay.png",
            mime="image/png",
        )
        applied_before_grid_lines = _warp_grid_lines_pixels(
            fine_result.bounds,
            max(cluster_grid_spacing, 1.0),
            warped_he_metadata,
            fine_result=None,
        )
        applied_after_grid_lines = _warp_grid_lines_pixels(
            fine_result.bounds,
            max(cluster_grid_spacing, 1.0),
            warped_he_metadata,
            fine_result=fine_result,
        )
        applied_grid_figure = visualize_warp_grid_overlay(
            warped_he_image,
            applied_before_grid_lines,
            applied_after_grid_lines,
            title="Applied warp grid QC: cyan before / orange after",
        )
        st.pyplot(applied_grid_figure, clear_figure=False)
        st.download_button(
            "Download applied warp grid QC PNG",
            data=figure_to_png_bytes(applied_grid_figure),
            file_name="applied_warp_grid_qc.png",
            mime="image/png",
        )

    st.subheader("Exports")
    st.download_button(
        "Download transformed HE centers CSV",
        data=transformed_fine_points.to_csv(index=False).encode("utf-8"),
        file_name="he_centers_transformed_world_um.csv",
        mime="text/csv",
    )
    parameters = {
        "fine_alignment_method": fine_alignment_method,
        "local_translation_preset": local_preset,
        "he_coordinate_order": he_coordinate_order,
        "similarity_trim_quantile": similarity_trim,
        "affine_trim_quantile": affine_trim,
        "fine_match_radius_um": match_radius,
        "fine_grid_spacing_um": grid_spacing,
        "fine_bandwidth_um": fine_bandwidth,
        "fine_ridge": ridge,
        "fine_min_pair_confidence": min_pair_confidence,
        "fine_coherence_radius_um": coherence_radius,
        "fine_max_local_deviation_um": max_local_deviation,
        "fine_snap_strength": snap_strength,
        "fine_max_snap_displacement_um": max_snap_displacement,
        "fine_min_matched_anchor_pairs": min_matched_anchor_pairs,
        "local_density_sigma_um": local_density_sigma,
        "local_density_pixel_size_um": local_density_pixel_size,
        "local_point_weight": local_point_weight,
        "local_grid_spacing_um": local_grid_spacing,
        "local_patch_radius_um": local_patch_radius,
        "local_search_radius_um": local_search_radius,
        "local_min_correlation": local_min_correlation,
        "local_max_shift_um": local_max_shift,
        "local_outlier_percentile": local_outlier_percentile,
        "local_neighbor_consistency_radius_um": local_neighbor_radius,
        "local_smoothing": local_smoothing,
        "local_kernel": local_kernel,
        "local_neighbors": local_neighbors,
        "local_min_accepted_anchors": local_min_anchors,
        "cluster_grid_spacing_um": cluster_grid_spacing,
        "cluster_patch_radius_um": cluster_patch_radius,
        "cluster_search_radius_um": cluster_search_radius,
        "cluster_search_step_um": cluster_search_step,
        "cluster_min_points_per_cluster": min_points_per_cluster,
        "cluster_match_threshold_um": cluster_match_threshold,
        "cluster_min_improvement_um": cluster_min_improvement,
        "cluster_max_shift_um": cluster_max_shift,
        "cluster_min_accepted_anchors": cluster_min_anchors,
        "cluster_interpolation": cluster_interpolation,
        "local_support_radius_um": local_support_radius,
        "control_grid_spacing_um": control_grid_spacing,
        "cluster_regularization": cluster_regularization,
        "tissue_mask_threshold": tissue_mask_threshold,
        "edge_margin_px": edge_margin,
        "enable_boundary_pinning": enable_boundary_pinning,
        "boundary_anchor_spacing_px": boundary_anchor_spacing,
        "boundary_anchor_weight": boundary_anchor_weight,
        "jacobian_max_threshold": jacobian_max_threshold,
        "registration_display_origin": registration_display_origin,
        "warped_he_output_origin": warped_he_output_origin,
        "flip_mode": flip_mode,
        "warped_he_output_pixel_size_um": warped_he_pixel_size,
        "max_warped_overlay_points": max_warped_overlay_points,
    }
    st.download_button(
        "Download HE-GeoJSON transform summary",
        data=_he_geojson_summary_to_json(affine_result, fine_result, parameters, warped_he_metadata),
        file_name="he_geojson_transform_summary.json",
        mime="application/json",
    )

    # TODO: Add inverse-warp refinement controls and larger tiled exports for full-resolution HE images.
    # TODO: Add GeoJSON polygon overlay and warp-field vector QC panels.


def main() -> None:
    st.title("Cell Registration Prototype")
    st.caption("Research prototype for point-based registration, matching, and QC. Not for diagnostic use.")

    workflow = st.sidebar.selectbox(
        "Workflow",
        [
            "Workflow A: Point registration",
            "Workflow B: Mask-derived point registration",
            "Workflow C: HE-GeoJSON alignment",
        ],
    )
    st.sidebar.divider()
    density_sigma = st.sidebar.slider("Density map sigma", min_value=1.0, max_value=100.0, value=10.0, step=1.0)
    st.sidebar.divider()
    st.sidebar.header("Matching thresholds")
    max_distance = st.sidebar.number_input("Max distance", min_value=0.1, value=50.0, step=1.0)
    min_area_ratio = st.sidebar.number_input("Min area ratio", min_value=0.01, value=0.5, step=0.05)
    max_area_ratio = st.sidebar.number_input("Max area ratio", min_value=0.01, value=2.0, step=0.05)
    max_score = st.sidebar.number_input("Max score", min_value=0.0, value=1.5, step=0.1)
    st.sidebar.divider()
    st.sidebar.header("Visualization")
    max_pairs_to_display = st.sidebar.number_input(
        "Max pairs to display",
        min_value=1,
        max_value=5000,
        value=200,
        step=50,
    )

    workflow_kwargs = {
        "density_sigma": density_sigma,
        "max_distance": max_distance,
        "min_area_ratio": min_area_ratio,
        "max_area_ratio": max_area_ratio,
        "max_score": max_score,
        "max_pairs_to_display": max_pairs_to_display,
    }
    if workflow.startswith("Workflow A"):
        show_point_registration_workflow(**workflow_kwargs)
    elif workflow.startswith("Workflow B"):
        show_mask_to_mask_workflow(**workflow_kwargs)
    else:
        show_he_geojson_preparation()


if __name__ == "__main__":
    main()
