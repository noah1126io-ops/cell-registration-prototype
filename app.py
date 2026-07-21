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
from src.density_flow import (
    density_flow_deformation_diagnostics,
    density_flow_image_outputs,
    tissue_aware_density_flow_registration,
)
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
    point_bidirectional_distance_metrics,
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
    visualize_absolute_image_difference,
    visualize_displacement_field,
    visualize_displacement_magnitude_heatmap,
    visualize_density_flow_point_comparison,
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


def _workflow_c_result_variants(affine_points: np.ndarray, fine_result: FineWarpResult) -> dict:
    """Separate attempted diagnostics from the result that is safe to apply."""
    affine_points = np.asarray(affine_points, dtype=float)
    fine_applied = bool(getattr(fine_result, "applied", False))
    attempted_points_value = getattr(fine_result, "attempted_transformed_points", None)
    attempted_dx_value = getattr(fine_result, "attempted_displacement_x", None)
    attempted_dy_value = getattr(fine_result, "attempted_displacement_y", None)
    zero_dx = np.zeros_like(fine_result.displacement_x, dtype=float)
    zero_dy = np.zeros_like(fine_result.displacement_y, dtype=float)
    attempted_points = (
        np.asarray(attempted_points_value, dtype=float)
        if attempted_points_value is not None
        else affine_points.copy()
    )
    attempted_dx = np.asarray(attempted_dx_value, dtype=float) if attempted_dx_value is not None else zero_dx.copy()
    attempted_dy = np.asarray(attempted_dy_value, dtype=float) if attempted_dy_value is not None else zero_dy.copy()
    if fine_applied:
        final_points = np.asarray(fine_result.transformed_points, dtype=float)
        final_dx = np.asarray(fine_result.displacement_x, dtype=float)
        final_dy = np.asarray(fine_result.displacement_y, dtype=float)
        applied_result_label = "Affine + fine warp"
    else:
        final_points = affine_points.copy()
        final_dx = zero_dx.copy()
        final_dy = zero_dy.copy()
        applied_result_label = "Affine only"
    return {
        "fine_applied": fine_applied,
        "attempted_points": attempted_points,
        "attempted_displacement_x": attempted_dx,
        "attempted_displacement_y": attempted_dy,
        "final_points": final_points,
        "final_displacement_x": final_dx,
        "final_displacement_y": final_dy,
        "applied_result_label": applied_result_label,
    }


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


def _count_points_inside_tissue(points: np.ndarray, warp_metadata: dict | None, tissue_mask: np.ndarray | None) -> tuple[int, int]:
    if warp_metadata is None or tissue_mask is None:
        return int(len(points)), 0
    pixels = world_points_to_warped_image_pixels(points, warp_metadata)
    height, width = tissue_mask.shape[:2]
    cols = pixels[:, 0]
    rows = pixels[:, 1]
    inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    valid = np.zeros(len(points), dtype=bool)
    if len(points):
        row_i = np.clip(np.rint(rows).astype(int), 0, max(height - 1, 0))
        col_i = np.clip(np.rint(cols).astype(int), 0, max(width - 1, 0))
        valid[inside] = tissue_mask[row_i[inside], col_i[inside]]
    n_valid = int(np.count_nonzero(valid))
    return n_valid, int(len(points) - n_valid)


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
    valid_weight: float,
    edge_candidate_weight: float,
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
    weights[valid] = float(valid_weight)
    weights[edge_candidate] = float(edge_candidate_weight)

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
    include_tissue_boundary: bool = True,
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

    if include_tissue_boundary and np.any(tissue_mask):
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
    language = st.selectbox(
        "Language / 言語",
        ["日本語", "English"],
        key="workflow-c-language",
        help="Workflow Cの表示言語だけを変更します。計算結果には影響しません。",
    )
    is_ja = language == "日本語"

    def tr(ja: str, en: str) -> str:
        return ja if is_ja else en

    method_labels = {
        "tissue-aware density flow": tr(
            "組織考慮density flow［実験的］",
            "Tissue-aware density flow [Experimental]",
        ),
        "cluster-anchor": tr("クラスターアンカー", "Cluster-anchor"),
        "matched nuclei RBF": tr("対応核RBF", "Matched nuclei RBF"),
        "local translation field": tr("局所平行移動場", "Local translation field"),
        "center-snap": tr("中心スナップ", "Center-snap"),
        "off": tr("なし（アフィンのみ）", "Off (affine only)"),
    }

    st.header(tr("Workflow C: HE-GeoJSON位置合わせ", "Workflow C: HE-GeoJSON alignment"))
    st.caption(
        tr(
            "HE核点群を蛍光GeoJSONのworld-um座標へ合わせる特殊座標系ワークフローです。",
            "Align HE nuclei points to fluorescence GeoJSON in world-um coordinates.",
        )
    )
    st.subheader(tr("1. 入力ファイル", "1. Input files"))

    input_left, input_right = st.columns(2)
    with input_left:
        he_centers_file = st.file_uploader(
            tr("HE核中心 (.npy)", "HE nuclei centers (.npy)"),
            type=["npy"],
            key="he-nuclei-npy",
            help=tr("shape (n, 2) の事前計算済み核中心です。", "Precomputed nuclei centers with shape (n, 2)."),
        )
        he_coordinate_order = st.selectbox(
            tr("HE .npyの座標順", "HE .npy coordinate order"),
            ["xy", "yx"],
            key="he-nuclei-order",
            help=tr("1列目と2列目がx,yかy,xかを指定します。", "Specify whether columns are x,y or y,x."),
        )
        he_image_file = st.file_uploader(
            tr("任意のHE画像（Y反転・QC背景用）", "Optional HE image for Y-flip height / QC background"),
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="he-qc-image",
            help=tr("位置合わせの主入力ではなく、座標変換と目視QCに使います。", "Used for coordinate conversion and visual QC, not as the primary registration input."),
        )
    with input_right:
        geojson_file = st.file_uploader(
            tr("蛍光核GeoJSON", "Fluorescence nuclei GeoJSON"),
            type=["geojson", "json"],
            key="fluorescence-geojson",
            help=tr("固定側の核重心または核ポリゴンを含むGeoJSONです。", "GeoJSON containing fixed nuclei centroids or polygons."),
        )

    local_preset = st.session_state.get("workflow-c-local-preset", "balanced")
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
    st.subheader(tr("2. 粗い位置合わせ", "2. Coarse alignment"))
    coarse_left, coarse_mid, coarse_right = st.columns(3)
    with coarse_left:
        flip_mode = st.selectbox(
            tr("HE座標の反転候補", "HE coordinate flip candidates"),
            ["auto", "none", "x", "y", "x+y"],
            index=0,
            key="workflow-c-flip-mode",
            help=tr("autoは反転候補を比較して最良の粗位置合わせを選びます。", "Auto compares flip candidates and selects the best coarse alignment."),
        )
    with coarse_mid:
        similarity_trim = st.slider(
            tr("Similarity ICPの採用分位", "Similarity ICP trim quantile"), 0.1, 1.0, 0.8, 0.05,
            key="workflow-c-similarity-trim", help=tr("初期ICPで残す近い対応の割合です。", "Fraction of nearest pairs retained during similarity ICP."),
        )
    with coarse_right:
        affine_trim = st.slider(
            tr("Affine ICPの採用分位", "Affine ICP trim quantile"), 0.1, 1.0, 0.7, 0.05,
            key="workflow-c-affine-trim", help=tr("アフィンICPで残す近い対応の割合です。", "Fraction of nearest pairs retained during affine ICP."),
        )

    # Hidden methods keep the exact defaults from the original UI.
    match_radius, fine_bandwidth, grid_spacing, ridge = 10.0, 12.0, 6.0, 0.3
    min_pair_confidence, coherence_radius, max_local_deviation = 0.05, 30.0, 8.0
    snap_strength, max_snap_displacement, min_matched_anchor_pairs = 1.25, 25.0, 6
    local_density_sigma, local_density_pixel_size, local_point_weight = local_default["density_sigma"], 1.0, 1.0
    local_grid_spacing, local_patch_radius = local_default["grid_spacing"], local_default["patch_radius"]
    local_search_radius, local_min_correlation = local_default["search_radius"], local_default["min_correlation"]
    local_max_shift, local_min_anchors = local_default["max_shift"], local_default["min_accepted_anchors"]
    local_outlier_percentile, local_neighbor_radius = 95.0, 120.0
    local_smoothing, local_kernel, local_neighbors = 10.0, "linear", 40
    density_flow_pixel_size = 2.0
    density_flow_blur_scales_text = "8, 4, 2"
    density_flow_levels = 3
    density_flow_iterations = 12
    density_flow_learning_rate = 0.2
    density_flow_update_smoothing = 3.0
    density_flow_smoothness_weight = 0.05
    density_flow_magnitude_weight = 0.0005
    density_flow_jacobian_weight = 1.0
    density_flow_boundary_weight = 0.02
    density_flow_inverse_weight = 0.0
    density_flow_detect_axis_reversal = True
    density_flow_global_initialization = "off"

    st.subheader(tr("3. 詳細位置合わせ", "3. Fine alignment"))
    fine_alignment_method = st.selectbox(
        tr("詳細位置合わせ方式", "Fine alignment method"),
        [
            "cluster-anchor",
            "tissue-aware density flow",
            "matched nuclei RBF",
            "local translation field",
            "center-snap",
            "off",
        ],
        index=0,
        key="workflow-c-fine-method",
        format_func=lambda method: method_labels[method],
        help=tr("選択した方式に関係する設定だけを表示します。GeoJSON固定点は移動しません。", "Only settings for the selected method are shown. Fixed GeoJSON points are never moved."),
    )

    if fine_alignment_method == "tissue-aware density flow":
        st.warning(
            tr(
                "実験的な独立実装です。HE画像warpも安全判定前後を分けて表示し、結果は必ずQCしてください。",
                "This is an independent experimental implementation. HE raster warps are shown separately before and after safety gating and require visual QC.",
            )
        )
        flow_left, flow_right = st.columns(2)
        with flow_left:
            density_flow_global_initialization = st.selectbox(
                tr("グローバル残差平行移動の初期化", "Global residual translation initialization"),
                ["off", "auto"],
                index=0,
                key="workflow-c-density-flow-global-initialization",
                format_func=lambda value: "Off" if value == "off" else "Auto",
                help=tr(
                    "Offは変位場をゼロから開始します。Autoはaffine後に追加の全体平行移動候補を選びます。",
                    "Off initializes the field at zero. Auto selects an additional post-affine global translation candidate.",
                ),
            )
            density_flow_pixel_size = st.number_input(
                tr("密度画素サイズ (um)", "Density pixel size (um)"),
                min_value=0.1,
                value=2.0,
                step=0.5,
                key="workflow-c-density-flow-pixel-size",
                help=tr("共有world-xy密度グリッドの画素サイズです。", "Pixel size of the shared world-xy density grid."),
            )
            density_flow_blur_scales_text = st.text_input(
                tr("密度blur scale (pixel)", "Density blur scales (pixels)"),
                value="8, 4, 2",
                key="workflow-c-density-flow-blur-scales",
                help=tr("粗い順にカンマ区切りで指定します。", "Comma-separated Gaussian scales, ordered coarse to fine."),
            )
            density_flow_levels = st.number_input(
                tr("最適化level数", "Number of optimization levels"),
                min_value=1,
                value=3,
                step=1,
                key="workflow-c-density-flow-levels",
            )
            density_flow_iterations = st.number_input(
                tr("各levelの反復回数", "Iterations per level"),
                min_value=1,
                value=12,
                step=1,
                key="workflow-c-density-flow-iterations",
            )
        with flow_right:
            density_flow_learning_rate = st.number_input(
                tr("更新率", "Learning rate"),
                min_value=0.01,
                value=0.2,
                step=0.05,
                key="workflow-c-density-flow-learning-rate",
            )
            density_flow_update_smoothing = st.number_input(
                tr("更新場の平滑化sigma (pixel)", "Update smoothing sigma (pixels)"),
                min_value=0.1,
                value=3.0,
                step=0.5,
                key="workflow-c-density-flow-update-smoothing",
            )
            density_flow_detect_axis_reversal = st.checkbox(
                tr("x/y逆転を検出して停止", "Detect x/y reversal and stop"),
                value=True,
                key="workflow-c-density-flow-detect-axis-reversal",
            )
        with st.expander(tr("Density-flow正則化", "Density-flow regularization"), expanded=False):
            regularization_left, regularization_right = st.columns(2)
            with regularization_left:
                density_flow_smoothness_weight = st.number_input(tr("平滑性penalty", "Smoothness penalty"), min_value=0.0, value=0.05, step=0.01, key="workflow-c-density-flow-smoothness")
                density_flow_magnitude_weight = st.number_input(tr("変位量penalty", "Field-magnitude penalty"), min_value=0.0, value=0.0005, step=0.0005, format="%.4f", key="workflow-c-density-flow-magnitude")
                density_flow_boundary_weight = st.number_input(tr("組織境界penalty", "Tissue-boundary penalty"), min_value=0.0, value=0.02, step=0.01, key="workflow-c-density-flow-boundary")
            with regularization_right:
                density_flow_jacobian_weight = st.number_input(tr("Jacobian barrier重み", "Jacobian barrier weight"), min_value=0.0, value=1.0, step=0.25, key="workflow-c-density-flow-jacobian")
                density_flow_inverse_weight = st.number_input(tr("逆整合性penalty（任意）", "Inverse-consistency penalty (optional)"), min_value=0.0, value=0.0, step=0.01, key="workflow-c-density-flow-inverse")

    elif fine_alignment_method == "matched nuclei RBF":
        rbf_left, rbf_right = st.columns(2)
        with rbf_left:
            match_radius = st.number_input(tr("対応探索半径 (um)", "Match radius (um)"), min_value=0.1, value=10.0, step=1.0, key="workflow-c-rbf-match-radius", help=tr("HE核とGeoJSON核の対応候補を探す半径です。", "Radius used to find candidate HE-to-GeoJSON matches."))
            min_pair_confidence = st.slider(tr("最小対応信頼度", "Minimum pair confidence"), 0.0, 1.0, 0.05, 0.01, key="workflow-c-rbf-confidence")
            coherence_radius = st.number_input(tr("局所整合性半径 (um)", "Coherence radius (um)"), min_value=0.1, value=30.0, step=5.0, key="workflow-c-rbf-coherence")
            max_local_deviation = st.number_input(tr("最大局所偏差 (um)", "Maximum local deviation (um)"), min_value=0.1, value=8.0, step=1.0, key="workflow-c-rbf-local-deviation")
        with rbf_right:
            max_snap_displacement = st.number_input(tr("最大スナップ変位 (um)", "Maximum snap displacement (um)"), min_value=0.1, value=25.0, step=5.0, key="workflow-c-rbf-max-displacement")
            min_matched_anchor_pairs = st.number_input(tr("最小対応アンカー数", "Minimum matched anchors"), min_value=1, value=6, step=1, key="workflow-c-rbf-min-anchors")
            local_smoothing = st.number_input(tr("RBF平滑化", "RBF smoothing"), min_value=0.0, value=10.0, step=1.0, key="workflow-c-matched-smoothing", help=tr("大きいほど滑らかな変位場になります。", "Higher values produce a smoother displacement field."))
            local_kernel = st.selectbox(tr("RBFカーネル", "RBF kernel"), ["thin_plate_spline", "linear", "cubic", "quintic"], index=1, key="workflow-c-matched-kernel")
            local_neighbors = st.number_input(tr("RBF近傍アンカー数", "RBF neighbors"), min_value=0, value=40, step=5, key="workflow-c-matched-neighbors", help=tr("多いほど滑らかで安定し、少ないほど局所的です。", "Higher values are smoother and more stable; lower values are more local."))

    elif fine_alignment_method == "local translation field":
        local_preset = st.selectbox(
            tr("局所平行移動プリセット", "Local translation preset"), list(local_presets),
            index=list(local_presets).index(local_preset), key="workflow-c-local-preset",
            help=tr("balancedが標準です。aggressive/debugは診断用の緩い設定です。", "Balanced is standard; aggressive/debug are permissive diagnostic settings."),
        )
        local_default = local_presets[local_preset]
        if local_preset in {"aggressive", "debug"}:
            st.warning(tr("強い変形を許容する設定です。Jacobian QCを確認してください。", "This preset permits stronger deformation. Review Jacobian QC."))
        local_left, local_right = st.columns(2)
        with local_left:
            local_density_sigma = st.number_input(tr("密度sigma (um)", "Density sigma (um)"), min_value=0.1, value=local_default["density_sigma"], step=0.5, key="workflow-c-local-density-sigma", help=tr("核点群を密度マップ化するガウシアン幅です。", "Gaussian width used to create density maps."))
            local_density_pixel_size = st.number_input(tr("密度画素サイズ (um)", "Density pixel size (um)"), min_value=0.1, value=1.0, step=0.5, key="workflow-c-local-pixel-size")
            local_grid_spacing = st.number_input(tr("局所グリッド間隔 (um)", "Local grid spacing (um)"), min_value=1.0, value=local_default["grid_spacing"], step=5.0, key="workflow-c-local-grid-spacing")
            local_patch_radius = st.number_input(tr("局所パッチ半径 (um)", "Local patch radius (um)"), min_value=1.0, value=local_default["patch_radius"], step=5.0, key="workflow-c-local-patch-radius")
        with local_right:
            local_search_radius = st.number_input(tr("局所探索半径 (um)", "Local search radius (um)"), min_value=1.0, value=local_default["search_radius"], step=2.0, key="workflow-c-local-search-radius", help=tr("局所平行移動を探す最大範囲です。", "Maximum range searched for local translation."))
            local_min_correlation = st.slider(tr("最小局所相関", "Minimum local correlation"), 0.0, 1.0, local_default["min_correlation"], 0.05, key="workflow-c-local-correlation")
            local_max_shift = st.number_input(tr("最大局所移動量 (um)", "Maximum local shift (um)"), min_value=1.0, value=local_default["max_shift"], step=5.0, key="workflow-c-local-max-shift")
            local_min_anchors = st.number_input(tr("最小採用アンカー数", "Minimum accepted anchors"), min_value=1, value=local_default["min_accepted_anchors"], step=1, key="workflow-c-local-min-anchors")
        with st.expander(tr("詳細設定", "Advanced settings"), expanded=False):
            local_point_weight = st.number_input(tr("点の重み", "Point weight"), min_value=0.1, value=1.0, step=0.5, key="workflow-c-local-point-weight")
            local_outlier_percentile = st.slider(tr("アンカー外れ値分位 (%)", "Anchor outlier percentile (%)"), 50.0, 100.0, 95.0, 1.0, key="workflow-c-local-outlier")
            local_neighbor_radius = st.number_input(tr("近傍整合性半径 (um)", "Neighbor consistency radius (um)"), min_value=1.0, value=120.0, step=10.0, key="workflow-c-local-neighbor-radius")
            local_smoothing = st.number_input(tr("RBF平滑化", "RBF smoothing"), min_value=0.0, value=10.0, step=1.0, key="workflow-c-local-smoothing")
            local_kernel = st.selectbox(tr("RBFカーネル", "RBF kernel"), ["thin_plate_spline", "linear", "cubic", "quintic"], index=1, key="workflow-c-local-kernel")
            local_neighbors = st.number_input(tr("RBF近傍アンカー数", "RBF neighbors"), min_value=0, value=40, step=5, key="workflow-c-local-neighbors", help=tr("多いほど滑らかで、少ないほど局所的です。", "Higher values are smoother; lower values are more local."))

    elif fine_alignment_method == "center-snap":
        snap_left, snap_right = st.columns(2)
        with snap_left:
            match_radius = st.number_input(tr("対応探索半径 (um)", "Match radius (um)"), min_value=0.1, value=10.0, step=1.0, key="workflow-c-snap-match-radius")
            fine_bandwidth = st.number_input(tr("変位平滑化幅 (um)", "Warp bandwidth (um)"), min_value=0.1, value=12.0, step=1.0, key="workflow-c-snap-bandwidth")
            grid_spacing = st.number_input(tr("変位場グリッド間隔 (um)", "Warp grid spacing (um)"), min_value=0.1, value=6.0, step=1.0, key="workflow-c-snap-grid-spacing")
            ridge = st.number_input(tr("Ridge正則化", "Ridge regularization"), min_value=0.0, value=0.3, step=0.1, key="workflow-c-snap-ridge")
        with snap_right:
            min_pair_confidence = st.slider(tr("最小対応信頼度", "Minimum pair confidence"), 0.0, 1.0, 0.05, 0.01, key="workflow-c-snap-confidence")
            coherence_radius = st.number_input(tr("局所整合性半径 (um)", "Coherence radius (um)"), min_value=0.1, value=30.0, step=5.0, key="workflow-c-snap-coherence")
            max_local_deviation = st.number_input(tr("最大局所偏差 (um)", "Maximum local deviation (um)"), min_value=0.1, value=8.0, step=1.0, key="workflow-c-snap-local-deviation")
            snap_strength = st.slider(tr("スナップ強度", "Snap strength"), 0.1, 2.5, 1.25, 0.05, key="workflow-c-snap-strength")
            max_snap_displacement = st.number_input(tr("最大スナップ変位 (um)", "Maximum snap displacement (um)"), min_value=0.1, value=25.0, step=5.0, key="workflow-c-snap-max-displacement")

    elif fine_alignment_method == "off":
        st.info(tr("詳細位置合わせは無効です。最終結果にはアフィン位置合わせのみを使います。", "Fine alignment is disabled. The final result will use affine registration only."))

    if fine_alignment_method == "cluster-anchor":
        st.caption(tr("局所的に一致する核集団の移動をアンカーとして滑らかな変位場を作ります。", "Builds a smooth displacement field from reliable local cluster translations."))
        cluster_selection_mode = st.selectbox(
            tr("クラスタ選択方式", "Cluster selection mode"),
            ["radius", "hybrid k-nearest"],
            index=1,
            format_func=lambda value: {
                "radius": tr("半径（従来方式）", "Radius (legacy)"),
                "hybrid k-nearest": tr("Hybrid k-nearest", "Hybrid k-nearest"),
            }[value],
            key="workflow-c-cluster-selection-mode",
            help=tr(
                "Hybridでは各アンカーの核数を揃えつつ、遠すぎる核を除外します。",
                "Hybrid balances cluster sizes while excluding nuclei beyond the radius limit.",
            ),
        )
        cluster_patch_radius = 18.0
        cluster_left, cluster_right = st.columns(2)
        with cluster_left:
            cluster_grid_spacing = st.number_input(tr("クラスタグリッド間隔 (um)", "Cluster grid spacing (um)"), min_value=1.0, value=35.0, step=5.0, key="workflow-c-cluster-grid-spacing", help=tr("局所アンカー候補を評価する格子間隔です。", "Spacing between local anchor candidate centers."))
            if cluster_selection_mode == "radius":
                cluster_patch_radius = st.number_input(tr("クラスタパッチ半径 (um)", "Cluster patch radius (um)"), min_value=1.0, value=18.0, step=2.0, key="workflow-c-cluster-patch-radius", help=tr("各格子点で核集団を集める半径です。", "Radius used to collect nuclei around each grid center."))
            cluster_search_radius = st.number_input(tr("クラスタ探索半径 (um)", "Cluster search radius (um)"), min_value=1.0, value=25.0, step=2.0, key="workflow-c-cluster-search-radius", help=tr("最良の局所平行移動を探す最大範囲です。", "Maximum region searched when estimating the best local translation."))
            min_points_per_cluster = st.number_input(tr("クラスタ最小点数", "Minimum points per cluster"), min_value=1, value=8, step=1, key="workflow-c-cluster-min-points", help=tr("fixedまたはmovingがこの点数未満の局所領域は除外します。", "Local windows are skipped when either fixed or moving has fewer points."))
        with cluster_right:
            cluster_min_improvement = st.number_input(tr("最小改善距離 (um)", "Minimum cluster improvement (um)"), min_value=0.0, value=1.0, step=0.5, key="workflow-c-cluster-min-improvement", help=tr("移動前より中央値距離が最低限改善すべき量です。", "Required reduction in median distance versus zero shift."))
            cluster_max_shift = st.number_input(tr("最大クラスタ移動量 (um)", "Maximum cluster shift (um)"), min_value=1.0, value=35.0, step=5.0, key="workflow-c-cluster-max-shift", help=tr("個々のクラスタアンカーで試す最大移動量です。最終変位の安全上限とは別です。", "Maximum translation tested for an individual cluster anchor. This differs from the final displacement safety limit."))
            cluster_interpolation = st.selectbox(tr("補間方式", "Interpolation method"), ["rbf", "b-spline"], index=0, key="workflow-c-cluster-interpolation", help=tr("採用アンカーから変位場を作る方式です。", "Method used to fit a displacement field from accepted anchors."))
        target_points_per_cluster = 20
        max_cluster_radius_um = 40.0
        moving_candidate_pool_ratio = 1.5
        if cluster_selection_mode == "hybrid k-nearest":
            hybrid_left, hybrid_mid, hybrid_right = st.columns(3)
            with hybrid_left:
                target_points_per_cluster = st.number_input(
                    tr("目標クラスタ点数", "Target points per cluster"),
                    min_value=1,
                    value=20,
                    step=1,
                    key="workflow-c-cluster-target-points",
                    help=tr("fixed側で各アンカーから選ぶ近傍核数の上限です。", "Maximum number of nearest fixed nuclei selected per anchor."),
                )
            with hybrid_mid:
                max_cluster_radius_um = st.number_input(
                    tr("最大クラスタ半径 (um)", "Maximum cluster radius (um)"),
                    min_value=0.1,
                    value=40.0,
                    step=5.0,
                    key="workflow-c-cluster-max-radius",
                    help=tr("k近傍でもこの距離を超えるfixed核は除外します。", "Fixed nuclei beyond this radius are excluded even when k-nearest."),
                )
            with hybrid_right:
                moving_candidate_pool_ratio = st.number_input(
                    tr("Moving候補倍率", "Moving candidate pool ratio"),
                    min_value=0.1,
                    value=1.5,
                    step=0.1,
                    key="workflow-c-cluster-moving-pool-ratio",
                    help=tr("moving側候補数を目標点数の何倍まで集めるかを指定します。", "Multiplier applied to the target count for the moving candidate pool."),
                )
        with st.expander(tr("詳細設定", "Advanced settings"), expanded=False):
            adv_left, adv_right = st.columns(2)
            with adv_left:
                cluster_search_step = st.number_input(tr("クラスタ探索刻み (um)", "Cluster search step (um)"), min_value=0.1, value=2.5, step=0.5, key="workflow-c-cluster-search-step")
                cluster_match_threshold = st.number_input(tr("クラスタ一致閾値 (um)", "Cluster match threshold (um)"), min_value=0.1, value=5.0, step=0.5, key="workflow-c-cluster-match-threshold")
                local_outlier_percentile = st.slider(tr("アンカー外れ値分位 (%)", "Anchor outlier percentile (%)"), 50.0, 100.0, 95.0, 1.0, key="workflow-c-cluster-outlier")
                local_neighbor_radius = st.number_input(tr("近傍整合性半径 (um)", "Neighbor consistency radius (um)"), min_value=1.0, value=120.0, step=10.0, key="workflow-c-cluster-neighbor-radius")
                local_support_radius = st.number_input(tr("局所支持半径 (um)", "Local support radius (um)"), min_value=1.0, value=120.0, step=10.0, key="workflow-c-cluster-support-radius")
            with adv_right:
                local_kernel = st.selectbox(tr("RBFカーネル", "RBF kernel"), ["thin_plate_spline", "linear", "cubic", "quintic"], index=1, key="workflow-c-cluster-kernel")
                local_smoothing = st.number_input(tr("RBF平滑化", "RBF smoothing"), min_value=0.0, value=10.0, step=1.0, key="workflow-c-cluster-smoothing")
                local_neighbors = st.number_input(tr("RBF近傍アンカー数", "RBF neighbors"), min_value=0, value=40, step=5, key="workflow-c-cluster-neighbors", help=tr("多いほど滑らかで安定し、少ないほど局所的です。", "Higher values are smoother and more stable; lower values are more local."))
                control_grid_spacing = st.number_input(tr("B-spline制御格子間隔 (um)", "B-spline control grid spacing (um)"), min_value=1.0, value=35.0, step=5.0, key="workflow-c-bspline-grid-spacing")
                cluster_regularization = st.number_input(tr("B-spline正則化", "B-spline regularization"), min_value=0.0, value=3.0, step=1.0, key="workflow-c-bspline-regularization")
                cluster_min_anchors = st.number_input(tr("最小採用アンカー数", "Minimum accepted anchors"), min_value=1, value=8, step=1, key="workflow-c-cluster-min-anchors")
    else:
        cluster_grid_spacing, cluster_patch_radius, cluster_search_radius = 35.0, 18.0, 25.0
        cluster_selection_mode = "radius"
        target_points_per_cluster, min_points_per_cluster = 20, 8
        max_cluster_radius_um, moving_candidate_pool_ratio = 40.0, 1.5
        cluster_search_step, cluster_match_threshold = 2.5, 5.0
        cluster_min_improvement, cluster_max_shift, cluster_min_anchors = 1.0, 35.0, 8
        cluster_interpolation, control_grid_spacing = "rbf", 35.0
        local_support_radius, cluster_regularization = 120.0, 3.0
    st.subheader(tr("4. 組織領域と境界処理", "4. Tissue and boundary handling"))
    tissue_left, tissue_mid, tissue_right = st.columns(3)
    with tissue_left:
        tissue_mask_threshold = st.slider(
            tr("組織マスク閾値", "Tissue mask threshold"), 0.0, 1.0, 0.05, 0.01,
            key="workflow-c-tissue-threshold", help=tr("Warp済みHE画像から背景を除く閾値です。", "Threshold used to separate tissue from background in the warped HE image."),
        )
        edge_margin = st.number_input(
            tr("Fine warp組織端マージン (pixel)", "Fine-warp tissue edge margin (pixels)"), min_value=0.0, value=10.0, step=2.0,
            key="workflow-c-edge-margin", help=tr("境界付近の点をvalidではなくedge candidateに分類する幅です。", "Points near tissue or image boundaries are classified as edge candidates rather than reliable valid targets."),
        )
    with tissue_mid:
        use_edge_candidates_for_anchors = st.checkbox(
            tr("Edge candidateをアンカーに使用", "Use edge candidates for anchors"), value=False,
            key="workflow-c-use-edge-candidates", help=tr("OFFではvalid GeoJSON点だけをfine warpに使います。", "When off, only valid GeoJSON points are used for fine warp."),
        )
        valid_geojson_weight = st.number_input(tr("Valid GeoJSONの重み", "Valid GeoJSON weight"), min_value=0.1, max_value=2.0, value=1.0, step=0.1, key="workflow-c-valid-weight")
        edge_candidate_weight = st.number_input(tr("Edge candidateの重み", "Edge candidate weight"), min_value=0.0, max_value=1.0, value=0.0, step=0.05, key="workflow-c-edge-weight", help=tr("使用する場合の相対的な寄与です。", "Relative contribution when edge candidates are enabled."))
    with tissue_right:
        enable_boundary_pinning = st.checkbox(tr("境界ピン留めを有効化", "Enable boundary pinning"), value=True, key="workflow-c-enable-boundary-pinning", help=tr("ゼロ変位アンカーで中央の変形が背景へ広がるのを抑えます。", "Zero-displacement anchors prevent central deformation from propagating into the background."))
        boundary_anchor_spacing = st.number_input(tr("境界アンカー間隔 (pixel)", "Boundary anchor spacing (pixels)"), min_value=2.0, value=40.0, step=5.0, key="workflow-c-boundary-spacing")
        boundary_anchor_weight = st.number_input(tr("境界アンカー重み", "Boundary anchor weight"), min_value=0.01, value=30.0, step=5.0, key="workflow-c-boundary-weight", help=tr("画像端への変形伝播を防ぐゼロ変位アンカーの強さです。", "Strength of zero-displacement anchors used to prevent central deformation from propagating into the image border."))
        include_image_border_pins = st.checkbox(tr("画像枠ピンを含める", "Include image border pins"), value=True, key="workflow-c-image-border-pins")
        include_tissue_boundary_pins = st.checkbox(tr("組織境界ピンを含める", "Include tissue boundary pins"), value=True, key="workflow-c-tissue-boundary-pins")

    st.subheader(tr("5. Warp安全条件", "5. Warp safety"))
    safety_left, safety_mid, safety_right = st.columns(3)
    with safety_left:
        max_final_displacement_um = st.number_input(
            tr("最大最終変位 (um)", "Maximum final displacement (um)"),
            min_value=0.1,
            value=35.0,
            step=5.0,
            key="workflow-c-max-final-displacement",
            help=tr("補間後の最終変位場に対する安全上限です。最大クラスタ移動量とは別です。", "Safety limit for the final interpolated displacement field. This differs from maximum cluster shift."),
        )
    with safety_mid:
        jacobian_min_limit = st.number_input(tr("Jacobian最小値", "Jacobian minimum limit"), min_value=-1.0, value=0.1, step=0.05, key="workflow-c-jacobian-min", help=tr("局所折り返しや過圧縮を検出する下限です。", "Lower limit used to detect fold-over or excessive compression."))
        jacobian_max_limit = st.number_input(tr("Jacobian最大値", "Jacobian maximum limit"), min_value=1.0, value=3.0, step=0.5, key="workflow-c-jacobian-max", help=tr("過度な局所膨張を検出する上限です。", "Upper limit used to detect excessive local expansion."))
    with safety_right:
        enable_displacement_p95_limit = st.checkbox(tr("変位95分位上限を有効化", "Enable displacement p95 limit"), value=True, key="workflow-c-enable-p95-limit")
        displacement_p95_limit_um = st.number_input(tr("変位95分位上限 (um)", "Displacement p95 limit (um)"), min_value=0.1, value=30.0, step=5.0, key="workflow-c-p95-limit", help=tr("大部分の領域で変位が大きすぎないか確認します。", "Checks that displacement is not excessive across most of the field."))

    st.subheader(tr("6. 出力と表示", "6. Output and display"))
    registration_display_origin = st.selectbox(
        tr("位置合わせQC表示の原点", "Registration QC display origin"),
        ["lower-left", "upper-left", "upper-right"],
        index=0,
        key="workflow-c-registration-origin",
        help=tr("散布図/QCの向きだけを変え、計算結果には影響しません。", "Controls scatter/QC orientation only and does not affect registration."),
    )
    warped_he_output_origin = st.selectbox(
        tr("Warp済みHE出力の原点", "Warped HE output origin"),
        ["lower-left", "upper-left", "upper-right"],
        index=0,
        key="workflow-c-output-origin",
        help=tr("出力画像の向きだけを変え、位置合わせ品質には影響しません。", "Controls exported image orientation only, not registration quality."),
    )
    st.info(tr("出力原点は画像の表示方向だけを制御し、位置合わせ品質のパラメータではありません。", "Warped HE output origin controls exported image orientation, not registration quality."))
    warped_he_pixel_size = st.number_input(
        tr("Warp済みHEの画素サイズ (um)", "Warped HE output pixel size (um)"),
        min_value=0.1,
        value=1.0,
        step=0.5,
        key="workflow-c-output-pixel-size",
        help=tr("小さいほど高解像度ですがPNGが大きくなります。", "Smaller values create higher-resolution, larger PNG files."),
    )
    max_warped_overlay_points = st.number_input(
        tr("オーバーレイ最大点数", "Maximum warped HE overlay points"),
        min_value=100,
        max_value=20000,
        value=3000,
        step=500,
        key="workflow-c-max-overlay-points",
        help=tr("表示負荷を抑えるため描画点数を制限します。", "Limits plotted points to keep QC rendering responsive."),
    )
    show_excluded_geojson_points = st.checkbox(tr("除外GeoJSON点を表示", "Show excluded GeoJSON points"), value=False, key="workflow-c-show-excluded")
    show_edge_candidate_geojson_points = st.checkbox(tr("Edge candidateを表示", "Show edge candidates"), value=True, key="workflow-c-show-edge")
    show_boundary_pin_anchors = st.checkbox(tr("境界ピンアンカーを重ねる", "Overlay boundary pin anchors"), value=True, key="workflow-c-show-boundary-pins")

    if fine_alignment_method == "tissue-aware density flow":
        selected_preset = "experimental multiscale"
        search_radius_summary = "density objective"
        local_shift_summary = max_final_displacement_um
        interpolation_summary = "composed smooth updates"
    else:
        selected_preset = local_preset if fine_alignment_method == "local translation field" else ("current defaults" if fine_alignment_method == "cluster-anchor" else "-")
        search_radius_summary = cluster_search_radius if fine_alignment_method == "cluster-anchor" else (local_search_radius if fine_alignment_method == "local translation field" else match_radius)
        local_shift_summary = cluster_max_shift if fine_alignment_method == "cluster-anchor" else (local_max_shift if fine_alignment_method == "local translation field" else max_snap_displacement)
        interpolation_summary = cluster_interpolation if fine_alignment_method == "cluster-anchor" else ("RBF" if fine_alignment_method in {"matched nuclei RBF", "local translation field"} else ("center-snap" if fine_alignment_method == "center-snap" else "-"))
    ui_parameter_summary = {
        "fine_alignment_method": fine_alignment_method,
        "preset": selected_preset,
        "use_edge_candidates_for_anchors": use_edge_candidates_for_anchors,
        "interpolation": interpolation_summary,
        "search_radius_um": search_radius_summary,
        "maximum_local_shift_um": local_shift_summary,
        "maximum_final_displacement_um": max_final_displacement_um,
        "jacobian_min_limit": jacobian_min_limit,
        "jacobian_max_limit": jacobian_max_limit,
        "boundary_pinning": enable_boundary_pinning,
    }
    st.subheader(tr("7. 実行設定の要約", "7. Run summary"))
    compact_summary = pd.DataFrame(
        {
            tr("項目", "Parameter"): [
                tr("詳細方式", "Fine method"), tr("プリセット", "Preset"),
                tr("Edge candidate使用", "Use edge candidates"), tr("補間", "Interpolation"),
                tr("探索半径 (um)", "Search radius (um)"), tr("最大局所移動 (um)", "Maximum local shift (um)"),
                tr("最大最終変位 (um)", "Maximum final displacement (um)"), "Jacobian min / max",
                tr("境界ピン留め", "Boundary pinning"),
            ],
            tr("値", "Value"): [
                method_labels[fine_alignment_method], selected_preset,
                tr("使用", "On") if use_edge_candidates_for_anchors else tr("不使用", "Off"), interpolation_summary,
                str(search_radius_summary), str(local_shift_summary), str(max_final_displacement_um),
                f"{jacobian_min_limit} / {jacobian_max_limit}",
                tr("有効", "Enabled") if enable_boundary_pinning else tr("無効", "Disabled"),
            ],
        }
    )
    st.dataframe(compact_summary, use_container_width=True, hide_index=True)
    with st.expander(tr("全パラメータ要約", "Full parameter summary"), expanded=False):
        st.json(ui_parameter_summary)

    st.subheader(tr("8. 結果と診断", "8. Results and diagnostics"))

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
        anchor_target_mask = np.ones(len(geojson_array), dtype=bool)
        success_metric_targets = geojson_array
        he_classification = np.full(len(he_array), "valid", dtype=object)
        he_metric_mask = np.ones(len(he_array), dtype=bool)
        success_metric_moving_points = affine_result.transformed_points
        boundary_anchor_points = None
        tissue_validity_table = None
        he_tissue_validity_table = None
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
                valid_weight=valid_geojson_weight,
                edge_candidate_weight=edge_candidate_weight if use_edge_candidates_for_anchors else 0.0,
                moving_points=affine_result.transformed_points,
                max_nearest_he_distance=cluster_patch_radius + cluster_search_radius,
            )
            anchor_target_mask = geojson_classification == "valid"
            if use_edge_candidates_for_anchors:
                anchor_target_mask |= geojson_classification == "edge_candidate"
            fine_target_array = geojson_array[anchor_target_mask]
            fine_target_weights = geojson_weights[anchor_target_mask]
            if tissue_validity_table is not None:
                tissue_validity_table["used_for_anchor"] = anchor_target_mask
                tissue_validity_table["target_weight"] = np.where(anchor_target_mask, geojson_weights, 0.0)
                tissue_validity_table["valid_for_fine_warp"] = anchor_target_mask
            valid_targets = geojson_array[geojson_classification == "valid"]
            success_metric_targets = valid_targets if len(valid_targets) else fine_target_array
            _, _, he_classification, he_tissue_validity_table = _point_tissue_validity(
                affine_result.transformed_points,
                affine_tissue_metadata,
                tissue_mask,
                edge_margin=edge_margin,
                valid_weight=1.0,
                edge_candidate_weight=1.0,
                moving_points=None,
                max_nearest_he_distance=None,
            )
            he_metric_mask = he_classification == "valid"
            success_metric_moving_points = (
                affine_result.transformed_points[he_metric_mask]
                if np.any(he_metric_mask)
                else affine_result.transformed_points
            )
            if enable_boundary_pinning:
                boundary_anchor_points = _boundary_anchor_points_from_tissue_mask(
                    tissue_mask,
                    affine_tissue_metadata,
                    spacing_px=boundary_anchor_spacing,
                    include_image_border=include_image_border_pins,
                    include_tissue_boundary=include_tissue_boundary_pins,
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
        elif fine_alignment_method == "tissue-aware density flow":
            try:
                density_flow_blur_scales = tuple(
                    float(value.strip())
                    for value in density_flow_blur_scales_text.split(",")
                    if value.strip()
                )
            except ValueError as exc:
                raise ValueError("Density blur scales must be comma-separated numbers.") from exc
            if not density_flow_blur_scales:
                raise ValueError("At least one density blur scale is required.")
            density_flow_fixed_points = (
                success_metric_targets if len(success_metric_targets) else fine_target_array
            )
            fine_result = tissue_aware_density_flow_registration(
                density_flow_fixed_points,
                affine_result.transformed_points,
                success_metric_fixed_points=density_flow_fixed_points,
                success_metric_moving_points=success_metric_moving_points,
                density_pixel_size=density_flow_pixel_size,
                density_blur_scales=density_flow_blur_scales,
                optimization_levels=int(density_flow_levels),
                iterations_per_level=int(density_flow_iterations),
                learning_rate=density_flow_learning_rate,
                update_smoothing_sigma=density_flow_update_smoothing,
                smoothness_weight=density_flow_smoothness_weight,
                magnitude_weight=density_flow_magnitude_weight,
                jacobian_barrier_weight=density_flow_jacobian_weight,
                tissue_boundary_weight=density_flow_boundary_weight,
                inverse_consistency_weight=density_flow_inverse_weight,
                jacobian_min_threshold=jacobian_min_limit,
                jacobian_max_threshold=jacobian_max_limit,
                max_displacement=max_final_displacement_um,
                displacement_p95_limit=(
                    displacement_p95_limit_um if enable_displacement_p95_limit else None
                ),
                global_translation_initialization=density_flow_global_initialization,
                detect_axis_reversal=density_flow_detect_axis_reversal,
            )
        elif fine_alignment_method == "cluster-anchor":
            fine_result = cluster_anchor_fine_warp(
                fine_target_array,
                affine_result.transformed_points,
                fixed_point_weights=fine_target_weights,
                success_metric_fixed_points=success_metric_targets,
                success_metric_moving_points=success_metric_moving_points,
                boundary_anchor_points=boundary_anchor_points,
                boundary_anchor_weight=boundary_anchor_weight,
                grid_spacing=control_grid_spacing if cluster_interpolation == "b-spline" else cluster_grid_spacing,
                patch_radius=cluster_patch_radius,
                search_radius=cluster_search_radius,
                search_step=cluster_search_step,
                cluster_selection_mode=cluster_selection_mode,
                target_points_per_cluster=int(target_points_per_cluster),
                min_points_per_cluster=int(min_points_per_cluster),
                max_cluster_radius_um=max_cluster_radius_um,
                moving_candidate_pool_ratio=moving_candidate_pool_ratio,
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
                jacobian_min_threshold=jacobian_min_limit,
                jacobian_max_threshold=jacobian_max_limit,
                max_final_displacement=max_final_displacement_um,
                displacement_p95_limit=displacement_p95_limit_um if enable_displacement_p95_limit else None,
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

    affine_points = np.asarray(affine_result.transformed_points, dtype=float)
    attempted_points_value = getattr(fine_result, "attempted_transformed_points", None)
    attempted_displacement_x_value = getattr(fine_result, "attempted_displacement_x", None)
    variants = _workflow_c_result_variants(affine_points, fine_result)
    fine_applied = variants["fine_applied"]
    attempted_points = variants["attempted_points"]
    attempted_displacement_x = variants["attempted_displacement_x"]
    attempted_displacement_y = variants["attempted_displacement_y"]
    final_points = variants["final_points"]
    final_displacement_x = variants["final_displacement_x"]
    final_displacement_y = variants["final_displacement_y"]
    applied_result_label = variants["applied_result_label"]
    density_flow_mode = fine_alignment_method == "tissue-aware density flow"
    image_fine_applied = fine_applied
    image_applied_result_label = applied_result_label

    final_fine_result = replace(
        fine_result,
        transformed_points=final_points,
        displacement_x=final_displacement_x,
        displacement_y=final_displacement_y,
        jacobian_min=fine_result.jacobian_min if fine_applied else 1.0,
        jacobian_max=fine_result.jacobian_max if fine_applied else 1.0,
        max_displacement=fine_result.max_displacement if fine_applied else 0.0,
        applied=fine_applied,
    )

    transformed_affine_points = he_points.copy()
    transformed_affine_points["centroid_x"] = affine_points[:, 0]
    transformed_affine_points["centroid_y"] = affine_points[:, 1]
    transformed_affine_points["source"] = "he_affine_world_um"

    transformed_fine_points = he_points.copy()
    transformed_fine_points["centroid_x"] = final_points[:, 0]
    transformed_fine_points["centroid_y"] = final_points[:, 1]
    transformed_fine_points["source"] = "he_final_applied_world_um"

    rejection_reason = getattr(fine_result, "rejection_reason", None) or ""

    transformed_attempted_points = he_points.copy()
    transformed_attempted_points["centroid_x"] = attempted_points[:, 0]
    transformed_attempted_points["centroid_y"] = attempted_points[:, 1]
    transformed_attempted_points["source"] = "he_attempted_fine_world_um"

    metric_fixed_points = success_metric_targets if len(success_metric_targets) else geojson_array
    metric_moving_affine_points = affine_points[he_metric_mask] if np.any(he_metric_mask) else affine_points
    metric_moving_attempted_points = attempted_points[he_metric_mask] if np.any(he_metric_mask) else attempted_points
    metric_moving_applied_points = final_points[he_metric_mask] if np.any(he_metric_mask) else final_points
    before_metrics = point_bidirectional_distance_metrics(metric_fixed_points, metric_moving_affine_points)
    attempted_metrics = point_bidirectional_distance_metrics(metric_fixed_points, metric_moving_attempted_points)
    applied_metrics = point_bidirectional_distance_metrics(metric_fixed_points, metric_moving_applied_points)
    safety_metrics = (fine_result.metrics or {}).get("safety", {}) if isinstance(fine_result.metrics, dict) else {}
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

    st.subheader(tr("実行状態", "Run status"))
    if status == "applied":
        st.success(tr("Fine warpを適用しました。最終結果は affine + fine warp です。", "Fine warp applied. The final result uses affine + fine warp."))
    elif status == "rejected":
        st.warning(tr("Fine warpは安全条件でrejectされました。最終結果はaffineのみです。", "Fine warp was rejected by safety checks. The final result uses affine-only registration."))
    else:
        st.info(tr("Fine alignmentは無効です。最終結果はaffineのみです。", "Fine alignment is disabled. The final result is affine-only."))
    if rejection_reason:
        st.caption(f"{tr('Reject理由', 'Rejection reason')}: {rejection_reason}")

    result_a, result_b, result_c, result_d = st.columns(4)
    result_a.metric("Applied result type", applied_result_label)
    result_b.metric("Affine median distance", f"{before_metrics['symmetric_median_distance']:.2f} um")
    result_c.metric("Attempted fine median", f"{attempted_metrics['symmetric_median_distance']:.2f} um")
    result_d.metric("Final applied median", f"{applied_metrics['symmetric_median_distance']:.2f} um")
    attempted_qc_a, attempted_qc_b, attempted_qc_c = st.columns(3)
    attempted_qc_a.metric(
        "Attempted Jacobian min / max",
        f"{safety_metrics.get('attempted_jacobian_min', fine_result.jacobian_min):.3f} / "
        f"{safety_metrics.get('attempted_jacobian_max', fine_result.jacobian_max):.3f}",
    )
    attempted_qc_b.metric(
        "Attempted max displacement",
        f"{safety_metrics.get('attempted_max_displacement', fine_result.max_displacement):.2f} um",
    )
    attempted_qc_c.metric("Final applied Jacobian", "fine field" if fine_applied else "1.000 (affine only)")

    st.subheader(tr("Fine alignment診断", "Fine alignment diagnostics"))
    diag_a, diag_b, diag_c, diag_d = st.columns(4)
    diag_a.metric("Fine method", fine_alignment_method)
    diag_b.metric("Fine status", status)
    if fine_alignment_method == "cluster-anchor" and fine_result.anchors is not None and "anchor_type" in fine_result.anchors:
        status_anchor_types = fine_result.anchors["anchor_type"]
        status_accepted = fine_result.anchors["accepted"].astype(bool)
        status_cluster_mask = status_anchor_types == "cluster"
        status_boundary_mask = status_anchor_types == "boundary_pin"
        diag_c.metric("Accepted cluster anchors", int(np.count_nonzero(status_cluster_mask & status_accepted)))
        diag_d.metric("Boundary pin anchors", int(np.count_nonzero(status_boundary_mask)))
    else:
        diag_c.metric("Accepted anchors", f"{accepted_anchor_count}/{total_anchor_count}")
        diag_d.metric("Rejected anchors", rejected_anchor_count)
    diag2_a, diag2_b, diag2_c, diag2_d = st.columns(4)
    diag2_a.metric("Median shift", f"{median_shift:.2f} um")
    diag2_b.metric("Shift p95", f"{p95_shift:.2f} um")
    diag2_c.metric("Max displacement", f"{fine_result.max_displacement:.2f} um")
    diag2_d.metric("Jacobian min", f"{fine_result.jacobian_min:.3f}")
    st.caption(f"Applied uses: {applied_result_label}")

    if fine_alignment_method == "cluster-anchor" and fine_result.anchors is not None:
        anchors = fine_result.anchors
        anchor_types = anchors["anchor_type"] if "anchor_type" in anchors else pd.Series("cluster", index=anchors.index)
        accepted_flags = anchors["accepted"].astype(bool) if "accepted" in anchors else pd.Series(False, index=anchors.index)
        cluster_mask = anchor_types == "cluster"
        boundary_mask = anchor_types == "boundary_pin"
        cluster_a, cluster_b, cluster_c, cluster_d = st.columns(4)
        cluster_a.metric("Accepted cluster anchors", int(np.count_nonzero(cluster_mask & accepted_flags)))
        cluster_b.metric("Rejected cluster anchors", int(np.count_nonzero(cluster_mask & ~accepted_flags)))
        cluster_c.metric("Boundary pin anchors", int(np.count_nonzero(boundary_mask)))
        cluster_d.metric("Total candidate cluster anchors", int(np.count_nonzero(cluster_mask)))
    else:
        st.caption(
            f"Fine alignment anchors: {fine_result.n_pairs} accepted / "
            f"{fine_result.n_candidate_pairs} candidates ({fine_result.n_filtered_pairs} rejected)"
        )

    overview_tab, point_tab, image_tab, tissue_tab, safety_tab, anchor_tab, downloads_tab = st.tabs(
        [
            tr("概要", "Overview"),
            tr("点群位置合わせ", "Point alignment"),
            tr("Warp済みHE画像", "Warped HE image"),
            tr("組織分類", "Tissue classification"),
            tr("Warp安全性", "Warp safety"),
            tr("アンカー診断", "Anchor diagnostics"),
            tr("ダウンロード", "Downloads"),
        ]
    )

    if fine_alignment_method == "tissue-aware density flow" and isinstance(fine_result.metrics, dict):
        flow_metadata = fine_result.metrics.get("density_flow", {})
        flow_history = pd.DataFrame(fine_result.metrics.get("optimization_history", []))
        flow_deformation = density_flow_deformation_diagnostics(
            attempted_displacement_x,
            attempted_displacement_y,
            pixel_size=fine_result.grid_spacing,
        )
        safety_tab.subheader(tr("Density-flow診断", "Density-flow diagnostics"))
        safety_tab.dataframe(
            pd.DataFrame(
                [
                    {
                        **flow_metadata,
                        "experimental_inverse_raster_warp": True,
                        "finite_output": safety_metrics.get("finite_output"),
                        "points_outside_tissue_before_fraction": safety_metrics.get(
                            "points_outside_tissue_before_fraction"
                        ),
                        "points_outside_tissue_attempted_fraction": safety_metrics.get(
                            "points_outside_tissue_attempted_fraction"
                        ),
                        "mutual_nearest_fraction_before": safety_metrics.get(
                            "mutual_nearest_fraction_before"
                        ),
                        "mutual_nearest_fraction_attempted": safety_metrics.get(
                            "mutual_nearest_fraction_attempted"
                        ),
                    }
                ]
            ),
            use_container_width=True,
        )
        safety_tab.caption(
            tr(
                "Global shiftはaffine後に追加された初期平行移動です。Local residualは最終fieldから中央値ベクトルを除いた空間変動です。",
                "Global shift is an additional post-affine initialization. Local residual is spatial variation after subtracting the field median vector.",
            )
        )
        safety_tab.dataframe(
            pd.DataFrame(
                [
                    {
                        "global_initialization": flow_metadata.get("global_translation_initialization", "off"),
                        "global_shift_x_um": flow_metadata.get("global_density_shift_x", 0.0),
                        "global_shift_y_um": flow_metadata.get("global_density_shift_y", 0.0),
                        **{
                            key: value
                            for key, value in flow_deformation.items()
                            if not isinstance(value, np.ndarray)
                        },
                    }
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        if bool(flow_deformation["translation_dominated"]):
            safety_tab.warning(
                "The density-flow result is dominated by global translation; little local nonlinear correction was detected."
            )
        else:
            safety_tab.info(
                tr(
                    "変位場には測定可能な空間変動があります。生物学的妥当性ではなく、Jacobianと画像QCで評価してください。",
                    "Measurable spatially varying deformation is present. This is not a biological-accuracy claim; review the Jacobian and image QC.",
                )
            )
        magnitude_col, residual_col = safety_tab.columns(2)
        with magnitude_col:
            flow_magnitude_figure = visualize_displacement_magnitude_heatmap(
                fine_result.grid_x,
                fine_result.grid_y,
                attempted_displacement_x,
                attempted_displacement_y,
                title="Attempted density-flow displacement magnitude",
            )
            st.pyplot(flow_magnitude_figure, clear_figure=False)
        with residual_col:
            flow_local_figure = visualize_displacement_magnitude_heatmap(
                fine_result.grid_x,
                fine_result.grid_y,
                flow_deformation["local_residual_x"],
                flow_deformation["local_residual_y"],
                title="Local residual displacement after median translation removal",
                colorbar_label="local residual (um)",
            )
            st.pyplot(flow_local_figure, clear_figure=False)
        if not flow_history.empty:
            safety_tab.caption(
                tr(
                    "目的関数値は各scale内の診断値です。異なるscale間の絶対値比較には注意してください。",
                    "Objective values are diagnostic within each scale; compare absolute values across scales cautiously.",
                )
            )
            safety_tab.line_chart(flow_history[["total", "density"]])
            with safety_tab.expander(tr("最適化履歴", "Optimization history"), expanded=False):
                st.dataframe(flow_history, use_container_width=True, hide_index=True)

    overview_tab.subheader(tr("座標診断", "Coordinate diagnostics"))
    overview_tab.caption(
        "Use this table to separate raw point ranges, registration output ranges, and possible GeoJSON ROI/range mismatch."
    )
    coordinate_diagnostics = pd.DataFrame(
        [
            _coordinate_summary_row("HE raw nuclei points", he_array),
            _coordinate_summary_row("GeoJSON centroids", geojson_array),
            _coordinate_summary_row("GeoJSON valid fine-warp targets", fine_target_array),
            _coordinate_summary_row("HE affine world points", affine_points),
            _coordinate_summary_row("HE final applied world points", final_points),
        ]
    )
    overview_tab.dataframe(coordinate_diagnostics, use_container_width=True)
    if tissue_validity_table is not None:
        tissue_tab.subheader(tr("組織有効性フィルタ", "Tissue-validity filtering"))
        validity_summary = (
            tissue_validity_table["exclusion_reason"]
            .value_counts(dropna=False)
            .rename_axis("status")
            .reset_index(name="n_points")
        )
        tissue_tab.dataframe(validity_summary, use_container_width=True)
        tissue_tab.download_button(
            "Download GeoJSON tissue-validity CSV",
            data=tissue_validity_table.to_csv(index=False).encode("utf-8"),
            file_name="geojson_tissue_validity.csv",
            mime="text/csv",
        )
        if he_tissue_validity_table is not None:
            he_validity_summary = (
                he_tissue_validity_table["exclusion_reason"]
                .value_counts(dropna=False)
                .rename_axis("status")
                .reset_index(name="n_points")
            )
            tissue_tab.caption("HE tissue-validity classification uses the same affine warped HE tissue mask.")
            tissue_tab.dataframe(he_validity_summary, use_container_width=True)
            tissue_tab.download_button(
                "Download HE tissue-validity CSV",
                data=he_tissue_validity_table.to_csv(index=False).encode("utf-8"),
                file_name="he_tissue_validity.csv",
                mime="text/csv",
            )
        if affine_tissue_image is not None and affine_tissue_metadata is not None:
            tissue_col, class_col = tissue_tab.columns(2)
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
    else:
        tissue_tab.info(
            tr(
                "HE画像または組織マスクがないため、組織分類は表示できません。点群位置合わせは利用できます。",
                "Tissue classification is unavailable without an HE image/tissue mask. Point alignment remains available.",
            )
        )

    safety_a, safety_b, safety_c, safety_d = safety_tab.columns(4)
    safety_a.metric(
        "Attempted max displacement",
        f"{safety_metrics.get('attempted_max_displacement', fine_result.max_displacement):.2f} um",
    )
    safety_b.metric(
        "Attempted p95 displacement",
        f"{safety_metrics.get('attempted_p95_displacement', p95_shift):.2f} um",
    )
    safety_c.metric(
        "Attempted Jacobian min",
        f"{safety_metrics.get('attempted_jacobian_min', fine_result.jacobian_min):.3f}",
    )
    safety_d.metric(
        "Attempted Jacobian max",
        f"{safety_metrics.get('attempted_jacobian_max', fine_result.jacobian_max):.3f}",
    )
    if rejection_reason and not fine_applied:
        safety_tab.warning(f"Rejection reason: {rejection_reason}")
    if not fine_applied:
        safety_tab.info(
            "Final applied result is affine-only: applied fine displacement = 0 and applied fine Jacobian = 1."
        )
    n_valid_geojson = int(np.count_nonzero(geojson_classification == "valid"))
    n_edge_geojson = int(np.count_nonzero(geojson_classification == "edge_candidate"))
    n_excluded_geojson = int(np.count_nonzero(geojson_classification == "excluded"))
    n_valid_used_for_anchors = int(np.count_nonzero(anchor_target_mask & (geojson_classification == "valid")))
    n_edge_used_for_anchors = int(np.count_nonzero(anchor_target_mask & (geojson_classification == "edge_candidate")))
    n_valid_he = int(np.count_nonzero(he_classification == "valid"))
    n_edge_he = int(np.count_nonzero(he_classification == "edge_candidate"))
    n_excluded_he = int(np.count_nonzero(he_classification == "excluded"))
    cluster_anchors_before_filter = 0
    cluster_anchors_after_filter = 0
    boundary_pinned_anchors = 0
    if fine_result.anchors is not None and "anchor_type" in fine_result.anchors:
        cluster_anchor_mask = fine_result.anchors["anchor_type"] == "cluster"
        boundary_pin_mask = fine_result.anchors["anchor_type"] == "boundary_pin"
        boundary_pinned_anchors = int(np.count_nonzero(boundary_pin_mask))
        if "accepted_before_filter" in fine_result.anchors:
            cluster_anchors_before_filter = int(
                np.count_nonzero(cluster_anchor_mask & fine_result.anchors["accepted_before_filter"].astype(bool))
            )
        else:
            cluster_anchors_before_filter = int(np.count_nonzero(cluster_anchor_mask))
        cluster_anchors_after_filter = int(
            np.count_nonzero(cluster_anchor_mask & fine_result.anchors["accepted"].astype(bool))
        )
    tissue_tab.dataframe(
        pd.DataFrame(
            [
                {
                    "total_geojson_points": int(len(geojson_array)),
                    "valid_geojson_points": n_valid_geojson,
                    "edge_candidate_geojson_points": n_edge_geojson,
                    "excluded_geojson_points": n_excluded_geojson,
                    "excluded_fraction": float(n_excluded_geojson / len(geojson_array)) if len(geojson_array) else np.nan,
                    "n_valid_used_for_anchors": n_valid_used_for_anchors,
                    "n_edge_used_for_anchors": n_edge_used_for_anchors,
                    "n_excluded_used_for_anchors": 0,
                    "valid_geojson_weight": valid_geojson_weight,
                    "edge_candidate_weight": edge_candidate_weight if use_edge_candidates_for_anchors else 0.0,
                    "total_he_points": int(len(attempted_points)),
                    "valid_he_points": n_valid_he,
                    "edge_candidate_he_points": n_edge_he,
                    "excluded_he_points": n_excluded_he,
                    "cluster_anchors_before_filtering": cluster_anchors_before_filter,
                    "cluster_anchors_after_filtering": cluster_anchors_after_filter,
                    "boundary_pinned_anchors": boundary_pinned_anchors,
                    "boundary_anchor_weight": boundary_anchor_weight,
                }
            ]
        ),
        use_container_width=True,
    )
    safety_tab.dataframe(
        pd.DataFrame(
            [
                {
                    "attempted_max_displacement": safety_metrics.get("attempted_max_displacement", fine_result.max_displacement),
                    "attempted_p95_displacement": safety_metrics.get("attempted_p95_displacement", np.nan),
                    "attempted_jacobian_min": safety_metrics.get("attempted_jacobian_min", fine_result.jacobian_min),
                    "attempted_jacobian_max": safety_metrics.get("attempted_jacobian_max", fine_result.jacobian_max),
                    "attempted_jacobian_median": safety_metrics.get("attempted_jacobian_median", np.nan),
                    "fraction_foldover_jacobian_le_0": safety_metrics.get("fraction_jacobian_foldover_le_0", np.nan),
                    "fraction_jacobian_below_min_limit": safety_metrics.get("fraction_jacobian_below_min_limit", np.nan),
                    "fraction_jacobian_above_max_limit": safety_metrics.get("fraction_jacobian_above_max_limit", np.nan),
                    "fine_status": status,
                    "rejection_reason": rejection_reason or "",
                }
            ]
        ),
        use_container_width=True,
    )

    def _metric_rows_for_group(group_name: str, fixed_subset: np.ndarray, moving_mask: np.ndarray) -> list[dict]:
        if len(fixed_subset) == 0 or not np.any(moving_mask):
            return []
        rows = []
        for stage, moving_subset in [
            ("affine", affine_points[moving_mask]),
            ("attempted_fine", attempted_points[moving_mask]),
            ("final_applied", final_points[moving_mask]),
        ]:
            values = point_bidirectional_distance_metrics(fixed_subset, moving_subset)
            rows.append(
                {
                    "point_group": group_name,
                    "stage": stage,
                    "symmetric_median": values["symmetric_median_distance"],
                    "he_to_geojson_median": values["he_to_geojson_median_distance"],
                    "geojson_to_he_median": values["geojson_to_he_median_distance"],
                    "mean": values["mean_distance"],
                    "he_to_geojson_within_3": values["he_to_geojson_within_3"],
                    "geojson_to_he_within_3": values["geojson_to_he_within_3"],
                    "he_to_geojson_within_5": values["he_to_geojson_within_5"],
                    "geojson_to_he_within_5": values["geojson_to_he_within_5"],
                    "he_to_geojson_within_10": values["he_to_geojson_within_10"],
                    "geojson_to_he_within_10": values["geojson_to_he_within_10"],
                }
            )
        return rows

    metric_rows = []
    metric_rows.extend(_metric_rows_for_group("all_points", geojson_array, np.ones(len(he_array), dtype=bool)))
    metric_rows.extend(
        _metric_rows_for_group("valid_region", geojson_array[geojson_classification == "valid"], he_classification == "valid")
    )
    metric_rows.extend(
        _metric_rows_for_group(
            "edge_candidate_region",
            geojson_array[geojson_classification == "edge_candidate"],
            he_classification == "edge_candidate",
        )
    )
    point_tab.dataframe(pd.DataFrame(metric_rows), use_container_width=True)

    point_tab.subheader(tr("点群registration結果", "Point registration result"))
    plot_left, plot_mid, plot_right = point_tab.columns(3)
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
            title=(
                f"Attempted fine alignment - {fine_alignment_method}"
                + (" (rejected)" if not fine_applied and fine_alignment_method != "off" else "")
            ),
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
            title=f"Final applied alignment - {applied_result_label}",
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
    overview_tab.subheader(tr("最終適用結果", "Final applied result"))
    overview_tab.pyplot(fine_figure, clear_figure=False)
    if density_flow_mode:
        density_flow_comparison_figure = visualize_density_flow_point_comparison(
            geojson_array,
            affine_points,
            attempted_points,
            final_points,
            title="Density-flow point registration: fixed / affine / attempted / applied",
            max_points=max_warped_overlay_points,
            invert_x_axis=invert_x_axis,
            invert_y_axis=invert_y_axis,
        )
        point_tab.subheader(tr("Density-flow点群4状態比較", "Density-flow four-state point comparison"))
        point_tab.caption(
            tr(
                "Fixed GeoJSONは移動しません。reject時のapplied HE点はaffine HE点と一致します。",
                "Fixed GeoJSON points never move. When rejected, applied HE points coincide with affine HE points.",
            )
        )
        point_tab.pyplot(density_flow_comparison_figure, clear_figure=False)
        point_tab.download_button(
            "Download density-flow four-state point comparison PNG",
            data=figure_to_png_bytes(density_flow_comparison_figure),
            file_name="density_flow_point_states.png",
            mime="image/png",
        )

    before_distances = point_nearest_distances(metric_fixed_points, metric_moving_affine_points)
    attempted_distances = point_nearest_distances(metric_fixed_points, metric_moving_attempted_points)
    after_distances = point_nearest_distances(metric_fixed_points, metric_moving_applied_points)
    hist_figure = visualize_distance_histogram(
        before_distances,
        after_distances,
        title="Valid-region nearest-neighbor distance before/after fine alignment",
        attempted_distances=attempted_distances,
    )
    point_tab.pyplot(hist_figure, clear_figure=False)
    point_tab.download_button(
        "Download distance histogram PNG",
        data=figure_to_png_bytes(hist_figure),
        file_name="fine_alignment_distance_histogram.png",
        mime="image/png",
    )

    residual_left, residual_right = point_tab.columns(2)
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
            final_points,
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
        anchor_tab.subheader(tr("Fine warp変位アンカー", "Fine warp displacement anchors"))
        cluster_diagnostic_columns = [
            "anchor_x",
            "anchor_y",
            "cluster_selection_mode",
            "fixed_cluster_point_count",
            "moving_cluster_point_count",
            "fixed_cluster_radius",
            "moving_cluster_radius",
            "mutual_matches_before",
            "mutual_matches_after",
            "score_before",
            "score_after",
            "score_improvement",
            "selected_dx",
            "selected_dy",
            "accepted",
            "rejection_reason",
        ]
        available_cluster_columns = [
            column for column in cluster_diagnostic_columns if column in fine_result.anchors.columns
        ]
        if available_cluster_columns and "anchor_type" in fine_result.anchors:
            cluster_diagnostics = fine_result.anchors.loc[
                fine_result.anchors["anchor_type"] == "cluster",
                available_cluster_columns,
            ]
            anchor_tab.subheader(tr("クラスタ構築診断", "Cluster construction diagnostics"))
            anchor_tab.dataframe(cluster_diagnostics, use_container_width=True, hide_index=True)
        if "anchor_type" in fine_result.anchors:
            anchor_type_summary = (
                fine_result.anchors.groupby(["anchor_type", "accepted"], dropna=False)
                .size()
                .reset_index(name="n_anchors")
            )
            anchor_tab.dataframe(anchor_type_summary, use_container_width=True)
            accepted_cluster_rows = fine_result.anchors[
                (fine_result.anchors["anchor_type"] == "cluster") & (fine_result.anchors["accepted"].astype(bool))
            ]
            if not accepted_cluster_rows.empty and "improvement" in accepted_cluster_rows:
                improved_mask = accepted_cluster_rows["improvement"] > 0
                anchor_local_metrics = pd.DataFrame(
                    [
                        {
                            "median_anchor_local_improvement": float(accepted_cluster_rows["improvement"].median()),
                            "fraction_anchors_improved": float(improved_mask.mean()),
                            "n_anchors_worsened": int(np.count_nonzero(accepted_cluster_rows["improvement"] < 0)),
                            "median_local_distance_before": float(
                                accepted_cluster_rows["median_distance_zero_shift"].median()
                            ),
                            "median_local_distance_after": float(
                                accepted_cluster_rows["median_distance_best_shift"].median()
                            ),
                            "median_fraction_within_threshold_before": float(
                                accepted_cluster_rows["fraction_within_threshold_zero_shift"].median()
                            ),
                            "median_fraction_within_threshold_after": float(
                                accepted_cluster_rows["fraction_within_threshold_best_shift"].median()
                            ),
                        }
                    ]
                )
                anchor_tab.subheader(tr("アンカー局所指標", "Anchor-local metrics"))
                anchor_tab.dataframe(anchor_local_metrics, use_container_width=True)
            boundary_pin_rows = fine_result.anchors[fine_result.anchors["anchor_type"] == "boundary_pin"]
            if not boundary_pin_rows.empty:
                with anchor_tab.expander(tr("境界ピンアンカー", "Boundary pinned anchors"), expanded=False):
                    st.dataframe(boundary_pin_rows, use_container_width=True)
        anchor_tab.dataframe(fine_result.anchors, use_container_width=True)
        anchor_filename = (
            "cluster_anchor_candidates.csv"
            if fine_alignment_method == "cluster-anchor"
            else "fine_warp_displacement_anchors.csv"
        )
        anchor_tab.download_button(
            "Download fine warp anchors CSV",
            data=fine_result.anchors.to_csv(index=False).encode("utf-8"),
            file_name=anchor_filename,
            mime="text/csv",
        )

        anchor_left, anchor_right = anchor_tab.columns(2)
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
    else:
        anchor_tab.info(tr("この結果にはアンカー診断データがありません。", "No anchor diagnostics are available for this result."))

    safety_tab.subheader(tr("変位場", "Displacement field"))
    field_left, field_right = safety_tab.columns(2)
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
            final_displacement_x,
            final_displacement_y,
            title=f"Final applied displacement field ({applied_result_label})",
        )
        st.pyplot(applied_field_figure, clear_figure=False)
        st.download_button(
            "Download applied displacement field PNG",
            data=figure_to_png_bytes(applied_field_figure),
            file_name="applied_fine_displacement_vector_field.png",
            mime="image/png",
        )

    attempted_jacobian = _field_jacobian(attempted_displacement_x, attempted_displacement_y, fine_result.grid_spacing)
    applied_jacobian = _field_jacobian(final_displacement_x, final_displacement_y, fine_result.grid_spacing)
    safety_tab.subheader("Jacobian QC")
    safety_tab.dataframe(
        pd.DataFrame(
            [
                _jacobian_summary_row("attempted_fine", attempted_jacobian),
                _jacobian_summary_row("applied_fine", applied_jacobian),
            ]
        ),
        use_container_width=True,
    )
    jac_left, jac_right = safety_tab.columns(2)
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

    affine_warped_he_image = None
    affine_warped_he_metadata = None
    warped_he_image = None
    warped_he_metadata = None
    attempted_warped_he_image = None
    attempted_warped_he_metadata = None
    if he_image is not None:
        try:
            affine_warped_he_image, affine_warped_he_metadata = warp_he_image_to_world(
                he_image,
                affine_result,
                None,
                output_pixel_size_um=warped_he_pixel_size,
                output_origin=warped_he_output_origin,
                bounds=fine_result.bounds,
            )
            if density_flow_mode:
                density_flow_images = density_flow_image_outputs(
                    affine_warped_he_image,
                    affine_warped_he_metadata,
                    fine_result,
                )
                attempted_warped_he_image = density_flow_images["attempted"]
                attempted_warped_he_metadata = dict(affine_warped_he_metadata)
                warped_he_image = density_flow_images["final"]
                warped_he_metadata = dict(affine_warped_he_metadata)
            elif image_fine_applied:
                warped_he_image, warped_he_metadata = warp_he_image_to_world(
                    he_image,
                    affine_result,
                    final_fine_result,
                    output_pixel_size_um=warped_he_pixel_size,
                    output_origin=warped_he_output_origin,
                    bounds=fine_result.bounds,
                )
            else:
                warped_he_image = affine_warped_he_image.copy()
                warped_he_metadata = dict(affine_warped_he_metadata)
        except ValueError as exc:
            image_tab.warning(f"Could not warp HE image: {exc}")
        if (
            fine_alignment_method != "off"
            and not density_flow_mode
            and attempted_displacement_x_value is not None
        ):
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
                    bounds=fine_result.bounds,
                )
            except ValueError as exc:
                image_tab.warning(f"Could not render attempted warp preview: {exc}")

    if warped_he_image is not None:
        image_tab.subheader(tr("Warp済みHE画像", "Warped HE image"))
        image_tab.info(tr("出力原点は画像の向きだけを制御し、位置合わせ品質には影響しません。", "Warped HE output origin controls exported image orientation, not registration quality."))
        if density_flow_mode:
            image_tab.subheader(tr("Density-flow画像出力状態", "Density-flow image output states"))
            image_tab.dataframe(
                pd.DataFrame(
                    [
                        {"image_output": "Affine-only HE image", "field": "affine", "purpose": "baseline"},
                        {"image_output": "Attempted density-flow warped HE image", "field": "attempted", "purpose": "QC even when rejected"},
                        {
                            "image_output": "Final applied HE image",
                            "field": "density-flow" if fine_applied else "affine fallback",
                            "purpose": "safety-gated result",
                        },
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        if density_flow_mode and fine_applied:
            image_tab.success(
                tr(
                    "点群結果と最終HE画像には、safety checkを通過したdensity-flow fieldを適用しています。",
                    "The safety-approved density-flow field is applied to both the point result and final HE image.",
                )
            )
        elif density_flow_mode:
            image_tab.warning(
                tr(
                    "Density-flowはrejectされました。attempted画像はQC用に表示し、最終HE画像はaffine-onlyです。",
                    "Density-flow was rejected. The attempted image remains visible for QC; the final HE image is affine-only.",
                )
            )
        elif fine_applied:
            image_tab.success("Warped HE image uses: affine + applied fine warp")
        else:
            image_tab.warning("Fine warp was rejected or disabled; warped HE image is affine-only.")
            if attempted_warped_he_image is not None:
                image_tab.info("Attempted warp preview shows the rejected candidate deformation for QC only.")
        warped_image_height, warped_image_width = warped_he_image.shape[:2]
        geojson_pixels = world_points_to_warped_image_pixels(geojson_array, warped_he_metadata)
        he_pixels = world_points_to_warped_image_pixels(final_points, warped_he_metadata)

        image_tab.subheader(tr("Warp後の画素座標診断", "Warped pixel-coordinate diagnostics"))
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
        image_tab.dataframe(pixel_diagnostics, use_container_width=True)

        if affine_warped_he_image is not None:
            image_tab.subheader(tr("Affine-only HE画像", "Affine-only HE image"))
            image_tab.image(
                affine_warped_he_image,
                caption="Affine-only warped HE image",
            )
            image_tab.download_button(
                "Download affine-only warped HE image PNG",
                data=array_to_png_bytes(affine_warped_he_image),
                file_name="affine_only_warped_he_image.png",
                mime="image/png",
            )

        pixel_scatter_figure = visualize_warped_pixel_point_scatter(
            geojson_pixels,
            he_pixels,
            image_width=warped_image_width,
            image_height=warped_image_height,
            title="Warped pixel coordinates without HE image background",
            max_points=max_warped_overlay_points,
        )
        image_tab.pyplot(pixel_scatter_figure, clear_figure=False)
        image_tab.download_button(
            "Download warped pixel scatter PNG",
            data=figure_to_png_bytes(pixel_scatter_figure),
            file_name="warped_pixel_points_scatter.png",
            mime="image/png",
        )

        image_only_col, overlay_col = image_tab.columns(2)
        with image_only_col:
            st.subheader(tr("最終適用HE画像", "Final applied HE image warp"))
            st.image(
                warped_he_image,
                caption=(
                    f"Final applied HE image - {image_applied_result_label} "
                    f"| pixel_size={warped_he_metadata['output_pixel_size_um']} um"
                ),
            )
        image_tab.download_button(
            "Download warped HE image PNG",
            data=array_to_png_bytes(warped_he_image),
            file_name="warped_he_world_um.png",
            mime="image/png",
        )
        if attempted_warped_he_image is not None:
            attempted_geojson_pixels = world_points_to_warped_image_pixels(geojson_array, attempted_warped_he_metadata)
            attempted_he_pixels = world_points_to_warped_image_pixels(attempted_points, attempted_warped_he_metadata)
            attempted_boundary_pin_pixels = (
                world_points_to_warped_image_pixels(boundary_anchor_points, attempted_warped_he_metadata)
                if show_boundary_pin_anchors and boundary_anchor_points is not None and len(boundary_anchor_points)
                else None
            )
            image_tab.subheader(tr("Attempted HE画像warp", "Attempted HE image warp"))
            image_tab.image(
                attempted_warped_he_image,
                caption=(
                    (
                        "Attempted density-flow warped HE image"
                        if density_flow_mode
                        else "Attempted fine-warp HE image"
                    )
                    + (" - rejected / unsafe" if not fine_applied else "")
                ),
            )
            if density_flow_mode and affine_warped_he_image is not None:
                difference_figure = visualize_absolute_image_difference(
                    affine_warped_he_image,
                    attempted_warped_he_image,
                    title="Absolute pixel difference: attempted density-flow minus affine-only",
                )
                image_tab.pyplot(difference_figure, clear_figure=False)
                image_tab.caption(
                    tr(
                        "画像差分はラスターが変化した場所を示すだけで、非線形補正の量はlocal residual fieldで判定します。",
                        "Raster difference only shows where pixels changed; nonlinear correction is assessed from the local residual field.",
                    )
                )
                image_tab.download_button(
                    "Download density-flow absolute pixel-difference PNG",
                    data=figure_to_png_bytes(difference_figure),
                    file_name="density_flow_absolute_pixel_difference.png",
                    mime="image/png",
                )
            image_tab.download_button(
                "Download attempted warped HE preview PNG",
                data=array_to_png_bytes(attempted_warped_he_image),
                file_name=(
                    "attempted_density_flow_warped_he.png"
                    if density_flow_mode
                    else "attempted_warped_he_preview.png"
                ),
                mime="image/png",
            )
            attempted_overlay_figure = visualize_warped_he_point_overlay(
                attempted_warped_he_image,
                attempted_geojson_pixels,
                attempted_he_pixels,
                title=(
                    "Attempted density-flow HE image with GeoJSON and HE nuclei overlay"
                    if density_flow_mode
                    else "Attempted HE warp preview with GeoJSON and HE nuclei overlay"
                ),
                max_points=max_warped_overlay_points,
                geojson_classifications=geojson_classification,
                show_excluded_geojson=show_excluded_geojson_points,
                show_edge_candidate_geojson=show_edge_candidate_geojson_points,
                boundary_pin_pixels=attempted_boundary_pin_pixels,
            )
            image_tab.pyplot(attempted_overlay_figure, clear_figure=False)
            image_tab.download_button(
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
            safety_tab.pyplot(attempted_grid_figure, clear_figure=False)
            safety_tab.download_button(
                "Download attempted warp grid QC PNG",
                data=figure_to_png_bytes(attempted_grid_figure),
                file_name="attempted_warp_grid_qc.png",
                mime="image/png",
            )

        with overlay_col:
            boundary_pin_pixels = (
                world_points_to_warped_image_pixels(boundary_anchor_points, warped_he_metadata)
                if show_boundary_pin_anchors and boundary_anchor_points is not None and len(boundary_anchor_points)
                else None
            )
            warped_overlay_figure = visualize_warped_he_point_overlay(
                warped_he_image,
                geojson_pixels,
                he_pixels,
                title="Warped HE image with registered nuclei overlay",
                max_points=max_warped_overlay_points,
                geojson_classifications=geojson_classification,
                show_excluded_geojson=show_excluded_geojson_points,
                show_edge_candidate_geojson=show_edge_candidate_geojson_points,
                boundary_pin_pixels=boundary_pin_pixels,
            )
            st.pyplot(warped_overlay_figure, clear_figure=False)
        image_tab.download_button(
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
            fine_result=final_fine_result if image_fine_applied else None,
        )
        applied_grid_figure = visualize_warp_grid_overlay(
            warped_he_image,
            applied_before_grid_lines,
            applied_after_grid_lines,
            title=f"Final applied warp grid QC ({image_applied_result_label}): cyan before / orange after",
        )
        safety_tab.pyplot(applied_grid_figure, clear_figure=False)
        safety_tab.download_button(
            "Download applied warp grid QC PNG",
            data=figure_to_png_bytes(applied_grid_figure),
            file_name="applied_warp_grid_qc.png",
            mime="image/png",
        )

    if he_image is None:
        image_tab.info(tr("HE画像が未入力のため、画像warpは表示しません。点群結果は他のタブで確認できます。", "No HE image was uploaded. Point results remain available in the other tabs."))
    elif warped_he_image is None:
        image_tab.warning(tr("HE画像warpを生成できませんでした。点群結果と診断は引き続き利用できます。", "The warped HE image could not be generated. Point results and diagnostics remain available."))

    downloads_tab.subheader(tr("出力ファイル", "Exports"))
    downloads_tab.download_button(
        "Download affine-transformed HE nuclei CSV",
        data=transformed_affine_points.to_csv(index=False).encode("utf-8"),
        file_name="affine_transformed_he_nuclei.csv",
        mime="text/csv",
    )
    if fine_alignment_method != "off" and attempted_points_value is not None:
        downloads_tab.download_button(
            "Download attempted fine HE nuclei CSV",
            data=transformed_attempted_points.to_csv(index=False).encode("utf-8"),
            file_name=(
                "attempted_fine_he_points.csv"
                if fine_applied
                else "attempted_fine_rejected_he_points.csv"
            ),
            mime="text/csv",
        )
    downloads_tab.download_button(
        f"Download final applied HE nuclei CSV ({applied_result_label})",
        data=transformed_fine_points.to_csv(index=False).encode("utf-8"),
        file_name="final_applied_he_nuclei.csv",
        mime="text/csv",
    )
    if affine_warped_he_image is not None:
        downloads_tab.download_button(
            "Download affine-only HE image PNG",
            data=array_to_png_bytes(affine_warped_he_image),
            file_name="affine_only_warped_he_image.png",
            mime="image/png",
        )
    if attempted_warped_he_image is not None:
        downloads_tab.download_button(
            "Download attempted fine HE image PNG",
            data=array_to_png_bytes(attempted_warped_he_image),
            file_name=(
                "attempted_fine_he_image.png"
                if fine_applied
                else "attempted_fine_rejected_he_image.png"
            ),
            mime="image/png",
        )
    if warped_he_image is not None:
        downloads_tab.download_button(
            f"Download final applied HE image PNG ({image_applied_result_label})",
            data=array_to_png_bytes(warped_he_image),
            file_name="final_applied_warped_he_image.png",
            mime="image/png",
        )
    if fine_result.anchors is not None:
        downloads_tab.download_button(
            "Download anchor diagnostics CSV",
            data=fine_result.anchors.to_csv(index=False).encode("utf-8"),
            file_name="fine_warp_anchor_diagnostics.csv",
            mime="text/csv",
        )
    parameters = {
        "fine_alignment_method": fine_alignment_method,
        "fine_applied": fine_applied,
        "applied_result_label": applied_result_label,
        "density_flow_experimental_raster_warp": density_flow_mode,
        "density_flow_pixel_size_um": density_flow_pixel_size,
        "density_flow_blur_scales_px": density_flow_blur_scales_text,
        "density_flow_optimization_levels": density_flow_levels,
        "density_flow_iterations_per_level": density_flow_iterations,
        "density_flow_learning_rate": density_flow_learning_rate,
        "density_flow_update_smoothing_sigma_px": density_flow_update_smoothing,
        "density_flow_smoothness_weight": density_flow_smoothness_weight,
        "density_flow_magnitude_weight": density_flow_magnitude_weight,
        "density_flow_jacobian_barrier_weight": density_flow_jacobian_weight,
        "density_flow_tissue_boundary_weight": density_flow_boundary_weight,
        "density_flow_inverse_consistency_weight": density_flow_inverse_weight,
        "density_flow_global_translation_initialization": density_flow_global_initialization,
        "density_flow_detect_axis_reversal": density_flow_detect_axis_reversal,
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
        "cluster_selection_mode": cluster_selection_mode,
        "cluster_target_points_per_cluster": target_points_per_cluster,
        "cluster_min_points_per_cluster": min_points_per_cluster,
        "cluster_max_radius_um": max_cluster_radius_um,
        "cluster_moving_candidate_pool_ratio": moving_candidate_pool_ratio,
        "cluster_match_threshold_um": cluster_match_threshold,
        "cluster_min_improvement_um": cluster_min_improvement,
        "cluster_max_shift_um": cluster_max_shift,
        "cluster_min_accepted_anchors": cluster_min_anchors,
        "use_edge_candidates_for_anchors": use_edge_candidates_for_anchors,
        "valid_geojson_weight": valid_geojson_weight,
        "edge_candidate_weight": edge_candidate_weight if use_edge_candidates_for_anchors else 0.0,
        "excluded_geojson_weight": 0.0,
        "cluster_interpolation": cluster_interpolation,
        "local_support_radius_um": local_support_radius,
        "control_grid_spacing_um": control_grid_spacing,
        "cluster_regularization": cluster_regularization,
        "tissue_mask_threshold": tissue_mask_threshold,
        "edge_margin_px": edge_margin,
        "max_final_displacement_um": max_final_displacement_um,
        "jacobian_min_limit": jacobian_min_limit,
        "jacobian_max_limit": jacobian_max_limit,
        "enable_displacement_p95_limit": enable_displacement_p95_limit,
        "displacement_p95_limit_um": displacement_p95_limit_um if enable_displacement_p95_limit else None,
        "enable_boundary_pinning": enable_boundary_pinning,
        "include_image_border_pins": include_image_border_pins,
        "include_tissue_boundary_pins": include_tissue_boundary_pins,
        "boundary_anchor_spacing_px": boundary_anchor_spacing,
        "boundary_anchor_weight": boundary_anchor_weight,
        "registration_display_origin": registration_display_origin,
        "warped_he_output_origin": warped_he_output_origin,
        "flip_mode": flip_mode,
        "warped_he_output_pixel_size_um": warped_he_pixel_size,
        "max_warped_overlay_points": max_warped_overlay_points,
        "show_excluded_geojson_points": show_excluded_geojson_points,
        "show_edge_candidate_geojson_points": show_edge_candidate_geojson_points,
        "show_boundary_pin_anchors": show_boundary_pin_anchors,
    }
    downloads_tab.download_button(
        "Download HE-GeoJSON transform summary",
        data=_he_geojson_summary_to_json(affine_result, final_fine_result, parameters, warped_he_metadata),
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
