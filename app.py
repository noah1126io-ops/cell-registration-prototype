from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.density import create_density_map
from src.export import array_to_png_bytes, figure_to_png_bytes
from src.features import extract_cell_features, point_features_to_cell_features
from src.geojson_utils import load_geojson_centroids
from src.io_utils import read_uploaded_image, read_uploaded_mask
from src.matching import match_cells
from src.point_io import load_csv_points, load_npy_centers
from src.pointset_registration import (
    estimate_affine_with_y_flip,
    FineWarpResult,
    fine_center_snap_warp,
    local_translation_fine_warp,
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
    visualize_point_sets,
    visualize_translation_anchors,
    visualize_warped_he_point_overlay,
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


def _he_geojson_summary_to_json(affine_result, fine_result, parameters: dict, warp_metadata: dict | None = None) -> bytes:
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
        ["local translation field", "center-snap", "off"],
        index=0,
        help="Local translation field estimates patch-wise shifts from density maps; center-snap uses matched nuclei pairs.",
    )
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
    st.subheader("Local translation field")
    local_left, local_mid, local_right = st.columns(3)
    with local_left:
        local_density_sigma = st.number_input("Density sigma (um)", min_value=0.1, value=3.0, step=0.5)
        local_density_pixel_size = st.number_input("Density pixel size (um)", min_value=0.1, value=1.0, step=0.5)
        local_point_weight = st.number_input("Point weight", min_value=0.1, value=1.0, step=0.5)
    with local_mid:
        local_grid_spacing = st.number_input("Local grid spacing (um)", min_value=1.0, value=60.0, step=5.0)
        local_patch_radius = st.number_input("Patch radius (um)", min_value=1.0, value=25.0, step=5.0)
        local_search_radius = st.number_input("Search radius (um)", min_value=1.0, value=12.0, step=2.0)
    with local_right:
        local_min_correlation = st.slider("Min local correlation", 0.0, 1.0, 0.25, 0.05)
        local_max_shift = st.number_input("Max local shift (um)", min_value=1.0, value=20.0, step=5.0)
        local_min_anchors = st.number_input("Min accepted anchors", min_value=1, value=8, step=1)
    local_outlier_percentile = st.slider("Anchor outlier percentile", 50.0, 100.0, 95.0, 1.0)
    local_neighbor_radius = st.number_input("Neighbor consistency radius (um)", min_value=1.0, value=120.0, step=10.0)
    local_smoothing = st.number_input("RBF smoothing", min_value=0.0, value=10.0, step=1.0)
    local_kernel = st.selectbox("RBF kernel", ["thin_plate_spline", "linear", "cubic", "quintic"], index=0)
    local_neighbors = st.number_input("RBF neighbors", min_value=0, value=50, step=5)
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
        if fine_alignment_method == "local translation field":
            fine_result = local_translation_fine_warp(
                geojson_array,
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
                geojson_array,
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

    plot_left, plot_right = st.columns(2)
    geojson_features = _cell_features_from_points(geojson_points)
    affine_features = _cell_features_from_points(transformed_affine_points)
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
    with plot_right:
        fine_figure = visualize_point_sets(
            geojson_features,
            fine_features,
            title=f"Fine aligned HE centers vs GeoJSON centroids ({fine_alignment_method})",
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

    before_distances = point_nearest_distances(geojson_array, affine_result.transformed_points)
    after_distances = point_nearest_distances(geojson_array, fine_result.transformed_points)
    hist_figure = visualize_distance_histogram(
        before_distances,
        after_distances,
        title="Nearest-neighbor distance before/after fine alignment",
    )
    st.pyplot(hist_figure, clear_figure=False)
    st.download_button(
        "Download distance histogram PNG",
        data=figure_to_png_bytes(hist_figure),
        file_name="fine_alignment_distance_histogram.png",
        mime="image/png",
    )

    if fine_result.anchors is not None:
        st.subheader("Local translation anchors")
        st.dataframe(fine_result.anchors, use_container_width=True)
        st.download_button(
            "Download local translation anchors CSV",
            data=fine_result.anchors.to_csv(index=False).encode("utf-8"),
            file_name="local_translation_anchors.csv",
            mime="text/csv",
        )

        anchor_left, anchor_right = st.columns(2)
        with anchor_left:
            anchor_figure = visualize_translation_anchors(
                fine_result.anchors,
                title="Accepted and rejected local translation anchors",
            )
            st.pyplot(anchor_figure, clear_figure=False)
            st.download_button(
                "Download anchor vectors PNG",
                data=figure_to_png_bytes(anchor_figure),
                file_name="local_translation_anchor_vectors.png",
                mime="image/png",
            )
        with anchor_right:
            corr_figure = visualize_anchor_correlation_heatmap(
                fine_result.anchors,
                title="Local translation anchor correlation",
            )
            st.pyplot(corr_figure, clear_figure=False)
            st.download_button(
                "Download correlation heatmap PNG",
                data=figure_to_png_bytes(corr_figure),
                file_name="local_translation_correlation_heatmap.png",
                mime="image/png",
            )

    field_figure = visualize_displacement_field(
        fine_result.grid_x,
        fine_result.grid_y,
        fine_result.displacement_x,
        fine_result.displacement_y,
        title="Fine displacement vector field",
    )
    st.pyplot(field_figure, clear_figure=False)
    st.download_button(
        "Download displacement field PNG",
        data=figure_to_png_bytes(field_figure),
        file_name="fine_displacement_vector_field.png",
        mime="image/png",
    )

    warped_he_image = None
    warped_he_metadata = None
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

    if warped_he_image is not None:
        st.subheader("Warped HE image")
        st.image(
            warped_he_image,
            caption=(
                "HE image warped into GeoJSON world-um coordinates "
                f"| pixel_size={warped_he_metadata['output_pixel_size_um']} um"
            ),
        )
        st.download_button(
            "Download warped HE image PNG",
            data=array_to_png_bytes(warped_he_image),
            file_name="warped_he_world_um.png",
            mime="image/png",
        )

        geojson_pixels = world_points_to_warped_image_pixels(geojson_array, warped_he_metadata)
        he_pixels = world_points_to_warped_image_pixels(fine_result.transformed_points, warped_he_metadata)
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

    st.subheader("Exports")
    st.download_button(
        "Download transformed HE centers CSV",
        data=transformed_fine_points.to_csv(index=False).encode("utf-8"),
        file_name="he_centers_transformed_world_um.csv",
        mime="text/csv",
    )
    parameters = {
        "fine_alignment_method": fine_alignment_method,
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
