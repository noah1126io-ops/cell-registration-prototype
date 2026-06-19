from __future__ import annotations

import json

import numpy as np
import streamlit as st

from src.density import create_density_map
from src.export import array_to_png_bytes, figure_to_png_bytes
from src.features import extract_cell_features
from src.io_utils import read_uploaded_image, read_uploaded_mask
from src.matching import match_cells
from src.registration import (
    estimate_affine_transform,
    transform_cell_centroids,
    warp_image,
    warp_mask,
)
from src.visualization import colorize_label_image, visualize_cell_matches


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
        file_name="cell_correspondence.csv",
        mime="text/csv",
        key="download-cell-correspondence.csv",
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

    if fixed_image is None or fixed_features is None or moving_features is None or matches is None:
        st.info("Upload fixed image and masks, then create correspondence results to visualize matches.")
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


def main() -> None:
    st.title("Cell Registration Prototype")
    st.caption("Research prototype only. Not for diagnostic use.")

    st.sidebar.header("Input files")
    file_types = ["png", "jpg", "jpeg", "tif", "tiff"]

    fixed_image = st.sidebar.file_uploader("Fixed image", type=file_types)
    moving_image = st.sidebar.file_uploader("Moving image", type=file_types)
    fixed_mask = st.sidebar.file_uploader("Fixed mask", type=file_types)
    moving_mask = st.sidebar.file_uploader("Moving mask", type=file_types)

    st.sidebar.divider()
    density_sigma = st.sidebar.slider(
        "Density map sigma",
        min_value=1.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
    )
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
    st.sidebar.divider()
    st.sidebar.info("Masks are interpreted as integer label images.")

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

    # TODO: Add Cellpose execution for optional mask generation.
    # TODO: Add GeoJSON nuclei input support for HE-to-GeoJSON alignment mode.
    # TODO: Add HE nuclei .npy input support for precomputed StarDist centers.
    # TODO: Add HE-to-GeoJSON alignment mode separate from mask-to-mask registration.
    # TODO: Add world-um coordinate transform utilities for GeoJSON alignment.
    # TODO: Add Y-flip handling for fluorescence GeoJSON world coordinates.
    # TODO: Add Jacobian quality check for fine center-snap warp validation.
    # TODO: Add multi-scale density map presets for registration experiments.
    # TODO: Add non-rigid registration after affine registration is validated.
    # TODO: Add batch export of matched cells, transforms, and preview images.


if __name__ == "__main__":
    main()
