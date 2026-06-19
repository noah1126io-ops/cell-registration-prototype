from __future__ import annotations

from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image

from src.density import create_density_map
from src.features import extract_cell_features
from src.io_utils import read_uploaded_image, read_uploaded_mask
from src.matching import match_cells
from src.visualization import colorize_label_image


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
    density_map = np.asarray(density_map, dtype=np.float32)
    max_value = float(density_map.max())
    if max_value > 0:
        preview = density_map / max_value
    else:
        preview = density_map

    image = Image.fromarray((preview * 255).astype(np.uint8), mode="L")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


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


def show_density_map(title: str, features, image_shape, sigma: float, filename: str) -> None:
    st.subheader(title)

    if features is None or image_shape is None:
        st.info("Upload a mask to create a density map.")
        return

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
        return

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
        return
    st.dataframe(matches, use_container_width=True)
    st.download_button(
        "Download CSV",
        data=matches.to_csv(index=False).encode("utf-8"),
        file_name="cell_correspondence.csv",
        mime="text/csv",
        key="download-cell-correspondence.csv",
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
        show_density_map(
            "Fixed density map",
            fixed_features,
            None if fixed_mask_array is None else fixed_mask_array.shape,
            density_sigma,
            "fixed_density_map.png",
        )
    with density_right:
        show_density_map(
            "Moving density map",
            moving_features,
            None if moving_mask_array is None else moving_mask_array.shape,
            density_sigma,
            "moving_density_map.png",
        )

    st.divider()
    show_cell_correspondence(
        fixed_features,
        moving_features,
        max_distance=max_distance,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        max_score=max_score,
    )

    # TODO: Add Cellpose execution for optional mask generation.
    # TODO: Add multi-scale density map presets for registration experiments.
    # TODO: Add image registration and transformed overlay visualization.
    # TODO: Add export of matched cells, transforms, and preview images.


if __name__ == "__main__":
    main()
