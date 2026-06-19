from __future__ import annotations

import streamlit as st

from src.io_utils import read_uploaded_image, read_uploaded_mask
from src.visualization import colorize_label_image


st.set_page_config(
    page_title="Cell Registration Prototype",
    layout="wide",
)


def show_uploaded_image(title: str, uploaded_file, *, is_mask: bool = False) -> None:
    st.subheader(title)

    if uploaded_file is None:
        st.info("Upload a file to preview it.")
        return

    try:
        if is_mask:
            image = read_uploaded_mask(uploaded_file)
            preview = colorize_label_image(image)
            st.image(preview, caption=f"{uploaded_file.name} | shape={image.shape} | dtype={image.dtype}")
            st.caption(f"Label range: {int(image.min())} - {int(image.max())}")
        else:
            image = read_uploaded_image(uploaded_file)
            st.image(image, caption=f"{uploaded_file.name} | shape={image.shape} | dtype={image.dtype}")
    except Exception as exc:  # pragma: no cover - Streamlit UI feedback
        st.error(f"Could not read {uploaded_file.name}: {exc}")


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
    st.sidebar.info("Masks are interpreted as integer label images.")

    image_left, image_right = st.columns(2)
    with image_left:
        show_uploaded_image("Fixed image", fixed_image)
    with image_right:
        show_uploaded_image("Moving image", moving_image)

    mask_left, mask_right = st.columns(2)
    with mask_left:
        show_uploaded_image("Fixed mask", fixed_mask, is_mask=True)
    with mask_right:
        show_uploaded_image("Moving mask", moving_mask, is_mask=True)

    # TODO: Add Cellpose execution for optional mask generation.
    # TODO: Add cell feature extraction from integer label masks.
    # TODO: Add cell correspondence estimation between fixed and moving masks.
    # TODO: Add image registration and transformed overlay visualization.
    # TODO: Add export of matched cells, transforms, and preview images.


if __name__ == "__main__":
    main()
