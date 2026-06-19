from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image


def matches_to_csv(matches: pd.DataFrame) -> bytes:
    """Serialize a future match table to CSV bytes."""
    return matches.to_csv(index=False).encode("utf-8")


def array_to_png_bytes(image: np.ndarray) -> bytes:
    """Convert a NumPy image array to PNG bytes."""
    image = np.asarray(image)

    if np.issubdtype(image.dtype, np.floating):
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        min_value = float(image.min())
        max_value = float(image.max())
        if max_value > min_value:
            image = (image - min_value) / (max_value - min_value)
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        pil_image = Image.fromarray(image, mode="L")
    elif image.ndim == 3 and image.shape[2] == 3:
        pil_image = Image.fromarray(image, mode="RGB")
    elif image.ndim == 3 and image.shape[2] == 4:
        pil_image = Image.fromarray(image, mode="RGBA")
    else:
        raise ValueError("PNG export expects a 2D grayscale or 3D RGB/RGBA array.")

    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def figure_to_png_bytes(figure) -> bytes:
    """Convert a matplotlib Figure to PNG bytes."""
    buffer = BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
    return buffer.getvalue()


# TODO: Add export for batch transforms and rendered quality-control reports.
