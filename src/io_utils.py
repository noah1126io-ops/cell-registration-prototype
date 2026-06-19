from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
import tifffile
from PIL import Image


TIFF_SUFFIXES = {".tif", ".tiff"}


def _read_bytes(uploaded_file: BinaryIO) -> bytes:
    uploaded_file.seek(0)
    return uploaded_file.read()


def _suffix(uploaded_file: BinaryIO) -> str:
    name = getattr(uploaded_file, "name", "")
    return Path(name).suffix.lower()


def read_uploaded_image(uploaded_file: BinaryIO) -> np.ndarray:
    """Read an uploaded microscopy image as a NumPy array."""
    data = _read_bytes(uploaded_file)

    if _suffix(uploaded_file) in TIFF_SUFFIXES:
        return np.asarray(tifffile.imread(BytesIO(data)))

    with Image.open(BytesIO(data)) as image:
        return np.asarray(image)


def read_uploaded_mask(uploaded_file: BinaryIO) -> np.ndarray:
    """Read an uploaded segmentation mask as an integer label image."""
    image = read_uploaded_image(uploaded_file)

    if image.ndim == 3:
        image = image[..., 0]

    if not np.issubdtype(image.dtype, np.integer):
        image = np.rint(image).astype(np.int32)

    return image


# TODO: Add validation for paired image/mask dimensions and supported bit depths.
# TODO: Add optional loading from local paths for batch research workflows.
