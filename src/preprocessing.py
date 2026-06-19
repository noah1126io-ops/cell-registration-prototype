from __future__ import annotations

import numpy as np


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image intensity to the 0-1 range for preview or downstream features."""
    image = np.asarray(image)
    min_value = float(np.min(image))
    max_value = float(np.max(image))

    if max_value == min_value:
        return np.zeros_like(image, dtype=np.float32)

    return ((image - min_value) / (max_value - min_value)).astype(np.float32)


# TODO: Add stain normalization, denoising, and scale handling for serial sections.
