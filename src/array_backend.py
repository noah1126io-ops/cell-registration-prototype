from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import ndimage as scipy_ndimage


DeviceName = Literal["cpu", "cuda", "auto"]


class BackendUnavailableError(RuntimeError):
    """Raised when an explicitly requested numerical backend is unavailable."""


@dataclass(frozen=True)
class ArrayBackend:
    """Small compatibility surface for future CPU/CUDA grid kernels."""

    name: Literal["cpu", "cuda"]
    xp: Any
    ndimage: Any
    dtype: Any

    def asarray(self, values: Any):
        return self.xp.asarray(values, dtype=self.dtype)

    def to_cpu(self, values: Any) -> np.ndarray:
        if self.name == "cpu":
            return np.asarray(values)
        return self.xp.asnumpy(values)

    def gaussian_filter(self, values: Any, *, sigma: float, mode: str = "reflect"):
        return self.ndimage.gaussian_filter(values, sigma=sigma, mode=mode)

    def map_coordinates(self, values: Any, coordinates: Any, **kwargs):
        normalized = coordinates
        if self.name == "cuda" and isinstance(coordinates, (list, tuple)):
            normalized = self.xp.stack(coordinates)
        return self.ndimage.map_coordinates(values, normalized, **kwargs)

    def gradient(self, values: Any, *spacing: float):
        return self.xp.gradient(values, *spacing)

    def synchronize(self) -> None:
        if self.name == "cuda":
            self.xp.cuda.get_current_stream().synchronize()


def _load_cupy():
    try:
        import cupy as cp
        from cupyx.scipy import ndimage as cupy_ndimage
    except ImportError as exc:
        raise BackendUnavailableError(
            "CUDA backend requires the optional cupy-cuda13x package."
        ) from exc
    return cp, cupy_ndimage


def cuda_available() -> bool:
    try:
        cp, _ = _load_cupy()
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def get_array_backend(device: DeviceName = "cpu", *, dtype: str = "float64") -> ArrayBackend:
    """Resolve a float64 CPU or CUDA backend without importing CuPy for CPU use."""
    requested = str(device).lower()
    if requested not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be one of: cpu, cuda, auto")
    if dtype != "float64":
        raise ValueError("Only float64 is supported until CPU/CUDA parity is established.")
    if requested == "auto":
        requested = "cuda" if cuda_available() else "cpu"
    if requested == "cpu":
        return ArrayBackend(name="cpu", xp=np, ndimage=scipy_ndimage, dtype=np.float64)

    cp, cupy_ndimage = _load_cupy()
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise BackendUnavailableError("CUDA backend requested but no CUDA device is visible.")
    except BackendUnavailableError:
        raise
    except Exception as exc:
        raise BackendUnavailableError("CUDA backend requested but CUDA initialization failed.") from exc
    return ArrayBackend(name="cuda", xp=cp, ndimage=cupy_ndimage, dtype=cp.float64)
