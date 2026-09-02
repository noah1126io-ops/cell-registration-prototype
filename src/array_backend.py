from __future__ import annotations

from dataclasses import dataclass
import os
import platform
import subprocess
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

    def scalar(self, value: Any):
        """Cross the device boundary for one scalar, never an entire grid."""
        return value.item() if hasattr(value, "item") else value

    def gaussian_filter(self, values: Any, *, sigma: float, mode: str = "reflect"):
        return self.ndimage.gaussian_filter(values, sigma=sigma, mode=mode)

    def map_coordinates(self, values: Any, coordinates: Any, **kwargs):
        normalized = coordinates
        if self.name == "cuda" and isinstance(coordinates, (list, tuple)):
            normalized = self.xp.stack(coordinates)
        return self.ndimage.map_coordinates(values, normalized, **kwargs)

    def gradient(self, values: Any, *spacing: float):
        return self.xp.gradient(values, *spacing)

    def laplace(self, values: Any, *, mode: str = "reflect"):
        return self.ndimage.laplace(values, mode=mode)

    def synchronize(self) -> None:
        if self.name == "cuda":
            self.xp.cuda.get_current_stream().synchronize()

    def provenance(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "compute_backend": self.name,
            "hostname": platform.node(),
            "dtype": "float64",
        }
        if self.name == "cpu":
            return result
        device = self.xp.cuda.Device()
        properties = self.xp.cuda.runtime.getDeviceProperties(device.id)
        name = properties.get("name", "unknown")
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        result.update({
            "gpu_model": str(name),
            "gpu_uuid": None,
            "total_vram_bytes": int(properties["totalGlobalMem"]),
            "driver_version": int(self.xp.cuda.runtime.driverGetVersion()),
            "cuda_runtime_version": int(self.xp.cuda.runtime.runtimeGetVersion()),
            "cupy_version": self.xp.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        })
        try:
            query = subprocess.run(
                ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                check=True, capture_output=True, text=True, timeout=5,
            )
            uuids = [line.strip() for line in query.stdout.splitlines() if line.strip()]
            if uuids:
                result["gpu_uuid"] = uuids[0]
        except (OSError, subprocess.SubprocessError):
            pass
        return result


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
