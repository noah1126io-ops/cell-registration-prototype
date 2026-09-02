from __future__ import annotations

import json
import os
import platform

import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter, map_coordinates


def main() -> None:
    cp.show_config()
    device_count = cp.cuda.runtime.getDeviceCount()
    if device_count < 1:
        raise RuntimeError("No CUDA device is visible to CuPy.")

    device = cp.cuda.Device(0)
    properties = cp.cuda.runtime.getDeviceProperties(0)
    with device:
        source = cp.arange(64, dtype=cp.float64).reshape(8, 8)
        elementwise = source * 1.5 + 2.0
        blurred = gaussian_filter(elementwise, sigma=1.0)
        rows, cols = cp.indices(source.shape, dtype=cp.float64)
        coordinates = cp.stack([rows + 0.1, cols + 0.2])
        sampled = map_coordinates(blurred, coordinates, order=1, mode="nearest")
        gradient_y, gradient_x = cp.gradient(sampled)
        host = cp.asnumpy(cp.stack([sampled, gradient_x, gradient_y]))
        device.synchronize()

    report = {
        "hostname": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cupy_version": cp.__version__,
        "device_count": int(device_count),
        "device_name": properties["name"].decode(),
        "pci_bus_id": device.pci_bus_id,
        "compute_capability": device.compute_capability,
        "dtype": str(source.dtype),
        "elementwise_ok": bool(cp.isfinite(elementwise).all().item()),
        "gaussian_filter_ok": bool(cp.isfinite(blurred).all().item()),
        "map_coordinates_ok": bool(cp.isfinite(sampled).all().item()),
        "gradient_ok": bool(cp.isfinite(gradient_x).all().item() and cp.isfinite(gradient_y).all().item()),
        "host_device_transfer_ok": bool(host.shape == (3, 8, 8)),
        "result_checksum": float(host.sum()),
    }
    print("CUPY_SMOKE_RESULT=" + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
