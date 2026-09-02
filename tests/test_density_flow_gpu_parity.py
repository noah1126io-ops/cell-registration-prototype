import numpy as np
import pytest

from src.array_backend import cuda_available
from src.density_flow import tissue_aware_density_flow_registration


def _run(device):
    xs, ys = np.meshgrid(np.arange(10.0, 71.0, 10.0), np.arange(10.0, 71.0, 10.0))
    fixed = np.column_stack([xs.ravel(), ys.ravel()])
    moving = fixed + np.column_stack([
        1.5 * np.sin(fixed[:, 1] / 22.0),
        -1.2 * np.cos(fixed[:, 0] / 24.0),
    ])
    return tissue_aware_density_flow_registration(
        fixed, moving,
        density_pixel_size=2.0,
        density_blur_scales=(4.0, 2.0),
        optimization_levels=2,
        iterations_per_level=4,
        max_displacement=12.0,
        displacement_p95_limit=12.0,
        detect_axis_reversal=False,
        global_translation_initialization="off",
        device=device,
        dtype="float64",
    )


@pytest.mark.skipif(not cuda_available(), reason="CUDA device is not available")
def test_density_flow_cpu_cuda_float64_synthetic_parity():
    cpu = _run("cpu")
    gpu = _run("cuda")

    # These strict initial tolerances are near float64 roundoff scaled through
    # interpolation; they are tightened/relaxed only from measured differences.
    rtol = 1e-10
    atol = 1e-10
    np.testing.assert_allclose(gpu.transformed_points, cpu.transformed_points, rtol=rtol, atol=atol)
    np.testing.assert_allclose(gpu.attempted_displacement_x, cpu.attempted_displacement_x, rtol=rtol, atol=atol)
    np.testing.assert_allclose(gpu.attempted_displacement_y, cpu.attempted_displacement_y, rtol=rtol, atol=atol)

    cpu_history = cpu.metrics["optimization_history"]
    gpu_history = gpu.metrics["optimization_history"]
    assert len(gpu_history) == len(cpu_history)
    for cpu_row, gpu_row in zip(cpu_history, gpu_history, strict=True):
        assert gpu_row["accepted"] == cpu_row["accepted"]
        for key in ("total", "density", "smoothness", "magnitude", "jacobian_barrier"):
            np.testing.assert_allclose(gpu_row[key], cpu_row[key], rtol=rtol, atol=atol)

    assert gpu.metrics["density_flow"]["selected_checkpoint_type"] == cpu.metrics["density_flow"]["selected_checkpoint_type"]
    for section, keys in {
        "attempted": ("symmetric_median_distance", "mutual_nearest_fraction"),
        "safety": (
            "attempted_jacobian_min", "attempted_jacobian_median",
            "attempted_jacobian_max", "fraction_jacobian_foldover_le_0",
            "raw_displacement_median", "raw_displacement_p95", "raw_displacement_max",
        ),
    }.items():
        for key in keys:
            np.testing.assert_allclose(gpu.metrics[section][key], cpu.metrics[section][key], rtol=rtol, atol=atol)
    assert (gpu.success, gpu.applied, gpu.rejection_reason) == (cpu.success, cpu.applied, cpu.rejection_reason)
    assert gpu.metrics["compute_backend"] == "cuda"
    provenance = gpu.metrics["backend_provenance"]
    assert provenance["dtype"] == "float64"
    assert provenance["hostname"]
    assert provenance["gpu_model"]
    assert provenance["gpu_uuid"].startswith("GPU-")
    assert provenance["total_vram_bytes"] > 0
    assert provenance["driver_version"] > 0
    assert provenance["cuda_runtime_version"] > 0
    assert provenance["cupy_version"]
    assert provenance["cuda_visible_devices"] is not None
