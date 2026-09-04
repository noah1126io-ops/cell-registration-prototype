import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from src.array_backend import cuda_available
from src.density_flow import joint_density_tissue_structure_registration


RTOL = 1e-10
ATOL = 1e-10


def _metadata(size: int) -> dict:
    return {
        "bounds_um": [0.0, 0.0, float(size), float(size)],
        "output_pixel_size_um": 1.0,
        "output_origin": "upper-left",
        "width": size,
        "height": size,
        "row0_world_y": 0.5,
        "col0_world_x": 0.5,
    }


def _run(device: str):
    size = 72
    yy, xx = np.indices((size, size), dtype=float)
    fixed = np.array(
        [(x, y) for y in range(12, 61, 8) for x in range(12, 61, 8)],
        dtype=float,
    )
    shift = 3.0 * np.exp(
        -((fixed[:, 0] - 36.0) ** 2 + (fixed[:, 1] - 36.0) ** 2) / (2.0 * 18.0**2)
    )
    moving = fixed.copy()
    moving[:, 0] -= shift
    fixed_mask = ((xx - 36.0) ** 2 / 29.0**2 + (yy - 36.0) ** 2 / 26.0**2) < 1.0
    source_x = xx + 3.0 * np.exp(
        -((xx - 36.0) ** 2 + (yy - 36.0) ** 2) / (2.0 * 18.0**2)
    )
    moving_mask = map_coordinates(
        fixed_mask.astype(float), [yy, source_x], order=1, mode="constant"
    ) >= 0.5
    image = np.full((size, size, 3), 235, dtype=np.uint8)
    image[moving_mask] = [160, 95, 145]
    for x, y in moving.astype(int):
        image[max(y - 1, 0): y + 2, max(x - 1, 0): x + 2] = [70, 35, 90]

    return joint_density_tissue_structure_registration(
        fixed,
        moving,
        affine_he_image=image,
        affine_he_tissue_mask=moving_mask,
        affine_he_metadata=_metadata(size),
        bounds=(0.0, 0.0, float(size), float(size)),
        density_pixel_size=2.0,
        stage_a_scales_um=(12.0, 6.0),
        stage_b_scales_um=(6.0, 3.0),
        stage_a_iterations=5,
        stage_b_iterations=5,
        stage_a_learning_rate=0.12,
        stage_b_learning_rate=0.07,
        support_weight=1.0,
        structure_weight=0.35,
        detect_axis_reversal=False,
        global_translation_initialization="off",
        max_displacement=12.0,
        displacement_p95_limit=10.0,
        minimum_absolute_median_improvement=0.01,
        minimum_relative_median_improvement=0.001,
        minimum_jacobian_p05=0.5,
        maximum_jacobian_p95=1.5,
        retain_research_diagnostic_fields=True,
        device=device,
        dtype="float64",
    )


def _assert_numeric_mapping_close(actual: dict, expected: dict, keys) -> None:
    for key in keys:
        np.testing.assert_allclose(actual[key], expected[key], rtol=RTOL, atol=ATOL)


@pytest.mark.skipif(not cuda_available(), reason="CUDA device is not available")
def test_joint_flow_cpu_cuda_float64_stage_and_final_parity():
    cpu = _run("cpu")
    gpu = _run("cuda")

    assert gpu.metrics["compute_backend"] == "cuda"
    assert gpu.metrics["stage_backends"] == {"stage_a": "cuda", "stage_b": "cuda"}
    assert cpu.metrics["stage_backends"] == {"stage_a": "cpu", "stage_b": "cpu"}
    assert (
        gpu.success, gpu.applied, gpu.rejection_reason,
        gpu.metrics["joint_flow"]["stage_a_selected_checkpoint"],
        gpu.metrics["joint_flow"]["selected_checkpoint"],
    ) == (
        cpu.success, cpu.applied, cpu.rejection_reason,
        cpu.metrics["joint_flow"]["stage_a_selected_checkpoint"],
        cpu.metrics["joint_flow"]["selected_checkpoint"],
    )

    cpu_joint = cpu.metrics["joint_flow"]
    gpu_joint = gpu.metrics["joint_flow"]
    for name in ("stage_a_displacement_x", "stage_a_displacement_y"):
        np.testing.assert_allclose(gpu_joint[name], cpu_joint[name], rtol=RTOL, atol=ATOL)

    for name, cpu_values in cpu_joint["intermediate_diagnostics"].items():
        np.testing.assert_allclose(
            gpu_joint["intermediate_diagnostics"][name], cpu_values, rtol=RTOL, atol=ATOL,
            err_msg=f"intermediate mismatch: {name}",
        )

    for name in ("stage_b_incremental_x", "stage_b_incremental_y"):
        np.testing.assert_allclose(gpu_joint[name], cpu_joint[name], rtol=RTOL, atol=ATOL)

    for name in (
        "attempted_transformed_points", "transformed_points",
        "attempted_displacement_x", "attempted_displacement_y",
        "displacement_x", "displacement_y",
    ):
        np.testing.assert_allclose(getattr(gpu, name), getattr(cpu, name), rtol=RTOL, atol=ATOL)

    stage_keys = (
        "displacement_p50", "displacement_p95", "displacement_max",
        "jacobian_min", "jacobian_p05", "jacobian_median", "jacobian_p95", "jacobian_max",
        "density_objective_selected", "support_objective_selected",
        "structure_objective_selected", "total_objective_selected",
    )
    _assert_numeric_mapping_close(gpu_joint["stage_a"], cpu_joint["stage_a"], stage_keys)
    _assert_numeric_mapping_close(gpu_joint["stage_b"], cpu_joint["stage_b"], stage_keys)

    assert len(gpu.metrics["optimization_history"]) == len(cpu.metrics["optimization_history"])
    for gpu_row, cpu_row in zip(
        gpu.metrics["optimization_history"], cpu.metrics["optimization_history"], strict=True
    ):
        assert (gpu_row["stage"], gpu_row["accepted"]) == (cpu_row["stage"], cpu_row["accepted"])
        _assert_numeric_mapping_close(
            gpu_row, cpu_row,
            ("total", "density", "support", "structure", "smoothness", "magnitude", "jacobian_barrier"),
        )

    final_metric_keys = (
        "symmetric_median_distance", "symmetric_within_3", "symmetric_within_5",
        "symmetric_within_10", "mutual_nearest_fraction",
    )
    _assert_numeric_mapping_close(gpu.metrics["attempted"], cpu.metrics["attempted"], final_metric_keys)
    safety_keys = (
        "attempted_jacobian_min", "attempted_jacobian_median", "attempted_jacobian_max",
        "fraction_jacobian_foldover_le_0", "raw_displacement_median",
        "raw_displacement_p95", "raw_displacement_max",
    )
    _assert_numeric_mapping_close(gpu.metrics["safety"], cpu.metrics["safety"], safety_keys)
    assert len(gpu.metrics["local_region_metrics"]) == len(cpu.metrics["local_region_metrics"])
    for gpu_row, cpu_row in zip(
        gpu.metrics["local_region_metrics"], cpu.metrics["local_region_metrics"], strict=True
    ):
        assert gpu_row.keys() == cpu_row.keys()
        for key in gpu_row:
            if isinstance(gpu_row[key], (int, float)):
                np.testing.assert_allclose(gpu_row[key], cpu_row[key], rtol=RTOL, atol=ATOL)
            else:
                assert gpu_row[key] == cpu_row[key]

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
    for stage in ("stage_a", "stage_b"):
        assert gpu.metrics["stage_backend_provenance"][stage]["compute_backend"] == "cuda"
        assert gpu.metrics["stage_backend_provenance"][stage]["gpu_uuid"] == provenance["gpu_uuid"]


@pytest.mark.skipif(not cuda_available(), reason="CUDA device is not available")
def test_joint_flow_auto_selects_cuda_and_propagates_it_to_both_stages():
    result = _run("auto")

    assert result.metrics["compute_backend"] == "cuda"
    assert result.metrics["stage_backends"] == {"stage_a": "cuda", "stage_b": "cuda"}
