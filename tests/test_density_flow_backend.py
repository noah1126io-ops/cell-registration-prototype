import numpy as np

from src.density_flow import tissue_aware_density_flow_registration


def _case():
    xs, ys = np.meshgrid(np.arange(10.0, 71.0, 10.0), np.arange(10.0, 71.0, 10.0))
    fixed = np.column_stack([xs.ravel(), ys.ravel()])
    moving = fixed + np.array([3.0, -2.0])
    return fixed, moving


def _run(*, device=None):
    fixed, moving = _case()
    kwargs = dict(
        density_pixel_size=2.0,
        density_blur_scales=(4.0, 2.0),
        optimization_levels=2,
        iterations_per_level=3,
        max_displacement=15.0,
        displacement_p95_limit=15.0,
        detect_axis_reversal=False,
        global_translation_initialization="auto",
    )
    if device is not None:
        kwargs["device"] = device
    return tissue_aware_density_flow_registration(fixed, moving, **kwargs)


def test_explicit_cpu_backend_preserves_default_density_flow_exactly():
    baseline = _run()
    explicit = _run(device="cpu")

    np.testing.assert_array_equal(explicit.transformed_points, baseline.transformed_points)
    np.testing.assert_array_equal(explicit.displacement_x, baseline.displacement_x)
    np.testing.assert_array_equal(explicit.displacement_y, baseline.displacement_y)
    assert explicit.success == baseline.success
    assert explicit.applied == baseline.applied
    assert explicit.rejection_reason == baseline.rejection_reason
    assert explicit.metrics["optimization_history"] == baseline.metrics["optimization_history"]
    assert explicit.metrics["compute_backend"] == "cpu"
    assert explicit.metrics["backend_provenance"]["dtype"] == "float64"
