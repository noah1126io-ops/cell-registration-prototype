import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, map_coordinates

from src.array_backend import BackendUnavailableError, cuda_available, get_array_backend


def test_cpu_backend_is_default_and_matches_existing_scipy_grid_operations():
    backend = get_array_backend()
    source = np.arange(36, dtype=np.float64).reshape(6, 6)
    rows, cols = np.indices(source.shape, dtype=float)

    actual_blur = backend.gaussian_filter(backend.asarray(source), sigma=1.2)
    expected_blur = gaussian_filter(source, sigma=1.2)
    actual_sample = backend.map_coordinates(
        actual_blur, [rows + 0.1, cols + 0.2], order=1, mode="nearest"
    )
    expected_sample = map_coordinates(
        expected_blur, [rows + 0.1, cols + 0.2], order=1, mode="nearest"
    )

    assert backend.name == "cpu"
    assert actual_blur.dtype == np.float64
    np.testing.assert_array_equal(actual_blur, expected_blur)
    np.testing.assert_array_equal(actual_sample, expected_sample)


def test_backend_selection_rejects_unknown_device_and_non_float64():
    with pytest.raises(ValueError, match="device"):
        get_array_backend("tpu")
    with pytest.raises(ValueError, match="float64"):
        get_array_backend("cpu", dtype="float32")


def test_auto_backend_always_resolves_to_an_available_backend():
    backend = get_array_backend("auto")
    assert backend.name in {"cpu", "cuda"}
    if backend.name == "cuda":
        assert cuda_available()


@pytest.mark.skipif(not cuda_available(), reason="CUDA device is not available in the CPU test environment")
def test_cuda_backend_import_and_float64_smoke():
    backend = get_array_backend("cuda")
    values = backend.asarray([[1.0, 2.0], [3.0, 4.0]])
    backend.synchronize()

    assert backend.name == "cuda"
    assert values.dtype == backend.xp.float64
    np.testing.assert_array_equal(backend.to_cpu(values), np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_explicit_cuda_request_fails_cleanly_without_a_visible_device():
    if cuda_available():
        pytest.skip("CUDA device is visible")
    with pytest.raises(BackendUnavailableError, match="CUDA"):
        get_array_backend("cuda")
