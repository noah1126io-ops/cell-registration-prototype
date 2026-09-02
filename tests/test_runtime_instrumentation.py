from src.runtime_instrumentation import RuntimeRecorder


def test_runtime_recorder_accumulates_stages_and_total_without_negative_values():
    ticks = iter([10.0, 11.0, 13.5, 15.0])
    recorder = RuntimeRecorder(clock=lambda: next(ticks))

    started = recorder.start()
    assert recorder.stop("affine_icp", started) == 2.5
    snapshot = recorder.snapshot()

    assert snapshot["affine_icp"] == 2.5
    assert snapshot["total_runtime_seconds"] == 5.0
