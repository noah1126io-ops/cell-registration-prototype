from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class RuntimeRecorder:
    """Accumulate monotonic stage timings without changing numerical code paths."""

    clock: Callable[[], float] = time.perf_counter
    stages: dict[str, float] = field(default_factory=dict)
    _started_at: float = field(init=False)

    def __post_init__(self) -> None:
        self._started_at = self.clock()

    def start(self) -> float:
        return self.clock()

    def stop(self, stage: str, started_at: float) -> float:
        elapsed = max(float(self.clock() - started_at), 0.0)
        self.stages[stage] = self.stages.get(stage, 0.0) + elapsed
        return elapsed

    def snapshot(self, *, include_total: bool = True) -> dict[str, float]:
        values = {name: float(seconds) for name, seconds in self.stages.items()}
        if include_total:
            values["total_runtime_seconds"] = max(float(self.clock() - self._started_at), 0.0)
        return values
