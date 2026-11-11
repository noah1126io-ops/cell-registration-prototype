from __future__ import annotations

import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import diagnose  # noqa: E402


@pytest.fixture(scope="session")
def frozen_real():
    fixed, moving = diagnose.load_real_points()
    state = diagnose.run_frozen_m5(fixed, moving)
    trace = diagnose.trace_pipeline(state)
    return fixed, moving, state, trace
