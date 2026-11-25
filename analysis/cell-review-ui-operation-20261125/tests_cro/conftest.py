from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: AppTest on real samples (minutes)")
