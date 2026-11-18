from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location("crx_conftest", REPO / "analysis/cell-review-explorer-20261117/tests_crx/conftest.py")
crx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(crx)
make_synthetic = crx.make_synthetic          # the 11-17 synthetic generator (reused unchanged)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: AppTest on real samples (minutes)")


@pytest.fixture(scope="session")
def syn():
    return make_synthetic()


@pytest.fixture(scope="session")
def syn_auto(syn):
    from src.registration_research import cell_review as CR
    return CR.compute_auto(syn["he_pts"], syn["he_ids"], syn["if_pts"], syn["if_ids"], syn["M"], syn["t"], roi=syn["roi"],
                           meta={"sample_id": "SYN"})
