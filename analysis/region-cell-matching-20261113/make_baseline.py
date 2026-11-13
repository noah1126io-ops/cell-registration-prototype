"""Freeze the baseline before any experiment: git state, hashes of production / engine / reused research files, the
frozen configurations that this study depends on.  -> baseline_manifest.json"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from src.registration_core import workflow as W  # noqa: E402
from src.registration_core.pipeline import ACCEPT, TRUST  # noqa: E402
from src.registration_research import legacy  # noqa: E402


def git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()


def tree_hashes(*globs):
    out = {}
    for g in globs:
        for p in sorted(REPO.glob(g)):
            if p.is_file() and "__pycache__" not in p.parts:
                out[str(p.relative_to(REPO))] = legacy.sha256_file(p)
    return out


def main():
    E, _, V = legacy.modules()
    b2 = json.loads(legacy.B2_SUMMARY.read_text())
    man = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "written before any Phase 1 / Phase 2 result of this study was inspected",
        "git": {"head": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "status_porcelain": git("status", "--porcelain").splitlines()},
        "core_version": W.CORE_VERSION,
        "engine_rules": {"ACCEPT": ACCEPT, "TRUST": TRUST, "QUADRANT_RULE": W.QUADRANT_RULE,
                         "WorkflowConfig_defaults": W.asdict(W.WorkflowConfig())},
        "production_and_engine_sha256": tree_hashes("app.py", "src/registration_core/*.py", "src/registration_explorer/*.py",
                                                    "src/registration_explorer/interactive/*", "src/workflow_d.py",
                                                    "src/workflow_c_*.py", "src/pointset_registration.py", "src/density_flow.py",
                                                    "src/joint_flow.py", "src/research_store.py"),
        "reused_research_sha256": legacy.reused_file_hashes(),
        "frozen_b2": {"source": str(legacy.B2_SUMMARY.relative_to(REPO)), "thresholds": b2["b2_thresholds"],
                      "decision": b2["decision"]},
        "uot_conservative": V.UOT_CONFIGS["UOT conservative"], "safety": V.SAFETY,
        "graph_uot_config": {"UOT": sys.modules["config"].UOT, "CANDIDATES": sys.modules["config"].CANDIDATES,
                             "GRAPH": sys.modules["config"].GRAPH},
        "pot_vendor": str(sys.modules["ot"].__file__), "pot_version": sys.modules["ot"].__version__,
        "inputs": {"real_derived_runs": "analysis/adaptive-registration-validation-20261111/runs/*__D/result.json",
                   "serial_section_pair": "sweeps/real-grid27-20260908-v2/input/success_metric_*_points.npy"},
    }
    (HERE / "baseline_manifest.json").write_text(json.dumps(legacy.modules()[0].json_safe(man), indent=1, ensure_ascii=False))
    print("written; status lines:", len(man["git"]["status_porcelain"]))


if __name__ == "__main__":
    main()
