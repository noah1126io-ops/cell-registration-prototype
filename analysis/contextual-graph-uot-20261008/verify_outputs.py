"""Verify completeness and isolation of the analysis outputs."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

from config import METHODS, OUT, ROOT


REQUIRED = (
    "REPORT.md",
    "method_metrics.csv",
    "per_case_metrics.csv",
    "candidate_recall.csv",
    "confidence_calibration.csv",
    "ablation_summary.csv",
    "method_correspondence_accuracy.png",
    "candidate_recall.png",
    "confidence_calibration.png",
    "missingness_sensitivity.png",
    "graph_support_correct_vs_wrong.png",
    "synthetic_example_matches.png",
    "real_match_candidates.png",
    "manual_review_candidates.csv",
    "fingerprint_component_summary.csv",
)


def main() -> None:
    per_case = pd.read_csv(OUT / "per_case_metrics.csv")
    method = pd.read_csv(OUT / "method_metrics.csv")
    candidates = pd.read_csv(OUT / "candidate_recall.csv")
    calibration = pd.read_csv(OUT / "confidence_calibration.csv")
    real = pd.read_csv(OUT / "real_correspondence_candidates.csv")
    tracked_changes = subprocess.check_output(
        ["git", "diff", "--name-only", "--", "app.py", "src", "tests", "docs"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    checks = {
        "required_files_exist": all((OUT / name).exists() for name in REQUIRED),
        "method_count_is_6": len(method) == len(METHODS),
        "scenario_count_is_45": per_case["scenario_id"].nunique() == 45,
        "primary_case_count_is_15": per_case.loc[per_case["analysis_group"] == "primary", "scenario_id"].nunique() == 15,
        "per_case_rows_are_270": len(per_case) == 45 * len(METHODS),
        "per_case_keys_unique": not per_case.duplicated(["scenario_id", "method"]).any(),
        "all_algorithm_outputs_finite": bool(per_case["finite"].all()),
        "all_uot_runs_converged": bool(
            per_case.loc[per_case["method"].str.contains("UOT"), "solver_converged"].all()
        ),
        "candidate_k_complete": set(candidates["k"].unique()) == {10, 20, 30},
        "calibration_methods_complete": set(calibration["method"].unique()) == set(METHODS),
        "real_rows_match_saved_moving_points": len(real) == 1754,
        "production_tracked_changes_absent": tracked_changes == [],
    }
    payload = {
        "state": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "counts": {
            "methods": len(method),
            "scenarios": per_case["scenario_id"].nunique(),
            "per_case_rows": len(per_case),
            "candidate_rows": len(candidates),
            "calibration_rows": len(calibration),
            "real_rows": len(real),
        },
        "tracked_production_changes": tracked_changes,
    }
    (OUT / "verification.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if payload["state"] != "PASS":
        raise SystemExit(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

