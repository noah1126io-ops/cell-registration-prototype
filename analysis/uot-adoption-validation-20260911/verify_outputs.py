"""Verify completeness, source consistency, and no tracked workflow changes."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from validation_lib import OUT, PRIOR_DENSITY, PRIOR_UOT, UOT_CONFIGS, save_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    required = [
        "REPORT.md", "real_dataset_metrics.csv", "real_dataset_pairwise_deltas.csv",
        "synthetic_metrics.csv", "synthetic_summary.csv", "uot_adoption_decision.md",
        "real_dataset_comparison.png", "pass_rate_comparison.png",
        "synthetic_ground_truth_error.png", "missingness_sensitivity.png",
        "accuracy_deformation_pareto.png",
    ]
    real = pd.read_csv(OUT / "real_dataset_metrics.csv")
    synthetic = pd.read_csv(OUT / "synthetic_metrics.csv")
    reproduction = json.loads((OUT / "real_reproduction_check.json").read_text(encoding="utf-8"))
    primary_source = json.loads(
        (PRIOR_UOT / "trials/uot/coarse_04/result.json").read_text(encoding="utf-8")
    )["parameters"]
    conservative_source = json.loads(
        (PRIOR_UOT / "trials/uot/coarse_09/result.json").read_text(encoding="utf-8")
    )["parameters"]
    config_checks = {
        "primary_cost_scale": UOT_CONFIGS["UOT primary"]["cost_scale_um"] == primary_source["cost_scale_um"],
        "primary_entropy": UOT_CONFIGS["UOT primary"]["entropic_regularization"] == primary_source["entropic_regularization"],
        "primary_relaxation": UOT_CONFIGS["UOT primary"]["mass_relaxation_kl"] == primary_source["mass_relaxation_kl"],
        "conservative_cost_scale": UOT_CONFIGS["UOT conservative"]["cost_scale_um"] == conservative_source["cost_scale_um"],
        "conservative_entropy": UOT_CONFIGS["UOT conservative"]["entropic_regularization"] == conservative_source["entropic_regularization"],
        "conservative_relaxation": UOT_CONFIGS["UOT conservative"]["mass_relaxation_kl"] == conservative_source["mass_relaxation_kl"],
    }
    tracked_diff = subprocess.run(
        ["git", "diff", "--name-only"], cwd=OUT.parents[1], text=True,
        stdout=subprocess.PIPE, check=True,
    ).stdout.splitlines()
    staged_diff = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=OUT.parents[1], text=True,
        stdout=subprocess.PIPE, check=True,
    ).stdout.splitlines()
    files = {name: {"exists": (OUT / name).is_file()} for name in required}
    status = {
        "state": "PASS",
        "required_files": files,
        "real_rows": len(real),
        "real_dataset_count": int(real["dataset"].nunique()),
        "real_method_count": int(real["method"].nunique()),
        "synthetic_rows": len(synthetic),
        "synthetic_scenario_count": int(synthetic[["seed", "scenario_id"]].drop_duplicates().shape[0]),
        "synthetic_seed_count": int(synthetic["seed"].nunique()),
        "synthetic_method_count": int(synthetic["method"].nunique()),
        "synthetic_duplicate_keys": int(synthetic[["seed", "scenario_id", "method"]].duplicated().sum()),
        "synthetic_exception_count": int(synthetic["rejection_reason"].fillna("").str.startswith("exception:").sum()),
        "known_affine_reconstruction_error_max_um": float(synthetic["known_affine_reconstruction_error_max_um"].max()),
        "real_uot_reproduction_exact_within_1e_10": {
            method: bool(reproduction[method]["exact_within_1e-10"])
            for method in ("UOT primary", "UOT conservative")
        },
        "uot_source_configuration_checks": config_checks,
        "tracked_modified_files": tracked_diff,
        "staged_files": staged_diff,
        "workflow_c_ui_d_tracked_changes": [],
        "source_hashes": {
            "prior_report": sha256(PRIOR_UOT / "REPORT.md"),
            "prior_uot_trials_csv": sha256(PRIOR_UOT / "uot_trials.csv"),
            "primary_result_json": sha256(PRIOR_UOT / "trials/uot/coarse_04/result.json"),
            "conservative_result_json": sha256(PRIOR_UOT / "trials/uot/coarse_09/result.json"),
            "joint_exact_config_json": sha256(PRIOR_DENSITY / "joint_exact_config.json"),
            "density_exact_config_json": sha256(PRIOR_DENSITY / "density_effective_parameters.json"),
        },
    }
    checks = [item["exists"] for item in files.values()] + list(config_checks.values()) + [
        len(real) == 5,
        real["method"].nunique() == 5,
        len(synthetic) == 250,
        synthetic["seed"].nunique() == 5,
        synthetic["method"].nunique() == 5,
        status["synthetic_duplicate_keys"] == 0,
        status["synthetic_exception_count"] == 0,
        all(status["real_uot_reproduction_exact_within_1e_10"].values()),
        not tracked_diff,
        not staged_diff,
    ]
    status["state"] = "PASS" if all(checks) else "FAIL"
    save_json(OUT / "verification.json", status)
    save_json(OUT / "provenance.json", {
        "date": "2026-09-11",
        "branch": "feature/gpu-slurm",
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=OUT.parents[1], text=True,
            stdout=subprocess.PIPE, check=True,
        ).stdout.strip(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pot": __import__("ot").__version__,
        "synthetic_slurm_jobs": {
            "canonical_tissue_domain_evaluation": 213211,
            "superseded_global_grid_pilot": 213206,
        },
        "runtime_is_formal_benchmark": False,
        "source_hashes": status["source_hashes"],
        "notes": [
            "The v2 shards are canonical because they retain attempted-field safety and evaluate field error within known tissue support.",
            "The earlier pilot shards are retained only as execution provenance and are not used in synthetic_metrics.csv.",
            "No parameter sweep or per-scenario tuning was performed.",
        ],
    })
    print(json.dumps({"state": status["state"], "checks": len(checks)}, indent=2))


if __name__ == "__main__":
    main()
