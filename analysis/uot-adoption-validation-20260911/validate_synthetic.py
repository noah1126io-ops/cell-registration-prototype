"""Run the frozen-method, known-ground-truth synthetic validation by seed shard."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from validation_lib import (
    OUT,
    affine_result,
    evaluate_synthetic,
    generate_synthetic_sample,
    run_density,
    run_joint,
    run_uot,
    save_json,
)


SEEDS = (1101, 1102, 1103, 1104, 1105)


def scenario_table() -> list[dict]:
    scenarios: dict[tuple, dict] = {}

    def add(*, amplitude, missing, unmatched, overlap, noise, difficulty=None,
            missingness=False, deformation=False):
        key = (float(amplitude), float(missing), float(unmatched), float(overlap), float(noise))
        row = scenarios.setdefault(key, {
            "amplitude_um": key[0],
            "missing_fraction": key[1],
            "unmatched_fraction": key[2],
            "overlap_fraction": key[3],
            "noise_um": key[4],
            "difficulty": None,
            "is_missingness_sensitivity": False,
            "is_deformation_sensitivity": False,
        })
        row["difficulty"] = difficulty or row["difficulty"]
        row["is_missingness_sensitivity"] |= bool(missingness)
        row["is_deformation_sensitivity"] |= bool(deformation)

    add(amplitude=4, missing=0.10, unmatched=0.05, overlap=1.00, noise=0.25, difficulty="Easy")
    add(amplitude=8, missing=0.30, unmatched=0.15, overlap=0.85, noise=0.75, difficulty="Medium")
    add(amplitude=12, missing=0.50, unmatched=0.30, overlap=0.70, noise=1.25, difficulty="Hard")
    for missing in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        add(amplitude=8, missing=missing, unmatched=0.15, overlap=0.85, noise=0.75, missingness=True)
    for amplitude in (4.0, 8.0, 12.0):
        add(amplitude=amplitude, missing=0.30, unmatched=0.15, overlap=0.85, noise=0.75, deformation=True)
    rows = list(scenarios.values())
    rows.sort(key=lambda row: (
        row["difficulty"] is None,
        {"Easy": 0, "Medium": 1, "Hard": 2}.get(row["difficulty"], 9),
        row["amplitude_um"], row["missing_fraction"],
    ))
    for index, row in enumerate(rows):
        row["scenario_id"] = f"scenario_{index:02d}"
    return rows


def failed_result(method: str, sample, reason: str, runtime: float) -> dict:
    zeros = np.zeros_like(sample.grid_x)
    affine = affine_result(sample)
    affine.update({
        "method": method,
        "applied": False,
        "rejection_reason": reason,
        "runtime_seconds": runtime,
    })
    affine["audit"]["common_audit_pass"] = False
    affine["audit"]["hard_safety_pass"] = False
    affine["audit"]["rejection_reason"] = reason
    affine["attempted_field_x"] = zeros
    affine["attempted_field_y"] = zeros.copy()
    return affine


def run_seed(seed: int, *, device: str, max_cases: int | None = None) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    errors = []
    scenarios = scenario_table()
    if max_cases is not None:
        scenarios = scenarios[:max_cases]
    for scenario_index, scenario in enumerate(scenarios, start=1):
        sample = generate_synthetic_sample(
            seed=seed,
            amplitude_um=scenario["amplitude_um"],
            missing_fraction=scenario["missing_fraction"],
            unmatched_fraction=scenario["unmatched_fraction"],
            overlap_fraction=scenario["overlap_fraction"],
            noise_um=scenario["noise_um"],
        )
        methods = ["Affine only", "Density Flow baseline", "UOT conservative", "UOT primary", "Joint Flow"]
        for method in methods:
            started = time.perf_counter()
            try:
                if method == "Affine only":
                    result = affine_result(sample)
                elif method == "Density Flow baseline":
                    result = run_density(sample, device=device)
                elif method in ("UOT conservative", "UOT primary"):
                    result = run_uot(
                        sample.fixed_points,
                        sample.moving_points,
                        sample.grid_x,
                        sample.grid_y,
                        sample.spacing,
                        method,
                    )
                else:
                    result = run_joint(sample, device=device)
            except Exception as exc:  # keep all predeclared cases visible
                runtime = time.perf_counter() - started
                reason = f"exception:{type(exc).__name__}:{exc}"
                result = failed_result(method, sample, reason, runtime)
                errors.append({"scenario_id": scenario["scenario_id"], "method": method, "error": reason})
            row = evaluate_synthetic(sample, result)
            row.update({
                "scenario_id": scenario["scenario_id"],
                "difficulty": scenario["difficulty"],
                "is_missingness_sensitivity": scenario["is_missingness_sensitivity"],
                "is_deformation_sensitivity": scenario["is_deformation_sensitivity"],
                "device": device,
            })
            rows.append(row)
            print(
                f"seed={seed} case={scenario_index}/{len(scenarios)} method={method} "
                f"applied={row['applied']} EPE={row['field_error_um_median']:.4f} "
                f"TRE={row['independent_landmark_tre_um_median']:.4f}",
                flush=True,
            )
    return pd.DataFrame(rows), errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-index", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", type=Path, default=OUT / "synthetic_shards")
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    if not 0 <= args.seed_index < len(SEEDS):
        raise SystemExit(f"seed-index must be in [0, {len(SEEDS) - 1}]")
    seed = SEEDS[args.seed_index]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    table, errors = run_seed(seed, device=args.device, max_cases=args.max_cases)
    table.to_csv(args.output_dir / f"synthetic_seed_{seed}.csv", index=False)
    save_json(args.output_dir / f"synthetic_seed_{seed}_status.json", {
        "state": "completed",
        "seed": seed,
        "rows": len(table),
        "scenarios": int(table["scenario_id"].nunique()),
        "methods": sorted(table["method"].unique()),
        "device": args.device,
        "errors": errors,
        "runtime_seconds": time.perf_counter() - started,
        "parameter_tuning_performed": False,
    })


if __name__ == "__main__":
    main()
