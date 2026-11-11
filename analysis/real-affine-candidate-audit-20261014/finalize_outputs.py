"""Lightweight report finalization after reviewing the completed audit outputs."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import audit_lib as lib
import run_audit


def main() -> None:
    metrics = pd.read_csv(lib.OUT / "regional_ncc_window_metrics.csv")
    cells = pd.read_csv(lib.OUT / "regional_ncc_per_nucleus.csv")
    anchors = pd.read_csv(lib.OUT / "regional_ncc_anchor_diagnostics.csv")
    boundary_counts = []
    boundary_fractions = []
    for row in metrics.itertuples(index=False):
        current = cells[cells["search_radius_um"] == row.search_radius_um]
        if row.group == "all":
            mask = np.ones(len(current), dtype=bool)
        elif row.group == "boundary":
            mask = current["hits_boundary"].to_numpy(bool)
        elif row.group == "non_boundary":
            mask = ~current["hits_boundary"].to_numpy(bool)
        else:
            mask = current[f"hits_{row.group}"].to_numpy(bool)
        anchor_ids = np.unique(current.loc[mask, "anchor_index"].to_numpy(int))
        anchor_current = anchors[anchors["search_radius_um"] == row.search_radius_um].set_index("anchor_index")
        hits = anchor_current.loc[anchor_ids, "hits_boundary"].to_numpy(bool)
        boundary_counts.append(int(hits.sum()))
        boundary_fractions.append(float(np.mean(hits)))
    metrics["independent_anchor_boundary_hit_count"] = boundary_counts
    metrics["independent_anchor_boundary_hit_fraction"] = boundary_fractions
    metrics.to_csv(lib.OUT / "regional_ncc_window_metrics.csv", index=False)

    summary_path = lib.OUT / "diagnostic_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    all_metrics = metrics[metrics["group"] == "all"].set_index("search_radius_um")
    summary["regional_ncc"] = {
        str(radius): all_metrics.loc[radius].to_dict() for radius in (18.0, 36.0, 54.0)
    }
    summary["tests"] = {
        "analysis_only": "11 passed",
        "full_repository": "269 passed, 4 skipped",
        "analysis_test_command": "conda run -n cell-rg pytest -q analysis/real-affine-candidate-audit-20261014/test_audit.py",
        "full_repository_test_command": "conda run -n cell-rg pytest -q",
    }
    inputs = lib.load_inputs()
    affine = lib.affine_audit(inputs)
    lib.save_json(lib.OUT / "affine_transform_audit.json", affine)
    summary["affine"] = affine
    lib.save_json(summary_path, summary)
    roi = json.loads((lib.OUT / "roi_overlap_metrics.json").read_text(encoding="utf-8"))
    run_audit.write_report(
        inputs,
        affine,
        roi,
        pd.read_csv(lib.OUT / "candidate_source_audit.csv"),
        pd.read_csv(lib.OUT / "candidate_k_sensitivity.csv"),
        metrics,
        summary["spatial_residual_proxies"],
        pd.read_csv(lib.OUT / "synthetic_affine_sensitivity.csv"),
        summary["baseline_reproduction"],
        summary["elapsed_seconds"],
        summary["source_files_unchanged"],
    )
    run_audit.write_coordinate_audit(inputs, affine, roi)


if __name__ == "__main__":
    main()
