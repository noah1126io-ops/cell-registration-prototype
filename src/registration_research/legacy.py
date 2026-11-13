"""Read-only access to the frozen research implementations that this package reuses.

Nothing here re-implements an algorithm.  The modules below are imported from their analysis directories exactly as
they were frozen (their own sys.path conventions are kept), and the frozen B2 adoption models are rebuilt from the
coefficients saved by the 2026-10-14 study (no refitting).

Reused (by import):
  analysis/contextual-graph-uot-20261008       regional NCC, fingerprints, candidate sets, graph support, UOT plan/extract
  analysis/contextual-uot-cost-ablation-20261008  build_fixed_context, run_uot (candidate-restricted UOT + diagnostics)
  analysis/graph-matching-adoption-separation-20261014  B0/B1/B3 runs, result_frame, LogisticCalibrator, B2 adoption
  analysis/uot-adoption-validation-20260911     UOT conservative field (run_uot), fit_linear_rbf, common_audit, synthetic generator
  src/pointset_registration.py                  mutual_nearest_pairs, matched_nuclei_rbf_fine_warp (production helpers)
"""
from __future__ import annotations

import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "analysis"
GRAPH_STUDY = ANALYSIS / "graph-matching-adoption-separation-20261014"
CONTEXT_STUDY = ANALYSIS / "contextual-graph-uot-20261008"
COST_STUDY = ANALYSIS / "contextual-uot-cost-ablation-20261008"
UOT_STUDY = ANALYSIS / "uot-adoption-validation-20260911"
B2_SUMMARY = GRAPH_STUDY / "diagnostic_summary.json"

# files whose bytes define the reused behaviour (recorded in baseline_manifest.json and checked by the tests)
REUSED_FILES = (
    CONTEXT_STUDY / "config.py", CONTEXT_STUDY / "candidate_generation.py", CONTEXT_STUDY / "graph_consistency.py",
    CONTEXT_STUDY / "contextual_uot.py", CONTEXT_STUDY / "local_fingerprint.py", CONTEXT_STUDY / "regional_pattern.py",
    CONTEXT_STUDY / "synthetic_correspondence.py", COST_STUDY / "run_cost_ablation.py", GRAPH_STUDY / "experiment_lib.py",
    B2_SUMMARY, UOT_STUDY / "validation_lib.py", REPO / "src/pointset_registration.py", REPO / "src/density_flow.py",
)


@lru_cache(maxsize=1)
def modules():
    """(experiment_lib, run_cost_ablation, validation_lib) imported from the frozen study directories."""
    if str(GRAPH_STUDY) not in sys.path:
        sys.path.insert(0, str(GRAPH_STUDY))
    import experiment_lib  # noqa: E402  (inserts the contextual / cost-ablation dirs itself)
    import run_cost_ablation  # noqa: E402
    import validation_lib  # noqa: E402  (found through synthetic_correspondence's own sys.path entry)
    for mod, d in ((experiment_lib, GRAPH_STUDY), (run_cost_ablation, COST_STUDY), (validation_lib, UOT_STUDY)):
        if Path(mod.__file__).resolve().parent != d.resolve():
            raise ImportError(f"{mod.__name__} resolved to {mod.__file__}, expected {d}")
    return experiment_lib, run_cost_ablation, validation_lib


@lru_cache(maxsize=1)
def frozen_b2():
    """The B2 models + thresholds exactly as saved on 2026-10-14 (synthetic Train fit, Validation threshold)."""
    E, _, _ = modules()
    d = json.loads(B2_SUMMARY.read_text())

    def calib(rec):
        return E.LogisticCalibrator(feature_names=tuple(rec["feature_names"]), center=np.asarray(rec["center"], float),
                                    scale=np.asarray(rec["scale"], float), coefficients=np.asarray(rec["coefficients"], float),
                                    intercept=float(rec["intercept"]), training_rows=int(rec["training_rows"]),
                                    positive_fraction=float(rec["positive_fraction"]), converged=bool(rec["converged"]))
    th = d["b2_thresholds"]
    return {"match_model": calib(d["b2_match_model"]), "unmatched_model": calib(d["b2_unmatched_model"]),
            "match_threshold": float(th["match"]["threshold"]), "unmatched_threshold": float(th["unmatched"]["threshold"]),
            "source": str(B2_SUMMARY.relative_to(REPO))}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reused_file_hashes() -> dict:
    return {str(Path(p).relative_to(REPO)): sha256_file(p) for p in REUSED_FILES}
