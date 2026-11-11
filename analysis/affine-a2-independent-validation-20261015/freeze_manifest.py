"""Record hashes/config of every frozen component before the validation runs (writes freeze/frozen_components.json)."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path("/mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu")
HERE = ROOT / "analysis/affine-a2-independent-validation-20261015"
sys.path.insert(0, str(ROOT / "analysis/affine-global-initialization-20261014"))
import init_lib as L  # noqa: E402

FROZEN = {
    "A2 / A0 / A3 / detector / evaluation (init_lib)": "analysis/affine-global-initialization-20261014/init_lib.py",
    "prior runner (reference for detector usage)": "analysis/affine-global-initialization-20261014/run_cases.py",
    "prior aggregate (Real detector with checkerboard held-out)": "analysis/affine-global-initialization-20261014/aggregate.py",
    "production ICP / A0": "src/pointset_registration.py",
    "Graph/UOT/B2 library": "analysis/graph-matching-adoption-separation-20261014/experiment_lib.py",
    "B2 training table": "analysis/graph-matching-adoption-separation-20261014/per_nucleus_results.csv",
    "UOT context/cost": "analysis/contextual-uot-cost-ablation-20261008/run_cost_ablation.py",
    "candidate generation": "analysis/contextual-graph-uot-20261008/candidate_generation.py",
    "graph consistency": "analysis/contextual-graph-uot-20261008/graph_consistency.py",
    "contextual UOT": "analysis/contextual-graph-uot-20261008/contextual_uot.py",
    "regional pattern (NCC +-18)": "analysis/contextual-graph-uot-20261008/regional_pattern.py",
    "local fingerprint": "analysis/contextual-graph-uot-20261008/local_fingerprint.py",
    "candidate config": "analysis/contextual-graph-uot-20261008/config.py",
    "prior Real A2 result": "analysis/affine-global-initialization-20261014/cache/real.json",
}


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


out = {
    "components": {k: {"path": v, "sha256": sha(v)} for k, v in FROZEN.items()},
    "a2_settings": {"SEARCH_RADIUS_UM": L.SEARCH_RADIUS_UM, "COARSE_ROT_STEP": L.COARSE_ROT_STEP, "FINE_ROT_HALF": L.FINE_ROT_HALF,
                    "FINE_ROT_STEP": L.FINE_ROT_STEP, "FINE_SCALE": L.FINE_SCALE, "N_REFINE": L.N_REFINE,
                    "raster_px_um": 4.0, "raster_sigma_um": 8.0, "min_overlap_fraction": 0.3,
                    "icp": "icp_similarity(trim 0.8, 80 it) -> affine_icp(trim 0.7, 80 it)"},
    "a3_settings": {"A3_ROT_STEP": L.A3_ROT_STEP, "A3_TRANS_STEP": L.A3_TRANS_STEP, "A3_SHORT_ITERS": L.A3_SHORT_ITERS, "A3_TOPK": L.A3_TOPK},
    "detector_thresholds": L.DETECT,
    "detector_usage": {"synthetic": "init_lib.detector(M, t, moving_px, fixed, basin_margin=<method diagnostics>) -- no held-out (as the 2026-10-14 synthetic 180)",
                       "real": "same + heldout_inlier = mean checkerboard held-out inlier (as 2026-10-14 aggregate.py)"},
    "python": sys.version, "platform": platform.platform(),
    "git_head": (HERE / "freeze/git_head.txt").read_text().strip(),
}
(HERE / "freeze/frozen_components.json").write_text(json.dumps(out, indent=2, default=list) + "\n")
print(json.dumps(out["components"], indent=1)[:2000])
