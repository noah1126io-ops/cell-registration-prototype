"""Timing smoke (development only, not reported): one real candidate through the Phase 1 record."""
import sys, time, json
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "analysis/adaptive-registration-validation-20261111"))
import common as C11
from src.registration_research import candidate_adapter as CA, cell_correspondence as CC
case = sys.argv[1] if len(sys.argv) > 1 else "P6_2_orig"
cs = next(c for c in C11.load_cases() if c["case_id"] == case)
he, mv = C11.DerivedSide(cs["he_sample"], "HE"), C11.DerivedSide(cs["if_sample"], "IF")
for o in cs["he_ops"]: he.apply(o)
for o in cs["if_ops"]: mv.apply(o)
cands = CA.from_benchmark_result(REPO / f"analysis/adaptive-registration-validation-20261111/runs/{case}__D/result.json")
print([c.id for c in cands], len(he.points), len(mv.points), flush=True)
for c in cands[:2]:
    t = time.perf_counter()
    rec, _ = CC.evaluate_candidate(c, he.points, he.extent, mv.points, mv.extent)
    print(round(time.perf_counter() - t, 1), json.dumps({k: rec.get(k) for k in ("candidate", "n_fixed_overlap", "n_moving_overlap", "B2_matched_n", "B1_matched_n", "B0_matched_n", "decoy_B2_matched_frac", "B2_matched_frac", "graph_support_median", "decoy_graph_support_median", "c3_le_3um", "runtime_s")}), flush=True)
