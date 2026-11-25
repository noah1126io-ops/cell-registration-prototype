"""Precompute one automatic estimate (frozen Graph/UOT/B2, unchanged code) for the browser checks: ROI = centre of the
default ROI +- HALF µm (the same ROI as the 11-18 UI tests).  usage: prepare.py ROOT SAMPLE"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from src.registration_explorer import catalog, results  # noqa: E402
from src.registration_research import cell_review as CR  # noqa: E402

root, sample = Path(sys.argv[1]), sys.argv[2]
HALF = 110.0
S = catalog.load_manifest()[sample]
recs = [r for r in results.load_records(S) if r.M is not None]
rec = next((r for r in recs if r.method == "WF"), recs[0])
M, t, _ = results.oriented(rec.M, rec.t, rec.fixed, "IF")
he, he_ids = catalog.load_nuclei(S.modality("HE"))
iff, if_ids = catalog.load_nuclei(S.modality("IF"))
roi = CR.default_roi(CR.map_points(M, t, he), iff, S.modality("IF").extent())
c = [(roi[0] + roi[2]) / 2, (roi[1] + roi[3]) / 2]
roi = [c[0] - HALF, c[1] - HALF, c[0] + HALF, c[1] + HALF]
auto = CR.compute_auto(he, he_ids, iff, if_ids, M, t, roi=roi, if_extent=S.modality("IF").extent(),
                       meta={"sample_id": sample, "global_source": f"結果の最終変換: {rec.config}", "result_id": rec.rid})
print(CR.save_auto(auto, sample, root))
