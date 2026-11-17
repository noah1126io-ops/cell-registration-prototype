"""Where does the image inverse map fail to converge (031-8_6, D field)?"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]; sys.path.insert(0, str(REPO))
from src.registration_explorer import catalog
from src.registration_research import cell_review as CR
sm = json.loads((HERE / "smoke_031-8_6.json").read_text())
S = catalog.load_manifest()["031-8_6"]
he_pts, _ = catalog.load_nuclei(S.modality("HE")); if_pts, _ = catalog.load_nuclei(S.modality("IF"))
auto = CR.load_auto(REPO / sm["auto_dir"])
fr = CR.compute_fields(auto, None, he_pts, if_pts, methods=("D",))["D"]
x0, y0, x1, y1 = fr.bounds
tgt, (H, W) = CR.target_grid([x0 - 10, y0 - 10, x1 + 10, y1 + 10], 2.0)
q, r, ok = CR.invert_field(tgt, fr)
bad = ~ok
edge = np.minimum.reduce([tgt[:, 0] - x0, x1 - tgt[:, 0], tgt[:, 1] - y0, y1 - tgt[:, 1]])
print("grid bounds", fr.bounds, "field at border max", float(np.abs(np.r_[fr.field_x[0], fr.field_x[-1], fr.field_x[:, 0], fr.field_x[:, -1]]).max()))
print("not converged", int(bad.sum()), "of", len(ok))
if bad.any():
    print("distance to grid edge of non-converged (um): min/median/max", np.percentile(edge[bad], [0, 50, 100]))
    print("residual of non-converged: median/max", np.median(r[bad]), r.max())
