"""Score a filled manual_eval/template.csv: per method x section, share of each judgement among rows with a judgement and
boundary_uncertain != 1; uncertain rows are counted separately.  Prints nothing as accuracy while the file is unfilled."""
import sys
from pathlib import Path

import pandas as pd

p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "template.csv"
d = pd.read_csv(p).fillna({"judgement": "", "boundary_uncertain": 0})
filled = d[d.judgement != ""]
if filled.empty:
    print("no judgements yet: no accuracy can be reported (no ground truth)")
    sys.exit(0)
unc = filled[filled.boundary_uncertain.astype(int) == 1]
ok = filled[filled.boundary_uncertain.astype(int) != 1]
out = ok.groupby(["section", "method"]).judgement.value_counts(normalize=True).unstack(fill_value=0)
out["n"] = ok.groupby(["section", "method"]).size()
out["n_boundary_uncertain"] = unc.groupby(["section", "method"]).size().reindex(out.index).fillna(0).astype(int)
print(out.round(3).to_string())
