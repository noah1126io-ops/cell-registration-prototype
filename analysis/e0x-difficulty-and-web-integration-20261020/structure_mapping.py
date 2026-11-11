"""Place the real pairs on the synthetic axes using only quantities measurable on real data (protocol §5).

Computes the same self-similarity index as aggregate_phase1.structure_summary() on the synthetic evaluation cases
(base and the repeat levels; fixed = large side, moving = small side), and the observed signal at the top candidate
(<=3 um proximity / mirror null) for the synthetic hard cases next to the real ones.

  python -B structure_mapping.py   -> structure_mapping.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aggregate_phase1 as agg  # noqa: E402
import synth  # noqa: E402


def main():
    rows = []
    for cid, factor, p in synth.conditions():
        if factor not in ("base", "repeat"):
            continue
        for seed in synth.EVAL_SEEDS:
            C = synth.make_case(p, seed)
            rows.append({"source": "synthetic", "cond": cid, "seed": seed,
                         "self_similarity_fixed": agg.self_similarity(C.fixed, (synth.WF, synth.WF)),
                         "self_similarity_moving": agg.self_similarity(C.moving - C.moving.min(0), (C.wm, C.wm))})
    st = json.loads((HERE / "diagnostic_summary.json").read_text())["structure"]
    for s, rec in st.items():
        rows.append({"source": "real", "cond": s, "seed": None,
                     "self_similarity_fixed": rec["HE"]["self_similarity_index"],
                     "self_similarity_moving": rec["IF"]["self_similarity_index"]})
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "structure_mapping.csv", index=False)
    print(df.groupby(["source", "cond"])[["self_similarity_fixed", "self_similarity_moving"]].agg(["mean", "min", "max"]).round(3).to_string())


if __name__ == "__main__":
    main()
