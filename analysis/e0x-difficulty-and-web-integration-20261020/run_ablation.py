"""Real-pair E0x ablation and top-candidate diagnostics; writes cache/abl_<sample>_<fixed>fix_<variant>.json.

  python -B run_ablation.py --sample 031-8_6 --fixed HE --variant X3
  python -B run_ablation.py --task 5          # index into TASKS (Slurm array)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e0xlab  # noqa: E402,F401  (sets sys.path for oir / init_lib)
from e0xlab import run_variant  # noqa: E402
from oir.frames import Affine2D, json_safe  # noqa: E402
from oir.pipeline import RunResult, Side, evaluate  # noqa: E402
from oir.sample import load_sample  # noqa: E402
import run_real as prev  # noqa: E402  (2026-10-19 runner: feature cache + provider reference, read-only)

CACHE = HERE / "cache"
TASKS = [(s, f, v) for s in ("031-6_2", "031-8_6") for f in ("HE", "IF") for v in ("X0", "X1", "X2", "X3")]


def run(sample, fixed_key, variant):
    S = load_sample(sample)
    he = Side("HE", S.he.nuclei_um, S.he.frame.extent_um(), None)
    iff = Side("IF", S.iff.nuclei_um, S.iff.frame.extent_um(), None)
    fixed, moving = (he, iff) if fixed_key == "HE" else (iff, he)
    rec = run_variant(sample, variant, fixed, moving)
    # IF-frame evaluation with the 2026-10-19 evaluator (same ROI / population as the earlier E0x numbers)
    he_f = prev.load_feats(prev.CACHE / f"feats_{sample}_HE_2um.npz")
    if_f = prev.load_feats(prev.CACHE / f"feats_{sample}_IF_2um.npz")
    ref = prev.reference_he_to_if(sample)
    for c in rec["candidates"]:
        if "T_final" not in c:
            continue
        R = RunResult(sample, variant, "P0", fixed.key, moving.key, Affine2D.from_dict(c["T_final"]), Affine2D.from_dict(c["T_search"]))
        ev = evaluate(R, he, iff, he_f, if_f, ref_he_to_if=ref)
        c["eval_if_frame"] = {"final": ev["final"], "null": ev["null"], "inlier_over_null": ev["inlier_over_null"],
                              "geometric_pass": ev["geometric_pass"], "params_he_to_if": ev["params_he_to_if"],
                              "vs_reference_um": ev.get("vs_reference_um"), "image": ev.get("image"), "n_if_eval": ev["n_if_eval"]}
    rec["sample_extents_um"] = {"HE": list(he.extent), "IF": list(iff.extent)}
    out = CACHE / f"abl_{sample}_{fixed_key}fix_{variant}.json"
    out.write_text(json.dumps(json_safe(rec), indent=2))
    top = rec["candidates"][0] if rec["candidates"] else {}
    print(sample, fixed_key, variant, rec["verdict"], "margin=%.3f" % rec["margin"], rec["safety"],
          "le3(IF)=%s" % top.get("eval_if_frame", {}).get("final", {}).get("le_3um"), "rt=%.0fs" % rec["runtime_s"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample"); ap.add_argument("--fixed"); ap.add_argument("--variant"); ap.add_argument("--task", type=int)
    a = ap.parse_args()
    if a.task is not None:
        run(*TASKS[a.task])
    else:
        run(a.sample, a.fixed, a.variant)
