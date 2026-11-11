"""Stage extraction around the frozen A2 (analysis only; no frozen file is modified).

`init_lib.method_a2` ranks fine NCC hypotheses and hands them to `init_lib._finish`, which runs the production ICP
(`icp_similarity` trim 0.8 -> `affine_icp` trim 0.7) on the top N_REFINE hypotheses and returns the lowest-ICP-objective
basin. To see what happens *before* ICP we wrap `_finish` for the duration of one call (runtime instrumentation of the
module attribute; the file is untouched) and record the hypothesis list it receives. The frozen `_finish` is then called
unchanged, so R2 is bit-identical to a plain `method_a2` run (checked in tests and at run time).

Stages (affine maps moving HE pixel -> fixed world um, `world = M p + t`):
  R1a  NCC-only:   the top fine-NCC hypothesis (fine[0]); no ICP information used.
  R1b  selected:   the pre-ICP similarity of the hypothesis whose ICP refine won (uses ICP objective for *selection*).
  R2s  (suppl.)    R1b after the similarity-ICP step only (before affine_icp).
  R2   A2 final:   R1b after the full production ICP (= frozen A2 output).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu")
sys.path.insert(0, str(ROOT / "analysis/affine-global-initialization-20261014"))
import init_lib as L  # noqa: E402  (frozen)


def hyp_affine(h: dict) -> tuple[np.ndarray, np.ndarray]:
    """Similarity affine of an A2 hypothesis exactly as `_transform_hyp` + translation: M = s R(theta) C."""
    M = float(h["scale"]) * L.rot(float(h["theta"])) @ L.CHIRALITY[h["chirality"]]
    return M, np.asarray(h["translation"], float)


def _hyp_dict(h) -> dict:
    return {"chirality": h.chirality, "theta": float(h.theta), "scale": float(h.scale),
            "translation": np.asarray(h.translation, float).tolist(), "score": float(h.score), "source": h.source}


def similarity_only(h: dict, moving_px, fixed):
    """First production ICP step only (icp_similarity, trim 0.8, 80 it) from hypothesis h (diagnostic stage R2s)."""
    C = L.CHIRALITY[h["chirality"]]
    init = L.SimilarityTransform(scale=float(h["scale"]), rotation_rad=float(np.deg2rad(h["theta"])),
                                 translation=np.asarray(h["translation"], float))
    sim, _, _ = L.icp_similarity(moving_px @ C.T, fixed, initial=init, max_iter=80, trim_quantile=0.8)
    M0, t0 = sim.to_affine()
    return M0 @ C, np.asarray(t0, float)


def run_a2_stages(moving_px, fixed, chiralities=("proper", "reflect")) -> dict:
    captured = {}
    frozen_finish = L._finish

    def recording_finish(method, hyps, moving, fx, t_start, n_scored, n_refine=L.N_REFINE, diag=None):
        captured["hyps"] = [_hyp_dict(h) for h in hyps[:max(n_refine, 12)]]
        captured["n_refine"] = n_refine
        return frozen_finish(method, hyps, moving, fx, t_start, n_scored, n_refine=n_refine, diag=diag)

    L._finish = recording_finish
    try:
        res = L.run_method("A2", moving_px, fixed, None, None, restrict_chirality=None if len(chiralities) == 2 else chiralities[0])
    finally:
        L._finish = frozen_finish
    hyps = captured["hyps"]
    r1a = hyps[0]
    r1b = res.refined[0][3]                       # hypothesis that produced the winning ICP basin
    # rank of the selected hypothesis in the NCC ordering
    rank = next((i for i, h in enumerate(hyps) if h["chirality"] == r1b["chirality"] and np.isclose(h["theta"], r1b["theta"])
                 and np.isclose(h["scale"], r1b["scale"]) and np.allclose(h["translation"], r1b["translation"])), -1)
    Ma, ta = hyp_affine(r1a)
    Mb, tb = hyp_affine(r1b)
    Ms, ts = similarity_only(r1b, moving_px, fixed)
    return {
        "stages": {"R1a": {"M": Ma.tolist(), "t": ta.tolist(), "hyp": r1a},
                   "R1b": {"M": Mb.tolist(), "t": tb.tolist(), "hyp": r1b},
                   "R2s": {"M": Ms.tolist(), "t": ts.tolist()},
                   "R2": {"M": res.M.tolist(), "t": res.t.tolist(), "objective": res.objective}},
        "r1a_equals_r1b": bool(rank == 0),
        "selected_ncc_rank": rank,
        "hypotheses_top": hyps,
        "basins": res.refined,
        "diagnostics": res.diagnostics,
        "runtime_s": res.runtime_s,
    }
