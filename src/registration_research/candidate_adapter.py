"""Top-k global candidates from the existing C1/C2 (registration_core), without changing the engine.

Three sources, one record format (`CandidatePose`):
  * an artifact written by `registration_core.run_workflow` (artifact["candidates"])
  * a 2026-11-11 benchmark result.json (gates[<gate>]["candidates"]; same per-candidate records)
  * a fresh C1 + C2 run through the unchanged engine functions (`localize` + `refine_and_validate`), for synthetic data

Transform convention (registration_core): fixed_world = M @ moving_world + t.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from src.registration_core import workflow as W
from src.registration_core.frames import json_safe


@dataclass
class CandidatePose:
    id: str                     # "<source>#<rank>" as in the engine
    source: str                 # image | points | a2 | init
    rank: int                   # C1 rank within its source (0 = top-1)
    verdict: str                # the source's C1 verdict (UNIQUE / AMBIGUOUS / ...)
    margin: float | None
    score: float | None         # C1 search score (z-sum); comparable only within one source
    M_search: np.ndarray
    t_search: np.ndarray
    M_final: np.ndarray         # after C2 (ICP, or the search pose when the engine kept it)
    t_final: np.ndarray
    flags: list = field(default_factory=list)
    c3: dict = field(default_factory=dict)     # existing nuclear-geometry check (fixed frame), recorded only

    def T(self, which="final"):
        return (self.M_final, self.t_final) if which == "final" else (self.M_search, self.t_search)

    def to_dict(self):
        return json_safe(asdict(self))


def _from_engine_records(cands: list, searches_meta: dict) -> list[CandidatePose]:
    out = []
    for c in cands:
        meta = searches_meta.get(c["source"], {})
        ff = (c.get("validation") or {}).get("fixed_frame") or {}
        out.append(CandidatePose(
            id=c["id"], source=c["source"], rank=int(c["rank"]), verdict=meta.get("verdict", ""), margin=meta.get("margin"),
            score=(c.get("search") or {}).get("score"),
            M_search=np.asarray(c["T_search"]["M"], float), t_search=np.asarray(c["T_search"]["t"], float),
            M_final=np.asarray(c["T_final"]["M"], float), t_final=np.asarray(c["T_final"]["t"], float),
            flags=list(c.get("flags") or []),
            c3={k: ff.get(k) for k in ("le_3um", "null_le_3um", "inlier_over_null", "geometric_pass", "n_eval")}))
    return out


def from_artifact(art: dict) -> list[CandidatePose]:
    return _from_engine_records(art["candidates"], art.get("localization", {}))


def from_benchmark_result(path: Path, gate: str = "same") -> list[CandidatePose]:
    """2026-11-11 benchmark (analysis/adaptive-registration-validation-20261111/runs/*/result.json).  gate='same' gives
    the ICP-refined poses (T_final); under 'adjacent' the engine keeps the search pose."""
    res = json.loads(Path(path).read_text())
    return _from_engine_records(res["gates"][gate]["candidates"], res["searches"])


def run_c1_topk(fixed_in, moving_in, cfg: W.WorkflowConfig | None = None, log=lambda m: None, relation="same"):
    """C1 + C2 through the unchanged engine functions; returns (candidates, searches, (fixed, moving) Sides)."""
    cfg = cfg or W.WorkflowConfig(localization="points")
    searches, st, (fixed, moving), _, _, _ = W.localize(fixed_in, moving_in, cfg, None, log)
    names = (fixed_in.frame_name, moving_in.frame_name)
    cands = W.refine_and_validate(searches, fixed, moving, cfg, log, names, relation)
    meta = {k: {kk: v[kk] for kk in ("verdict", "margin", "runtime_s", "n_poses", "coarse_top") if kk in v} for k, v in searches.items()}
    return _from_engine_records(cands, meta), meta, (fixed, moving)
