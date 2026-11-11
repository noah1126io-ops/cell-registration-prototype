"""Research-mode registration job started from the Registration Explorer (runs on a Slurm compute node).

  python scripts/run_registration_explorer_job.py --run-dir runs/registration-explorer/<run_id>

Reads <run_dir>/config.json, writes <run_dir>/result.json and status.json.  Calls the research E0x code
(analysis/e0x-difficulty-and-web-integration-20261020/e0xlab.py) or the existing A2 (init_lib.method_a2) unmodified.
Nothing here touches production Workflow C/D outputs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.workflow_c_slurm import update_status, write_json_atomic  # noqa: E402

LAB = Path(os.environ.get("CELLREG_E0X_LAB_DIR", str(REPO / "analysis/e0x-difficulty-and-web-integration-20261020")))


def _json_safe(v):
    import numpy as np
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, np.ndarray):
        return _json_safe(v.tolist())
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def run_engine(run_dir: Path, cfg: dict):
    """Research Workflow C on the shared engine (src/registration_core): C1 localize -> C2 refine -> C3 validate -> C4 artifact."""
    from src.registration_core import Constraints, WorkflowConfig, run_workflow, write_artifact
    from src.registration_explorer.catalog import load_manifest, side_input
    S = load_manifest(Path(cfg["manifest"]))[cfg["sample"]]
    fixed = side_input(S, cfg["fixed"], use_image=cfg.get("use_images", True))
    moving = side_input(S, S.other(cfg["fixed"]), use_image=cfg.get("use_images", True))
    cons = Constraints(**cfg["constraints"]) if cfg.get("constraints") else None
    log_path = run_dir / "progress.log"

    def log(msg):
        with open(log_path, "a") as f:
            f.write(msg + "\n")
        print(msg, flush=True)
    art = run_workflow(S.sample_id, fixed, moving, WorkflowConfig(localization=cfg["localization"], **({"image_feature": cfg["image_feature"]} if cfg.get("image_feature") else {})), cons, log=log, run_id=run_dir.name)
    write_artifact(art, run_dir / "artifact.json")
    d = art["decision"]
    update_status(run_dir, "completed", verdict=d["status"], accepted=d["accepted"], failure_stage=d.get("failure_stage"))


def run(run_dir: Path):
    import numpy as np
    cfg = json.loads((run_dir / "config.json").read_text())
    update_status(run_dir, "running")
    if cfg.get("engine") == "registration_core":
        return run_engine(run_dir, cfg)
    sys.path.insert(0, str(LAB))
    import e0xlab  # noqa: F401  (adds oir / init_lib to sys.path)
    import init_lib
    from oir.frames import Affine2D
    from oir.pipeline import RunResult, Side, evaluate
    from src.registration_explorer.catalog import load_manifest, load_nuclei

    samples = load_manifest(Path(cfg["manifest"]))
    S = samples[cfg["sample"]]
    fixed_key = cfg["fixed"]; moving_key = S.other(fixed_key)
    fm, mm = S.modality(fixed_key), S.modality(moving_key)
    fpts, _ = load_nuclei(fm); mpts, _ = load_nuclei(mm)
    method = cfg["method"]
    if method == "A2":
        res = init_lib.method_a2(mpts, fpts)
        det = init_lib.detector(res.M, res.t, mpts, fpts, basin_margin=res.diagnostics.get("basin_margin", np.nan))
        cands = []
        for rank, (obj, M, t, hyp) in enumerate(res.refined[:6]):
            from oir.search import pose_matrix
            Ms = pose_matrix(hyp["chirality"], hyp["theta"], hyp["scale"])
            cands.append({"rank": rank, "chirality": hyp["chirality"], "theta_deg": hyp["theta"], "scale": hyp["scale"], "t_um": hyp["translation"],
                          "score_z": None, "ncc": hyp.get("score"), "relative_gap_to_top": None,
                          "T_search": {"M": Ms.tolist(), "t": list(hyp["translation"])}, "T_final": {"M": np.asarray(M).tolist(), "t": np.asarray(t).tolist()},
                          "final_eval": {"geometric_pass": det["detector_status"] == "PASS" if rank == 0 else None}, "icp_objective": obj})
        rec = {"sample": S.sample_id, "variant": "A2", "canvas": "fixed nuclei bbox (A2)", "icp": "A2 production refine",
               "fixed": fixed_key, "moving": moving_key, "verdict": {"PASS": "UNIQUE", "AMBIGUOUS": "AMBIGUOUS"}.get(det["detector_status"], "FAIL"),
               "margin": res.diagnostics.get("basin_margin"), "accepted": det["detector_status"] == "PASS", "detector": det,
               "safety": "UNIQUE / Accepted" if det["detector_status"] == "PASS" else det["detector_status"], "candidates": cands,
               "runtime_s": res.runtime_s if hasattr(res, "runtime_s") else None}
    else:
        fe, me = fm.extent(), mm.extent()
        if fm.pixel_um is None or mm.pixel_um is None or fe is None or me is None:
            raise ValueError("E0x needs known pixel spacing and imaged extents on both sides (UNKNOWN here)")
        rec = e0xlab.run_variant(S.sample_id, cfg.get("variant", "X3"), Side(fixed_key, fpts, fe, None), Side(moving_key, mpts, me, None))
    # IF-frame evaluation (same evaluator as the saved research metrics) when both extents are known
    he = S.modality("HE"); iff = S.modality("IF")
    if he.extent() and iff.extent() and he.pixel_um and iff.pixel_um:
        hp, _ = load_nuclei(he); ip, _ = load_nuclei(iff)
        sh, si = Side("HE", hp, he.extent(), None), Side("IF", ip, iff.extent(), None)
        for c in rec["candidates"]:
            if not c.get("T_final"):
                continue
            Tf = Affine2D(c["T_final"]["M"], c["T_final"]["t"], moving_key, fixed_key)
            Ts = Affine2D(c["T_search"]["M"], c["T_search"]["t"], moving_key, fixed_key)
            ev = evaluate(RunResult(S.sample_id, method, "P0", fixed_key, moving_key, Tf, Ts), sh, si, None, None)
            c["eval_if_frame"] = {"final": ev["final"], "null": ev["null"], "inlier_over_null": ev["inlier_over_null"],
                                  "geometric_pass": ev["geometric_pass"], "n_if_eval": ev["n_if_eval"]}
    rec.update({"run_config": cfg, "producer": "scripts/run_registration_explorer_job.py (research mode)"})
    write_json_atomic(run_dir / "result.json", _json_safe(rec))
    update_status(run_dir, "completed", verdict=rec.get("verdict"), accepted=bool(rec.get("accepted")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    rd = Path(a.run_dir).resolve()
    try:
        run(rd)
    except BaseException as exc:  # noqa: BLE001 - always leave a terminal status
        (rd / "error.txt").write_text(traceback.format_exc())
        update_status(rd, "failed", error=f"{type(exc).__name__}: {exc}")
        raise
