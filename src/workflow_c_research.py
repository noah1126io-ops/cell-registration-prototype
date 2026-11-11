"""Workflow C research mode: use the research Workflow C (src/registration_core: region localization -> overlap ICP ->
validation) as the GLOBAL step of production Workflow C.

Off by default.  Only when CELLREG_RESEARCH_MODE is on AND the user explicitly selects an ACCEPTED, fully automatic
research artifact does the STEP 2 affine come from it instead of ``estimate_affine_with_y_flip``.  The fine stage, its
safety gates, the result artifact and Workflow D are unchanged.  Unaccepted candidates, assisted runs and manual
transforms cannot be selected.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from src.pointset_registration import AffineICPResult
from src.registration_core.workflow_d_bridge import check_handoff, he_to_if

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs/registration-explorer"
KEY = "workflow-c-research-global"
MAX_MEDIAN_RESIDUAL_UM = 10.0     # sanity check that the uploaded points are the ones the research run registered


def accepted_runs(runs_dir: Path = RUNS) -> list[dict]:
    out = []
    if not runs_dir.is_dir():
        return out
    for d in sorted(runs_dir.iterdir(), reverse=True):
        p = d / "artifact.json"
        if not p.is_file():
            continue
        try:
            art = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not check_handoff(art):
            out.append({"path": str(p), "run_id": art["run_id"], "sample": art["sample"], "art": art})
    return out


def he_pixel_um(art: dict) -> float:
    side = "fixed" if art["input"]["fixed"]["key"] == "HE" else "moving"
    um = art["input"][side].get("pixel_um")
    if not um:
        raise ValueError("H&E pixel spacing is not recorded in the research artifact")
    return float(um)


def affine_from_research(art: dict, he_px: np.ndarray, fixed_um: np.ndarray, image_width: float, image_height: float) -> AffineICPResult:
    """Production-shaped global affine (HE px -> fluorescence um, no flip) from an accepted research artifact."""
    why = check_handoff(art)
    if why:
        raise ValueError("research transform cannot be used: " + "; ".join(why))
    T = he_to_if(art)
    A = np.asarray(T.M, float) * he_pixel_um(art)
    t = np.asarray(T.t, float)
    moved = np.asarray(he_px, float) @ A.T + t
    d, _ = cKDTree(np.asarray(fixed_um, float)).query(moved)
    med = float(np.median(d))
    if med > MAX_MEDIAN_RESIDUAL_UM:
        raise ValueError(f"the uploaded nuclei do not match this research registration (median NN {med:.1f} um > {MAX_MEDIAN_RESIDUAL_UM} um); "
                         "check that the same sample / HE pixel coordinates are used")
    return AffineICPResult(affine_matrix=A, translation=t, transformed_points=moved, flip_x=False, flip_y=False,
                           image_width=float(image_width), image_height=float(image_height), mean_residual=float(np.mean(d)),
                           median_residual=med, n_pairs=int(len(moved)), success=True,
                           message=f"research Workflow C (registration_core) accepted run {art['run_id']} "
                                   f"[{art['decision']['status']}, {art['decision']['selected_candidate']}] used as the global affine")


def research_global_panel(st, tr):
    """UI (research mode only): explicit opt-in.  Returns the selected artifact dict or None."""
    with st.expander(tr("研究モード：Global位置合わせを『研究Workflow C』の採択済み結果にする", "Research mode: use an accepted research Workflow C result as the global affine")):
        st.caption(tr("対応領域の探索 → 重なり領域でのICP → 事前に固定した基準による検証、を経て採択された結果だけを選べます。"
                      "未採択の候補・注釈を使った結果・手動変換は選べません。STEP 3以降（fine stageとその安全ゲート）は変わりません。",
                      "Only results accepted by region localization -> overlap ICP -> prespecified validation can be chosen."))
        on = st.checkbox(tr("研究Workflow Cの結果を使う", "Use a research Workflow C result"), key=f"{KEY}-on")
        runs = accepted_runs()
        if not on:
            return None
        if not runs:
            st.info(tr("採択済み（完全自動）の研究Workflow Cの結果がありません。Registration Explorerの『新規実行』で作成できます。",
                       "No accepted research Workflow C result yet."))
            return None
        pick = st.selectbox(tr("採択済みの結果", "Accepted result"), [r["run_id"] for r in runs],
                            format_func=lambda i: next(f"{r['sample']} / {r['run_id']} / {r['art']['decision']['status']} / {r['art']['decision']['selected_candidate']}"
                                                       for r in runs if r["run_id"] == i), key=f"{KEY}-pick")
        art = next(r["art"] for r in runs if r["run_id"] == pick)
        st.caption(tr(f"H&E pixel spacing {he_pixel_um(art):.5f} µm/px を使って、H&E pixel → 蛍光 µm のaffineに変換します（反転なし）。",
                      "Converted to an HE px -> um affine (no flip)."))
        return art
