"""Registration Explorer: research tab "核対応（研究）" for the Region-Cell Matching study (2026-11-13).

Disabled unless CELLREG_REGION_CELL_MATCHING=1.  Read-only: it shows the saved results of
analysis/region-cell-matching-20261113 (runs/p1_<case>.json, viz/<case>.json) and never computes, never writes the
research store and never sends anything to Workflow C/D.  Local deformation is a research preview only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

STUDY = Path(__file__).resolve().parents[2] / "analysis/region-cell-matching-20261113"
ENV = "CELLREG_REGION_CELL_MATCHING"
TAB_LABEL = "核対応（研究）"
KEY = "regx-rcm"
STATUS_COLORS = {"matched": "#1a9850", "ambiguous": "#fdae61", "unmatched": "#d73027"}
METHOD_LABELS = {"A": "A Affineのみ", "B": "B 相互最近傍＋TPS（既存アプリ）", "B'": "B′ 相互最近傍＋linear RBF",
                 "C": "C B2対応＋linear RBF", "D": "D UOT conservative（既存）", "E": "E B2対応×信頼度＋linear RBF"}


def enabled() -> bool:
    return os.environ.get(ENV, "") == "1"


def _cases_for(sample_id: str):
    out = []
    for p in sorted((STUDY / "runs").glob("p1_*.json")):
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        m = d.get("meta", {})
        if d["task"]["dataset"] == "REAL-D" and m.get("he_sample") == sample_id and m.get("if_sample") == sample_id:
            out.append((d["task"]["case"], d))
    return out


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return "はい" if v else "いいえ"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def candidate_table(pose: dict) -> pd.DataFrame:
    recs = pose["records"]
    order = pose["orders"]["excess"]
    rows = []
    for r in recs:
        exc = (r.get("B2_matched_frac") or 0) - (r.get("decoy_B2_matched_frac") or 0) if r.get("status") == "OK" else None
        rows.append({"候補": r["candidate"], "C1順位": f"{r['source']} {r['c1_rank'] + 1}位", "再評価順位": order.index(r["candidate"]) + 1,
                     "参照領域": "○" if r.get("correct_region") else "×", "状態": r.get("status"),
                     "重なりの核数（Fixed/Moving）": f"{r.get('n_fixed_overlap')}/{r.get('n_moving_overlap')}",
                     "B2採択率": _fmt(r.get("B2_matched_frac")), "decoy採択率": _fmt(r.get("decoy_B2_matched_frac")),
                     "差（主指標）": _fmt(exc), "graph support中央値": _fmt(r.get("graph_support_median")),
                     "UOT輸送量比": _fmt(r.get("uot_mass_ratio_to_source")), "C3 ≤3µm": _fmt(r.get("c3_le_3um")),
                     "B2対応の局所ずれ中央値(µm)": _fmt(r.get("B2_local_dev_median_um"), 2), "B2対応の被覆率": _fmt(r.get("B2_coverage_cells"))})
    return pd.DataFrame(rows).sort_values("再評価順位")


def _scatter(ax, fixed, moving, status=None, pairs=None, title=""):
    ax.scatter(fixed[:, 0], fixed[:, 1], s=4, c="#9e9e9e", label="Fixed 核", linewidths=0)
    if status is None:
        ax.scatter(moving[:, 0], moving[:, 1], s=4, c="#2b8cbe", label="Moving 核（変換後）", linewidths=0)
    else:
        for k, col in STATUS_COLORS.items():
            sel = np.asarray(status) == k
            ax.scatter(moving[sel, 0], moving[sel, 1], s=5, c=col, label=f"{k}（{int(sel.sum())}）", linewidths=0)
    if pairs is not None and len(pairs):
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(pairs, colors="#1a9850", linewidths=0.6, alpha=0.8))
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_title(title, fontsize=9); ax.legend(fontsize=7, loc="upper right", markerscale=2)


def render(sample_id: str) -> None:
    """Japanese labels need a CJK font; rc_context keeps the setting local (no change to other pages' figures)."""
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.family": ["Noto Sans CJK JP", "IPAexGothic", "DejaVu Sans"]}):
        _render(sample_id)


def _render(sample_id: str) -> None:
    import matplotlib.pyplot as plt

    st.info("研究用プレビューです。核の対応は生物学的に同じ細胞であることを示しません。局所変形はWorkflow Dへ送られません。"
            "このタブは保存済みの研究結果を表示するだけで、計算・保存は行いません。")
    cases = _cases_for(sample_id)
    if not cases:
        st.warning(f"このサンプル（{sample_id}）の研究結果がありません（{STUDY.name}/runs）。")
        return
    names = [c for c, _ in cases]
    c1, c2 = st.columns(2)
    case = c1.selectbox("ケース", names, index=next((i for i, n in enumerate(names) if n.endswith("_orig")), 0), key=f"{KEY}-case")
    d = dict(cases)[case]
    pose_key = c2.selectbox("候補の姿勢", list(d["poses"]), format_func={"final": "C2後（T_final）", "search": "探索の姿勢（T_search）"}.get,
                            key=f"{KEY}-pose")
    pose = d["poses"][pose_key]
    m = d["meta"]
    st.caption(f"{m.get('kind')} / 期待: {m.get('expected')} / 本番の判定: {m.get('production_status')}（{m.get('production_selected')}、"
               f"ゲート {m.get('production_gate')}）")
    dec = pose["decision"]
    msg = (f"再評価の1位: **{dec.get('top')}**（主指標 {_fmt(dec.get('s_top'))}、2位 {_fmt(dec.get('s_second'))}） — "
           + ("Graph/UOTの事前指定条件を満たす" if dec.get("supported") else "条件を満たさない: " + ", ".join(dec.get("reasons") or [])))
    (st.success if dec.get("supported") else st.warning)(msg)
    st.markdown("**C1候補とGraph/UOT再評価の比較**（主指標 = B2採択率 − 鏡像decoyの採択率。Graph/UOTが好むことは独立した成功の証拠ではありません）")
    st.dataframe(candidate_table(pose), use_container_width=True, hide_index=True)

    viz_p = STUDY / "viz" / f"{case}__{pose_key}.json"
    viz = json.loads(viz_p.read_text()) if viz_p.is_file() else None
    if viz is None:
        st.caption("核の表示データ（viz）はこのケースにはありません。")
    else:
        ids = [c["id"] for c in viz["candidates"]]
        cid = st.selectbox("表示する候補", ids, key=f"{KEY}-cand-{case}-{pose_key}")
        cv = next(c for c in viz["candidates"] if c["id"] == cid)
        fx, mv = np.asarray(cv["fixed"]), np.asarray(cv["moving"])
        pairs = [[mv[i], fx[j]] for i, j in cv["pairs"]]
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        _scatter(axes[0], fx, mv, title="Global変換後の点群")
        _scatter(axes[1], fx, mv, cv["status"], pairs, title="B2の状態と対応線（matched）")
        st.pyplot(fig, clear_figure=True)
        rec = next(r for r in pose["records"] if r["candidate"] == cid)
        diag = {k: rec.get(k) for k in ("B0_matched_n", "B1_matched_n", "B3_matched_n", "B2_matched_n", "B2_ambiguous_n", "B2_unmatched_n",
                                       "uot_converged", "uot_entropy_median", "uot_row_mass_ratio_median", "uot_objective_per_row",
                                       "graph_support_ge05_frac", "B2_duplicate_destinations", "B2_local_dev_le3_frac", "decoy_graph_support_median")}
        st.markdown("**UOT / Graph の対応診断**（UOTの輸送量は同一細胞の確率ではありません）")
        st.dataframe(pd.DataFrame([{"指標": k, "値": _fmt(v)} for k, v in diag.items()]), hide_index=True)

    st.markdown("**局所変形（研究用プレビュー、Global変換は固定）**")
    for tag, p2 in pose.get("phase2", {}).items():
        st.markdown(f"- {'C1の1位' if tag == 'c1_top1' else '再評価の1位'}: {p2['candidate']}（参照領域 {'○' if p2['correct_region'] else '×'}）")
        rows = [{"方式": METHOD_LABELS.get(r["method"], r["method"]), "適用": _fmt(r["applied"]), "棄却理由": r.get("rejection_reason") or "",
                 "アンカー数": _fmt(r.get("anchors_n")), "最大変位(µm)": _fmt(r.get("attempted_displacement_max_um"), 1),
                 "Jacobian最小": _fmt(r.get("attempted_jacobian_min"), 2), "foldover": _fmt(r.get("attempted_foldover_fraction"), 4),
                 "NN中央値 前→後(µm)": f"{_fmt(r.get('nn_sym_median_before'), 2)} → {_fmt(r.get('nn_sym_median_after'), 2)}",
                 "ランドマーク残差 Affine→変形後(µm)": (f"{_fmt(r.get('landmark_residual_affine_median_um'), 1)} → "
                                                f"{_fmt(r.get('landmark_residual_after_median_um'), 1)}") if r.get("landmark_n") else "—"}
                for r in p2["rows"]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if viz is not None and viz.get("deformation"):
        dm = st.selectbox("変位ベクトルを表示する方式", list(viz["deformation"]), format_func=lambda k: METHOD_LABELS.get(k, k), key=f"{KEY}-def-{case}")
        dv = viz["deformation"][dm]
        fig, ax = plt.subplots(figsize=(6, 6))
        fx = np.asarray(viz["deformation_fixed"])
        ax.scatter(fx[:, 0], fx[:, 1], s=3, c="#9e9e9e", linewidths=0, label="Fixed 核")
        after = np.asarray(dv["points_after"])
        ax.scatter(after[:, 0], after[:, 1], s=3, c="#2b8cbe", linewidths=0, label="Moving 核（局所変形後）")
        q = np.asarray(dv["quiver"])
        if len(q):
            ax.quiver(q[:, 0], q[:, 1], q[:, 2], q[:, 3], angles="xy", scale_units="xy", scale=0.2, color="#d73027", width=0.002)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.legend(fontsize=7)
        ax.set_title(f"{METHOD_LABELS.get(dm, dm)}: 適用={dv['applied']}（矢印は5倍に拡大）", fontsize=9)
        st.pyplot(fig, clear_figure=True)
    st.caption("失敗理由の読み方: 『excess』=採択率がdecoyより十分高くない、『ratio_to_decoy』=decoyの3倍未満、『gap』=2位との差が小さい。"
               "隣接切片（031-8_6）では核が別の細胞なので、核対応による支持は期待できません。")
