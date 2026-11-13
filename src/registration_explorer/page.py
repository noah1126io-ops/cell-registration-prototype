"""Streamlit page: Registration Explorer (research mode only).

Shows saved research registrations (no recomputation), lets the user compare methods and candidates, and
optionally submits one research job to Slurm.  Nothing on this page feeds production Workflow C/D.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from . import catalog, imaging, jobs, manual, research_ui as rui, results, scene, viewer
from .. import research_store as rs
from . import reference_ui as refui
from . import region_cell_ui as rcm_ui   # research tab, disabled unless CELLREG_REGION_CELL_MATCHING=1 (2026-11-13)
from ..registration_core import reference as refreg

KEY = "regx"
SAFETY_STYLE = {"UNIQUE / Accepted": "success", "POINT_ONLY_FALLBACK": "warning", "AMBIGUOUS": "warning", "LOW_PEAK": "warning",
                "FAIL": "error", "IMAGE_UNUSABLE": "error", "UNVERIFIED": "warning", "MANUAL (UNVERIFIED)": "warning",
                "CONFLICT": "error", "NOT_ACCEPTED": "warning", "ASSISTED_REVIEW_REQUIRED": "warning",
                "TISSUE_LEVEL": "info"}
METHOD_ORDER = ["WF", "E0x", "A2", "E1", "E2", "E3", "ABL", "RUN", "NC", "MAN"]
NEEDS_IMAGES = {"E1": 1, "E2": 2, "E3": 2}
DIRS = {"HE": "H&E Fixed / 蛍光 Moving", "IF": "蛍光 Fixed / H&E Moving"}
# step sizes for manual adjustment: rotation (deg), scale factor, translation (frame units)
MANUAL_STEPS = {"粗": (5.0, 0.05, 50.0), "標準": (1.0, 0.01, 10.0), "細": (0.1, 0.001, 1.0)}
PIVOTS = {"moving": "Moving視野の中心（写像後）", "fixed": "Fixed視野の中心", "custom": "座標を指定"}


def _fmt(v, nd=3, unit=""):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, (list, tuple)):
        return "(" + ", ".join(f"{float(x):.1f}" for x in v) + ")" + (f" {unit}" if unit else "")
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, float, np.floating)):
        return f"{float(v):.{nd}g}{(' ' + unit) if unit else ''}"
    return str(v)


@st.cache_data(ttl=20, show_spinner=False)
def _records(sample_id: str, manifest_mtime: float):
    S = catalog.load_manifest()[sample_id]
    return results.load_records(S)


def _store_records(sample_id: str):
    """Manual / landmark transforms saved in the research store -> MANUAL (UNVERIFIED) records."""
    out = []
    for t in rui.store().transforms(sample_id):
        if t["kind"] not in ("manual", "landmark"):
            continue
        d = json.loads(t["detail_json"] or "{}")
        out.append(results.ResultRecord(
            rid=f"store:{t['id']}", sample=sample_id, method="MAN", config=f"{t['label'] or t['id']}（{'手動' if t['kind'] == 'manual' else '対応点'}）",
            fixed=t["fixed"], moving=t["moving"], M=np.asarray(json.loads(t["M_json"]), float), t=np.asarray(json.loads(t["t_json"]), float),
            verdict="MANUAL", margin=None, accepted=False, status_flags=["MANUAL_UNVERIFIED"],
            failure_reasons=["手動 / 対応点による変換。事前の採択規則による判定を経ていない"], fallback_source=None,
            metrics={"population": "not evaluated by the research pipeline; see UI-recomputed diagnostics"}, candidates=[], source="research-store",
            files=[str(rui.store().path)], artifact={**t, "detail": d}, note=f"parent {t['parent_id']}; by {t['created_by']} {t['created_at']}"))
    return out


def _manifest():
    p = catalog.manifest_path()
    if not p.is_file():
        st.error(f"sample manifest not found: {p}（環境変数 CELLREG_EXPLORER_MANIFEST で指定できます）")
        st.stop()
    return catalog.load_manifest(p), p.stat().st_mtime


def _kv_table(d: dict):
    rows = [{"指標": k, "値": _fmt(v)} for k, v in (d or {}).items() if k != "population"]
    if not rows:
        st.caption("（記録された指標なし）")
        return
    st.table(pd.DataFrame(rows).set_index("指標"))


def _metadata_table(S: catalog.SampleInfo, avail: dict, recs):
    rows = []
    for k, m in S.modalities.items():
        ext = m.extent()
        try:
            n = len(catalog.load_nuclei(m)[0])
        except (OSError, ValueError, KeyError):
            n = None
        rows.append({"modality": m.label, "画像": "あり" if avail[k]["image"] else f"なし（{avail[k]['image_reason']}）",
                     "核座標": f"あり ({n})" if n is not None else "なし",
                     "画像サイズ (px, level0)": f"{m.shape_px[1]} × {m.shape_px[0]}" if m.shape_px else "UNKNOWN",
                     "pixel spacing": f"{m.pixel_um:.4f} µm/px" if m.pixel_um else "UNKNOWN",
                     "physical FOV": (f"{ext[0]:.0f} × {ext[1]:.0f} µm" if m.pixel_um else f"UNKNOWN（frame範囲 {ext[0]:.0f} × {ext[1]:.0f} px、spacing不明）")
                                     if ext else "UNKNOWN",
                     "原点": m.origin, "channels": m.channels or "UNKNOWN", "核の出所": (m.nuclei or {}).get("source", "UNKNOWN")})
    st.table(pd.DataFrame(rows).set_index(pd.DataFrame(rows).columns[0]))
    with st.expander("Sample provenance / coordinate system", expanded=False):
        st.write(f"**Coordinate system**: {S.coordinate_system}")
        st.json(S.provenance)
        st.caption(f"既存Registration結果: {len([r for r in recs if r.method != 'ERR'])} 件（保存済みartifact / ablation / explorer runs）")


def _method_availability(m, avail, recs):
    n_img = sum(1 for v in avail.values() if v["image"])
    have = any(r.method == m for r in recs)
    if m in NEEDS_IMAGES and n_img < NEEDS_IMAGES[m]:
        return False, f"画像が{NEEDS_IMAGES[m]}枚必要（利用可能 {n_img}）"
    if not have:
        return False, "保存済み結果なし"
    return True, ""


def _safety_panel(rec: results.ResultRecord, effective: results.ResultRecord | None):
    lab = rec.safety_label()
    msg = (f"**{lab}** — {results.SAFETY_TEXT.get(lab, '')}（探索の判定 {rec.verdict}" + (f", margin {rec.margin:.3f}" if rec.margin is not None else "") + "）")
    getattr(st, SAFETY_STYLE.get(lab, "info"))(msg)
    if rec.status_flags:
        st.caption("status: " + " ; ".join(rec.status_flags))
    if rec.failure_reasons and any(rec.failure_reasons):
        st.caption("理由: " + " | ".join(x for x in rec.failure_reasons if x))
    if rec.accepted:
        st.caption(f"採択規則（事前固定・UIでは変更不可）: {results.ACCEPT_RULE}")
    else:
        st.warning("PREVIEW ONLY — この変換は正式に採択されていません。Workflow Dへは渡しません。")
    if effective is not None and effective is not rec:
        st.info(f"Point-only fallback: 最終出力は **{effective.config}**（{effective.safety_label()}）です。下の『表示する変換』で切り替えられます。")


def _candidate_options(rec):
    opts = [("final", "最終変換（この結果の正式/プレビュー変換）")]
    for c in rec.candidates:
        z = _fmt(c.score_z, 3); g = _fmt(c.gap_to_top, 2)
        src = f"［{rui.results_source_label(c)}］ " if c.source else ""
        opts.append((c.cand_id, f"{src}候補 {c.cand_id}: θ={c.theta_deg:.1f}°, s={c.scale:.3f}, z={z}, 1位との差={g} [PREVIEW ONLY]"))
    return opts


def _oriented(M, t, rec, fixed_key):
    return results.oriented(M, t, rec.fixed, fixed_key)


def _manual_translation_range(fm, mm):
    """Half-width of the translation sliders: the larger imaged extent of either modality (point bbox if no extent)."""
    sizes = []
    for m in (fm, mm):
        e = m.extent()
        if e is None:
            xy = catalog.load_nuclei(m)[0]
            e = tuple(np.ptp(xy, axis=0)) if len(xy) else None
        if e is not None:
            sizes.append(max(e))
    r = max(sizes) if sizes else 1e4
    return float(np.ceil(r / 100.0) * 100.0)


def _manual_sync(dst, src):
    st.session_state[dst] = float(st.session_state[src])


def _manual_slider(ck, default, step_name, label, rng, step, fmt):
    """Slider + number box bound to one stored value (`ck`).  Widget keys include the step mode, so changing the
    step builds fresh widgets that are seeded from the stored value instead of losing it."""
    if ck not in st.session_state:
        st.session_state[ck] = default
    v = float(st.session_state[ck])
    lo, hi = rng
    sk, nk = f"{ck}-sl-{step_name}", f"{ck}-ni-{step_name}"
    st.session_state[sk] = float(min(max(v, lo), hi))
    st.session_state[nk] = v
    a, b = st.columns([3, 1], vertical_alignment="bottom")
    a.slider(label, min_value=lo, max_value=hi, step=step, format=fmt, key=sk, label_visibility="collapsed",
             on_change=_manual_sync, args=(ck, sk))
    b.number_input(label, min_value=lo, max_value=hi, step=step, format=fmt, key=nk, on_change=_manual_sync, args=(ck, nk))
    return float(st.session_state[ck])


def _manual_controls(S, fixed_key, M, t, base):
    """Rotation / scale / translation on top of the displayed transform.  Returns (M, t, info or None)."""
    fm, mm = S.modality(fixed_key), S.modality(S.other(fixed_key))
    pref = f"{KEY}-man-{S.sample_id}-{fixed_key}-{base['key']}"
    keys = {"rot": f"{pref}-rot", "scale": f"{pref}-scale", "dx": f"{pref}-dx", "dy": f"{pref}-dy"}
    defaults = {"rot": 0.0, "scale": 1.0, "dx": 0.0, "dy": 0.0}
    with st.expander("手動調整（回転・拡大縮小・平行移動）", expanded=bool(st.session_state.get(f"{pref}-on"))):
        if not st.toggle("手動調整を有効にする", key=f"{pref}-on"):
            st.caption("表示中の変換（基準）に、回転・拡大縮小・平行移動を追加できます。結果は未検証の手動変換で、採択にはなりません。")
            st.session_state.pop(f"{KEY}-man-live", None)
            return M, t, None
        st.warning("MANUAL — 手動で調整した変換は未検証です。採択規則による判定を経ていないため、PREVIEW ONLYとして扱い、Workflow Dへは渡しません。")
        st.caption(f"基準: {base['label']}")
        c1, c2, c3 = st.columns([1.2, 1.6, 1])
        step = c1.radio("刻み", list(MANUAL_STEPS), index=1, horizontal=True, key=f"{KEY}-man-step")
        pv = c2.radio("回転・拡大縮小の中心", list(PIVOTS), format_func=PIVOTS.get, horizontal=True, key=f"{pref}-pivot")
        reset = c3.button("調整をリセット", key=f"{pref}-reset")
        if reset:
            for k, v in defaults.items():
                st.session_state[keys[k]] = v
        rs, ss, ts = MANUAL_STEPS[step]
        unit = fm.units
        tr = _manual_translation_range(fm, mm)
        spec = {  # label, (slider min, max), step, number format
            "rot": ("回転 Δθ (°、画面上で時計回りが正)", (-180.0, 180.0), rs, "%.3f"),
            "scale": ("拡大縮小 ×", (0.25, 4.0), ss, "%.4f"),
            "dx": (f"平行移動 Δx ({unit})", (-tr, tr), ts, "%.2f"),
            "dy": (f"平行移動 Δy ({unit})", (-tr, tr), ts, "%.2f"),
        }
        st.caption("スライダーをドラッグするか、選択して矢印キーで『刻み』ずつ動かせます。右の欄には数値を直接入力できます。")
        vals = {k: _manual_slider(keys[k], defaults[k], step, *spec[k]) for k in keys}
        rot, scl, dx, dy = vals["rot"], vals["scale"], vals["dx"], vals["dy"]
        if pv == "moving":
            me = mm.extent()
            if me is not None:
                pivot = scene.apply(M, t, np.array([[me[0] / 2, me[1] / 2]]))[0]
            else:
                pivot = scene.apply(M, t, catalog.load_nuclei(mm)[0]).mean(0)
        elif pv == "fixed":
            fe = fm.extent() or (0.0, 0.0)
            pivot = np.array([fe[0] / 2, fe[1] / 2])
        else:
            p1, p2 = st.columns(2)
            pivot = np.array([p1.number_input(f"中心 x ({unit})", value=0.0, step=ts, key=f"{pref}-px"),
                              p2.number_input(f"中心 y ({unit})", value=0.0, step=ts, key=f"{pref}-py")])
        adj = manual.Adjustment(float(rot), float(scl), float(dx), float(dy), (float(pivot[0]), float(pivot[1])))
        M2, t2 = manual.compose(M, t, adj)
        p = results.params(M2, t2)
        st.caption(f"中心 ({pivot[0]:.1f}, {pivot[1]:.1f}) {unit}（y軸は下向き）。"
                   f" 調整後の変換 ({S.other(fixed_key)}→{fixed_key}): scale {p['scale']:.4f}, 回転 {p['rotation_deg']:.2f}°, "
                   f"t=({p['tx']:.1f}, {p['ty']:.1f})。下の『UI再計算の診断』が調整後の値です。")
    st.session_state[f"{KEY}-man-live"] = {"sample": S.sample_id, "fixed": fixed_key, "M": M2, "t": t2, "label": f"手動調整（基準: {base['label']}）"}
    info = {"adj": adj, "base": base, "pref": pref}
    return M2, t2, info


def _manual_save(S, fixed_key, M, t, info, diag):
    base = info["base"]
    pref = info["pref"]
    st.markdown("**手動調整した変換の保存**（未検証の手動変換として記録。採択にはなりません）")
    c1, c2 = st.columns([1, 2])
    name = c1.text_input("名前", key=f"{pref}-name", placeholder="例: 8_6_候補1_微調整")
    note = c2.text_input("メモ", key=f"{pref}-note")
    base_rec = {k: v for k, v in base.items() if k not in ("M", "t", "key")}
    base_rec.update({"M": np.asarray(base["M"], float).tolist(), "t": np.asarray(base["t"], float).tolist()})
    diag_rec = {k: v for k, v in (diag or {}).items()}
    rec = manual.build_record(sample=S.sample_id, fixed=fixed_key, moving=S.other(fixed_key), base=base_rec, adj=info["adj"], M=M, t=t,
                              note=note, name=name, diagnostics=diag_rec)
    b1, b2 = st.columns(2)
    if b1.button("保存する", key=f"{pref}-save", disabled=info["adj"].is_identity()):
        try:
            sid = rui.ensure_session(S.sample_id)
            tid = rui.store().add_transform(sample_id=S.sample_id, session_id=sid, kind="manual", fixed=fixed_key, moving=S.other(fixed_key),
                                            M=M, t=t, status="MANUAL_UNVERIFIED", label=name or rec["id"], note=note,
                                            run_id=rui._run_id(base["result_id"]) if base.get("result_id") else None,
                                            candidate_id=base.get("candidate"),
                                            detail={"base": base_rec, "adjustment": rec["adjustment"], "ui_diagnostics": diag_rec})
            st.success(f"保存しました: {tid}（研究ストア {rui.store().path}）。元の自動推定は変更していません。"
                       "『Registration方式』で『手動調整した変換（未検証）』を選ぶと再表示できます。")
        except (OSError, ValueError) as exc:
            st.error(f"保存できませんでした: {exc}")
    b2.download_button("JSONをダウンロード（保存せずに）", json.dumps(rec, indent=1, ensure_ascii=False, default=float),
                       file_name=f"{S.sample_id}_{rec['id']}_MANUAL_UNVERIFIED.json", mime="application/json", key=f"{pref}-dl")
    if info["adj"].is_identity():
        st.caption("調整量がすべて0のため、保存はできません（基準の変換と同じです）。")


def _handle_viewer_event(ev, S, fixed_key, M, t, use, cand):
    """Events from the interactive viewer (each handled once, by nonce)."""
    if not ev or not isinstance(ev, dict):
        return
    seen = st.session_state.setdefault(f"{KEY}-seen-nonce", set())
    if ev.get("nonce") in seen:
        return
    seen.add(ev.get("nonce"))
    kind = ev.get("kind")
    pk = f"{KEY}-pending-{S.sample_id}-{fixed_key}"
    if kind == "pick" and ev.get("candidate"):
        st.session_state[f"{KEY}-pick-pending"] = (use.rid, ev["candidate"])
        st.rerun()
    elif kind == "landmarks":
        Mi = np.linalg.inv(np.asarray(M, float)); ti = -Mi @ np.asarray(t, float)
        pairs = [{"fixed": [float(x) for x in p["fixed"]], "moving": (Mi @ np.asarray(p["moving_display"], float) + ti).tolist(),
                  "moving_display": p["moving_display"]} for p in ev.get("pairs", [])]
        st.session_state.setdefault(pk, {}).setdefault("landmarks", []).extend(pairs)
    elif kind == "roi":
        x0, y0, x1, y1 = ev["rect"]
        st.session_state.setdefault(pk, {})["roi"] = [[min(x0, x1), min(y0, y1)], [max(x0, x1), min(y0, y1)], [max(x0, x1), max(y0, y1)], [min(x0, x1), max(y0, y1)]]


def _pending_annotations(S, fixed_key, M, t):
    """Landmarks / ROI clicked in the viewer: review, compute a landmark transform, save as annotations."""
    pk = f"{KEY}-pending-{S.sample_id}-{fixed_key}"
    pend = st.session_state.get(pk) or {}
    lms, roi = pend.get("landmarks") or [], pend.get("roi")
    if not lms and not roi:
        st.caption("ビューアー左の『クリック操作』で、対応点（ランドマーク）や領域（ROI）を画像上で指定できます。指定した座標は保存して次の実験に使えます。")
        return
    moving_key = S.other(fixed_key)
    with st.container(border=True):
        if lms:
            st.markdown(f"**指定した対応点** {len(lms)} 組（Fixed = {fixed_key} frame、Moving = {moving_key} frame。Moving座標は表示中の変換の逆写像で求めた値）")
            st.table(pd.DataFrame([{"#": i + 1, "Fixed x": f"{p['fixed'][0]:.1f}", "Fixed y": f"{p['fixed'][1]:.1f}",
                                    "Moving x": f"{p['moving'][0]:.1f}", "Moving y": f"{p['moving'][1]:.1f}"} for i, p in enumerate(lms)]).set_index("#"))
            st.caption("各対応点の種類と確信度を選べます（保存時に、両画像の画素座標・frame座標・画像のfingerprintと一緒に記録します。"
                       "共通の物理原点はUNKNOWNのまま扱います）。")
            meta = refui.landmark_editor(pk, len(lms))
            lab = st.text_input("対応点の名前", key=f"{pk}-lmname", placeholder="例: 8_6 腺管3点")
            recs, fps = refui.landmark_records(S, fixed_key, lms, meta)
            c1, c2, c3, c4 = st.columns(4)
            if c1.button("対応点を注釈として保存", key=f"{pk}-lmsave"):
                sid = rui.ensure_session(S.sample_id)
                aid = rui.store().add_annotation(sample_id=S.sample_id, session_id=sid, kind="landmarks", frame=f"{S.sample_id}:{fixed_key}",
                                                 geometry=[{"fixed": p["fixed"], "moving": p["moving"]} for p in lms],
                                                 payload={"fixed": fixed_key, "moving": moving_key, "n": len(lms), "landmarks": recs, "images": fps}, label=lab)
                st.success(f"保存しました（{aid}）。『新規実行』の注釈を使った探索で、初期位置として使えます。")
            if len(lms) >= 2:
                fit = refreg.fit_reference([p["fixed"] for p in lms], [p["moving"] for p in lms])
                Ml, tl, res = fit["M"], fit["t"], np.asarray(fit["residual_um"])
                pp = results.params(Ml, tl)
                st.caption(f"対応点から求めた相似変換（外れ値を除外、採用 {fit['n_inliers']} / {fit['n']} 組）：scale {pp['scale']:.4f}, 回転 {pp['rotation_deg']:.2f}°, "
                           f"t=({pp['tx']:.1f}, {pp['ty']:.1f})、残差 中央値 {fit['residual_median_um']:.1f} / 最大 {res.max():.1f}（{S.modality(fixed_key).units}）"
                           + (f"、leave-one-out 中央値 {fit['loo_median_um']:.1f}" if fit["loo_median_um"] is not None else "（3組以上で leave-one-out 誤差を表示）")
                           + (f"、held-out {fit['holdout']['median_um']:.1f}（{fit['holdout']['n_test']}点）" if fit["holdout"] else "（6組以上で held-out 誤差を表示）"))
                detail = {"landmarks": [{"fixed": p["fixed"], "moving": p["moving"]} for p in lms], "landmark_records": recs, "images": fps,
                          "residual": res.tolist(), "fit": {k: v for k, v in fit.items() if k not in ("M", "t")}}
                if c2.button("この変換を保存（対応点・未検証）", key=f"{pk}-lmtf"):
                    sid = rui.ensure_session(S.sample_id)
                    tid = rui.store().add_transform(sample_id=S.sample_id, session_id=sid, kind="landmark", fixed=fixed_key, moving=moving_key,
                                                    M=Ml, t=tl, status="MANUAL_UNVERIFIED", label=lab or "landmarks", detail=detail)
                    st.success(f"保存しました（{tid}）。『手動調整した変換（未検証）』として表示できます。")
                if c3.button("保存して参照変換に指定", key=f"{pk}-lmref", help="研究者が指定した暫定の参照変換として保存します（Ground Truthではありません）"):
                    sid = rui.ensure_session(S.sample_id)
                    tid = rui.store().add_transform(sample_id=S.sample_id, session_id=sid, kind="landmark", fixed=fixed_key, moving=moving_key,
                                                    M=Ml, t=tl, status="MANUAL_UNVERIFIED", label=lab or "reference landmarks",
                                                    detail={**detail, "role": "reference"})
                    refui.designate_reference(S.sample_id, tid, fixed_key, moving_key)
                    st.success(f"保存し、参照変換に指定しました（{tid}）。『対応領域の探索』タブで自動推定と比較できます。")
            if c4.button("対応点を消去", key=f"{pk}-lmclear"):
                pend.pop("landmarks", None); st.rerun()
        if roi:
            st.markdown("**指定した領域（ROI）**（Fixed frame）: " + ", ".join(f"({x:.0f}, {y:.0f})" for x, y in roi[::2]))
            c1, c2, c3, c4 = st.columns([1.4, 1.4, 1, 1])
            kind = c1.radio("ROIの意味", ["roi", "search_roi"], format_func=rs.ANNOTATION_KINDS.get, key=f"{pk}-roikind")
            lab = c2.text_input("名前", key=f"{pk}-roiname")
            if c3.button("ROIを保存", key=f"{pk}-roisave"):
                sid = rui.ensure_session(S.sample_id)
                aid = rui.store().add_annotation(sample_id=S.sample_id, session_id=sid, kind=kind, frame=f"{S.sample_id}:{fixed_key}", geometry=roi,
                                                 payload={"frame_key": fixed_key}, label=lab)
                st.success(f"保存しました（{aid}）。")
            if c4.button("ROIを消去", key=f"{pk}-roiclear"):
                pend.pop("roi", None); st.rerun()


def _transform_annotation_buttons(S, fixed_key, M, t, use, cand, before, man):
    """Turn the transform on screen into reusable annotations (exclusion / manual initial position)."""
    if before and man is None:
        return
    with st.expander("表示中の変換を、次の実験のための注釈として保存"):
        lab = st.text_input("名前", key=f"{KEY}-tfann-name", placeholder="例: 8_6 points#0 は誤り")
        rel = {"result_id": use.rid, "run_id": rui._run_id(use), "candidate_id": cand.cand_id if cand is not None else None, "manual": man is not None}
        c1, c2 = st.columns(2)
        if c1.button("この配置を『誤りの候補』として除外注釈に保存", key=f"{KEY}-tfann-ex"):
            aid = rui.save_annotation_from_transform(S.sample_id, "exclusion", M, t, fixed_key, S.other(fixed_key), lab, rel)
            st.success(f"保存しました（{aid}）。注釈を使った探索で選ぶと、この配置の近く（40 µm以内）の候補を除外します。")
        if c2.button("この配置を『手動の初期位置』として保存", key=f"{KEY}-tfann-init"):
            aid = rui.save_annotation_from_transform(S.sample_id, "manual_init", M, t, fixed_key, S.other(fixed_key), lab, rel)
            st.success(f"保存しました（{aid}）。注釈を使った探索で選ぶと、この配置の周辺（±10°、±5%、150 µm）を探索します。")
        st.caption("注釈は保存しただけでは次の自動実行に反映されません。『新規実行』で注釈を明示的に選んだ場合だけ使われます。")


def _manual_compare(S, fixed_key, base, M, t, diag):
    """Automatic (base) vs manual side by side, same population definition."""
    fm, mm = S.modality(fixed_key), S.modality(S.other(fixed_key))
    fpts, _ = catalog.load_nuclei(fm); mpts, _ = catalog.load_nuclei(mm)
    me = mm.extent()
    bd = scene.ui_diagnostics(fpts, scene.apply(base["M"], base["t"], mpts), fm.extent(),
                              scene.apply(base["M"], base["t"], scene.footprint(me)) if me else None)
    pb, pm = results.params(base["M"], base["t"]), results.params(M, t)
    rows = [{"": "自動推定（基準）", "scale": round(pb["scale"], 4), "回転 (°)": round(pb["rotation_deg"], 2), "tx": round(pb["tx"], 1), "ty": round(pb["ty"], 1),
             "NN median": _fmt(bd.get("nn_median")), "≤3": _fmt(bd.get("le_3")), "null ≤3": _fmt(bd.get("null_le_3")), "n": bd.get("n_eval")},
            {"": "手動調整後", "scale": round(pm["scale"], 4), "回転 (°)": round(pm["rotation_deg"], 2), "tx": round(pm["tx"], 1), "ty": round(pm["ty"], 1),
             "NN median": _fmt(diag.get("nn_median")), "≤3": _fmt(diag.get("le_3")), "null ≤3": _fmt(diag.get("null_le_3")), "n": diag.get("n_eval")}]
    st.markdown("**自動推定と手動調整の比較**（UI再計算、同じ定義。白い破線 = 自動推定の領域）")
    st.table(pd.DataFrame(rows).set_index(""))


def _viewer_tab(S, avail, rec, effective, fixed_key):
    c1, c2, c3 = st.columns([1.2, 1.6, 1.2])
    with c1:
        state = st.radio("状態", ["After（位置合わせ後）", "Before（位置合わせ前）"], key=f"{KEY}-state", horizontal=True)
        show_src = "this"
        if effective is not None and effective is not rec:
            show_src = st.radio("表示する変換", ["この手法の変換（Preview）", f"fallback: {effective.config}"], key=f"{KEY}-eff")
    use = effective if (effective is not None and show_src.startswith("fallback")) else rec
    with c2:
        opts = _candidate_options(use)
        ids = [o[0] for o in opts]
        cid = st.selectbox("候補 / 変換", ids, format_func=dict(opts).get, key=f"{KEY}-cand-{use.rid}")
        stage = "final"
        cand = next((c for c in use.candidates if c.cand_id == cid), None)
        if cand is not None and cand.M_final is not None:
            stage = st.radio("候補の段階", ["ICP後", "探索pose（ICP前）"], horizontal=True, key=f"{KEY}-stage-{use.rid}")
    with c3:
        res = st.selectbox("表示解像度", [3.2, 1.6, 0.8], index=1, format_func=lambda v: f"概観 {v} µm/px", key=f"{KEY}-res")
        he_mode = st.selectbox("H&E表示", ["rgb", "nuclear"], format_func={"rgb": "RGB", "nuclear": "核強調（反転グレー）"}.get, key=f"{KEY}-hem")
        if_mode = st.selectbox("蛍光表示", ["rgb", "nuclear"], format_func={"rgb": "表示用RGB", "nuclear": "核推定 B−R（シアン）"}.get, key=f"{KEY}-ifm")
    roi = None
    with st.expander("ROI詳細（高解像度タイル読み込み）"):
        en = st.checkbox("ROIを高解像度で重ねる", key=f"{KEY}-roi-on")
        fe = S.modality(fixed_key).extent() or (1000.0, 1000.0)
        r1, r2, r3, r4 = st.columns(4)
        cx = r1.number_input("中心 x", value=float(fe[0] / 2), key=f"{KEY}-roi-x")
        cy = r2.number_input("中心 y", value=float(fe[1] / 2), key=f"{KEY}-roi-y")
        size = r3.number_input("一辺", value=200.0, min_value=20.0, max_value=1000.0, key=f"{KEY}-roi-s")
        roi_um = r4.selectbox("ROI解像度", [0.2, 0.4, 0.8], index=1, format_func=lambda v: f"{v} µm/px", key=f"{KEY}-roi-r")
        if en:
            roi = (cx, cy, size)
        st.caption("level 0の画像全体は読み込みません。ROIが触れる512 pxタイルだけを復号します。")
    banners = []
    derived = False
    if state.startswith("Before"):
        M, t = np.eye(2), np.zeros(2)
        label = "Before: 位置合わせなし（両frameのframe-local原点(0,0)を重ねただけ。原点はUNKNOWN）"
        banners.append({"level": "info", "text": "BEFORE — 変換なし。座標系はAfterと同じFixed frame"})
    else:
        if cand is None:
            if use.M is None:
                st.error("この結果には変換がありません（採択された変換なし）。候補を選ぶとプレビューできます。")
                return
            M, t, derived = _oriented(use.M, use.t, use, fixed_key)
            label = f"After: {use.config}"
        else:
            Mc, tc = (cand.M_final, cand.t_final) if (stage == "ICP後" and cand.M_final is not None) else (cand.M_search, cand.t_search)
            M, t, derived = _oriented(Mc, tc, use, fixed_key)
            label = f"After: 候補 #{cand.rank} ({'ICP後' if stage == 'ICP後' and cand.M_final is not None else '探索pose'})"
            banners.append({"level": "warn", "text": "候補プレビュー — 候補の選択は採択ではありません"})
        lab = use.safety_label()
        if use.accepted and cand is None:
            banners.append({"level": "ok", "text": f"{lab}（研究artifactで採択）"})
        else:
            banners.append({"level": "err" if lab in ("FAIL", "IMAGE_UNUSABLE") else "warn", "text": f"PREVIEW ONLY — {lab}。Workflow Dへ渡しません"})
        if derived:
            banners.append({"level": "info", "text": f"この結果は {use.moving}→{use.fixed} で計算。表示方向に合わせ逆変換を使用"})
    before = state.startswith("Before")
    base = {"key": "before" if before else f"{use.rid}|{cid}|{stage}", "label": label, "result_id": None if before else use.rid,
            "config": None if before else use.config, "candidate": None if (before or cand is None) else cand.cand_id,
            "stage": None if (before or cand is None) else stage, "safety": None if before else use.safety_label(), "M": M, "t": t}
    M, t, man = _manual_controls(S, fixed_key, M, t, base)
    if man is not None:
        label = f"MANUAL: {label} + 手動調整"
        banners.insert(0, {"level": "warn", "text": "MANUAL（未検証）— 手動調整した変換。PREVIEW ONLY、Workflow Dへ渡しません"})
    # candidate overlap regions (mapped moving FOV) in the display frame
    mm = S.modality(S.other(fixed_key))
    me = mm.extent()
    cpolys, sel_poly = [], None
    if me is not None and state.startswith("After"):
        for c in use.candidates:
            Mc, tc = (c.M_final, c.t_final) if c.M_final is not None else (c.M_search, c.t_search)
            Mo, to, _ = _oriented(Mc, tc, use, fixed_key)
            cpolys.append((f"候補 {c.cand_id} の領域" + (f"（{rui.results_source_label(c)}）" if c.source else ""), scene.apply(Mo, to, scene.footprint(me)),
                           rui.SOURCE_COLORS.get(c.source, "#b39ddb"), c.cand_id))
        sel_poly = scene.apply(M, t, scene.footprint(me))
    elif me is not None and man is not None:
        sel_poly = scene.apply(M, t, scene.footprint(me))
    if man is not None and me is not None and not before:
        cpolys.append(("自動推定（基準）の領域", scene.apply(base["M"], base["t"], scene.footprint(me)), "#ffffff", None, [8, 4]))
    if not before:
        rp = refui.reference_poly(refui.active_reference(S.sample_id, fixed_key), S, fixed_key)
        if rp is not None:
            cpolys.append(rp)
    opts_img = scene.ImageOptions(target_um=res, he_mode=he_mode, if_mode=if_mode, roi=roi, roi_um=roi_um)
    with st.spinner("画像と点群を準備しています..."):
        try:
            sc, diag, info = scene.build_scene(S, avail, fixed_key, M, t, state_label=label, banners=banners, opts=opts_img,
                                               candidate_polys=cpolys, selected_poly=sel_poly)
        except FileNotFoundError as exc:
            st.error(f"核座標を読み込めません: {exc}")
            return
    vkey = f"{S.sample_id}-{fixed_key}"
    sc["view_key"] = vkey                  # keep zoom/pan across reruns (candidate switch, manual adjustment)
    ev = viewer.interactive(sc, 720, key=f"{KEY}-viewer-{S.sample_id}-{fixed_key}", view_key=vkey)
    _handle_viewer_event(ev, S, fixed_key, M, t, use, cand)
    st.session_state[f"{KEY}-ctx"] = {"sample": S.sample_id, "result_id": None if before else use.rid, "run_id": None if before else rui._run_id(use),
                                      "candidate": None if (before or cand is None) else cand.cand_id, "stage": None if (before or cand is None) else stage,
                                      "fixed": fixed_key, "moving": S.other(fixed_key), "manual": man is not None,
                                      "M": np.asarray(M).tolist(), "t": np.asarray(t).tolist(), "label": label}
    _pending_annotations(S, fixed_key, M, t)
    _transform_annotation_buttons(S, fixed_key, M, t, use, cand, before, man)
    if man is not None:
        _manual_compare(S, fixed_key, base, M, t, diag)
        _manual_save(S, fixed_key, M, t, man, diag)
    m = (cand.metrics if (cand is not None and cand.metrics) else use.metrics) or {}
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("NN median（保存値）", _fmt(m.get("nn_median_um"), 3, "µm"))
    k2.metric("≤3 µm 近接率", _fmt(m.get("le_3um")))
    k3.metric("Null ≤3 µm（偶然一致）", _fmt(m.get("null_le_3um")))
    k4.metric("評価核数 n", _fmt(m.get("n_eval"), 4))
    k5.metric("Safety", "MANUAL (UNVERIFIED)" if man is not None else (use.safety_label() if cand is None else "PREVIEW ONLY"))
    if man is not None:
        st.caption("上の保存済み指標は**基準の変換**の値です。手動調整後の値は右の『UI再計算の診断』を見てください。")
    a, b = st.columns(2)
    with a:
        st.markdown("**保存済み研究指標**（再計算なし）")
        st.caption(m.get("population", "population: 記録なし") if cand is None else
                   "候補の指標（ablation 20261020のみ。20261019の非top候補は探索poseのみ保存）")
        _kv_table(m)
    with b:
        st.markdown("**UI再計算の診断**（表示中の変換、参考値・採択には使わない）")
        st.caption(diag.get("population", ""))
        _kv_table(diag)
        p = results.params(M, t)
        st.caption(f"表示中の変換 ({S.other(fixed_key)}→{fixed_key}): scale {p['scale']:.4f}, 回転 {p['rotation_deg']:.2f}°, "
                   f"reflect {p['reflect']}, t=({p['tx']:.1f}, {p['ty']:.1f})")


def _candidates_tab(rec):
    if not rec.candidates:
        st.info("この結果には候補リストが保存されていません。")
        return
    rows = []
    for c in rec.candidates:
        mm = c.metrics or {}
        rows.append({"Candidate ID": c.cand_id, "rank": c.rank, "θ (°)": f"{c.theta_deg:.1f}", "scale": f"{c.scale:.4f}", "chirality": c.chirality,
                     "translation": _fmt(c.t_um), "z score": _fmt(c.score_z), "NCC": _fmt(c.ncc), "gap to top": _fmt(c.gap_to_top, 2),
                     "≤3 µm (IF)": _fmt(mm.get("le_3um")), "null ≤3": _fmt(mm.get("null_le_3um")), "NN median": _fmt(mm.get("nn_median_um")),
                     "ICP shift (µm)": _fmt(mm.get("icp_median_shift_um")), "overlap (mm²)": _fmt((c.overlap or {}).get("overlap_area_mm2")),
                     "Safety": "PREVIEW ONLY", "note": c.note})
    st.table(pd.DataFrame(rows).set_index(pd.DataFrame(rows).columns[0]))
    st.caption(f"結果全体: verdict {rec.verdict}, margin {_fmt(rec.margin)}。候補は『Viewer』タブの『候補 / 変換』で表示を切り替えます。"
               "候補を選んでも採択にはなりません。")


def _compare_tab(S, recs):
    rows = []
    for r in recs:
        if r.method == "ERR":
            continue
        m = r.metrics or {}
        rows.append({"Method": results.METHOD_LABELS.get(r.method, r.method), "config": r.config, "dir": f"{r.moving}→{r.fixed}",
                     "Safety": r.safety_label(), "NN median (µm)": _fmt(m.get("nn_median_um")), "≤3 µm": _fmt(m.get("le_3um")),
                     "null ≤3": _fmt(m.get("null_le_3um")), "rotation HE→IF (°)": _fmt(m.get("rotation_he_to_if_deg")),
                     "scale HE→IF": _fmt(m.get("scale_he_to_if"), 4), "translation HE→IF (µm)": _fmt(m.get("translation_he_to_if_um")),
                     "tissue Dice": _fmt(m.get("img_tissue_dice")) + ("" if m.get("img_tissue_independent") in (True, "True", None) else " †"),
                     "nuclear corr": _fmt(m.get("img_nuclear_corr")) + ("" if m.get("img_nuclear_independent") in (True, "True", None) else " †"),
                     "runtime (s)": _fmt(m.get("runtime_s"), 3), "fallback": r.fallback_source or "",
                     "reasons": " | ".join(x for x in r.failure_reasons if x), "source": r.source})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("指標の母集団: IF frameで、写したH&E視野内のIF核（10 µm内側）。同じ定義の指標どうしだけを比べてください。"
               "変換が違えば評価に入る核も変わるため、件数（n_eval）は結果ごとに異なります。†: その画像特徴を探索に使ったので独立な根拠ではない。")
    # Fixed/Moving swap consistency
    by = {}
    for r in recs:
        if r.M is not None and r.method not in ("ERR",):
            base = r.config.replace("_HEfix", "").replace("_IFfix", "").replace("_HEmov", "").replace("_IFmov", "")
            by.setdefault((r.source, base), []).append(r)
    he, iff = S.modalities.get("HE"), S.modalities.get("IF")
    if iff is None or iff.extent() is None:
        return
    g = np.stack(np.meshgrid(np.linspace(0, iff.extent()[0], 21), np.linspace(0, iff.extent()[1], 21)), -1).reshape(-1, 2)
    sw = []
    for (src, base), rs in by.items():
        a = [r for r in rs if r.fixed == "HE"]; b = [r for r in rs if r.fixed == "IF"]
        if a and b:
            Ma, ta, _ = results.oriented(a[0].M, a[0].t, "HE", "IF")   # stored IF->HE, re-expressed as HE->IF
            Mb, tb = b[0].M, b[0].t                                     # HE->IF
            # map IF grid back to HE with b's inverse, then forward with a (HE->IF): residual in IF frame
            Mbi, tbi = np.linalg.inv(Mb), -np.linalg.inv(Mb) @ tb
            he_pts = g @ Mbi.T + tbi
            d = np.linalg.norm(he_pts @ Ma.T + ta - g, axis=1)
            sw.append({"source": src, "method": base, "HE Fixed": a[0].safety_label(), "IF Fixed": b[0].safety_label(),
                       "差 median (µm)": round(float(np.median(d)), 3), "差 max (µm)": round(float(d.max()), 3)})
    if sw:
        st.markdown("**Fixed/Moving交換の幾何学的一貫性**（IF frameの21×21格子で、2方向の写像の差）")
        st.table(pd.DataFrame(sw).set_index("method"))


def _warp_tab(S, avail, rec, effective, fixed_key):
    moving_key = S.other(fixed_key)
    fm, mm = S.modality(fixed_key), S.modality(moving_key)
    if not (avail[fixed_key]["image"] and avail[moving_key]["image"]):
        st.info("Warp Previewには両側の画像が必要です（現在: " + ", ".join(f"{k}: {v['image_reason'] or 'OK'}" for k, v in avail.items()) + "）")
        return
    use = effective if (effective is not None and effective is not rec and st.checkbox(f"fallback ({effective.config}) の変換を使う", value=True,
                                                                                       key=f"{KEY}-wfb")) else rec
    live = st.session_state.get(f"{KEY}-man-live")
    if live and live["sample"] == S.sample_id and live["fixed"] == fixed_key and \
            st.checkbox("Viewerで手動調整中の変換（未検証）を使う", value=True, key=f"{KEY}-wman"):
        M, t, derived, config, accepted = live["M"], live["t"], False, "MANUAL", False
        st.caption(live["label"])
    elif use.M is None:
        st.error("変換がありません。")
        return
    else:
        M, t, derived = results.oriented(use.M, use.t, use.fixed, fixed_key)
        config = use.config
        accepted = bool(use.accepted)
    c1, c2 = st.columns(2)
    dst_um = c1.selectbox("出力解像度", [4.0, 2.0, 1.0], index=1, format_func=lambda v: f"{v} µm/px", key=f"{KEY}-wres")
    order = c2.selectbox("補間", [1, 0], format_func={1: "bilinear", 0: "nearest"}.get, key=f"{KEY}-word")
    if accepted:
        st.success(f"Accepted transform（{use.config}）による研究用プレビューです。Production Workflow Dの出力ではありません。")
    else:
        st.warning(f"PREVIEW ONLY（{'MANUAL (UNVERIFIED)' if config == 'MANUAL' else use.safety_label()}）— 不確かな変換による画像です。正式出力として扱いません。")
    with st.spinner("逆写像でresamplingしています..."):
        lvl, um_k = scene.overview_level(mm, dst_um / 2)
        src = imaging.read_level_cached(mm.image_path, lvl)
        src_rgb = imaging.to_display_rgb(src, mm.kind, "rgb")
        out, valid, record = imaging.warp_preview(src_rgb, um_k, imaging.level_corner_um(mm.pixel_um), M, t,
                                                  imaging.level_corner_um(fm.pixel_um), fm.extent(), dst_um, order=order)
        flvl, fum = scene.overview_level(fm, dst_um)
        fixed_rgb = imaging.to_display_rgb(imaging.read_level_cached(fm.image_path, flvl), fm.kind, "rgb")
    record.update({"source_frame": f"{S.sample_id}:{moving_key}", "target_frame": f"{S.sample_id}:{fixed_key}", "transform": config,
                   "transform_direction_used": f"{moving_key}->{fixed_key}" + (" (inverse of the stored direction)" if derived else ""),
                   "accepted": accepted, "source_level": lvl, "valid_fraction": float(valid.mean()),
                   "forward_used_moving_to_target": {"M": np.round(M, 6).tolist(), "t": np.round(t, 3).tolist()}})
    # viewer cross-check: inverse-warped image placed at identity + forward-mapped nuclei
    c = imaging.level_corner_um(fm.pixel_um)
    imgs = [viewer.image_layer("fixed", f"{fm.label} (fixed)", imaging.encode_jpeg(fixed_rgb), fixed_rgb.shape[1], fixed_rgb.shape[0],
                               fum * np.eye(2), [c, c], "fixed"),
            viewer.image_layer("moving", f"{mm.label} warped (inverse map, {record['interpolation']}; 無効領域は透明)",
                               imaging.encode_png(np.dstack([out, (valid * 255).astype(np.uint8)])), out.shape[1], out.shape[0],
                               dst_um * np.eye(2), [c, c], "moving", mime="image/png")]
    fpts, _ = catalog.load_nuclei(fm); mpts, _ = catalog.load_nuclei(mm)
    sc = {"images": imgs, "points": [viewer.points_layer("fixed_nuclei", f"{fm.label} 核", fpts, scene.COLORS[fixed_key], visible=False),
                                     viewer.points_layer("moving_nuclei", f"{mm.label} 核（順写像）", scene.apply(M, t, mpts), scene.COLORS[moving_key])],
          "polys": [viewer.poly_layer("valid", "有効画像領域（source footprint）", scene.apply(M, t, scene.footprint(mm.extent())), "#ffd54f", dash=[6, 4])],
          "vectors": None, "bounds": [c, c, fm.extent()[0], fm.extent()[1]], "units": "µm", "frame_name": f"{S.sample_id}:{fixed_key} (target frame)",
          "frame_note": "画像は逆写像でtarget gridへresampling、核は順写像。両者が一致すれば座標の扱いが整合しています。範囲外は0（外挿なし）。",
          "banners": [{"level": "ok" if accepted else "warn", "text": "ACCEPTED (research)" if accepted else "PREVIEW ONLY"}], "default_mode": "checker"}
    components.html(viewer.render(sc, 640), height=660)
    rgba = np.dstack([out, (valid * 255).astype(np.uint8)])
    st.download_button("Warp preview PNG（alpha = 有効領域）", imaging.encode_png(rgba),
                       file_name=f"{S.sample_id}_{config}_{moving_key}_to_{fixed_key}{'' if accepted else '_PREVIEW_ONLY'}.png", mime="image/png",
                       key=f"{KEY}-wdl")
    st.json(record)


def _run_tab(S, manifest_file):
    rui.run_panel(S, manifest_file)
    with st.expander("旧Runner：核探索の構成要素の比較（2026-10-20 ablation X0–X3 / A2）"):
        _legacy_run_tab(S, manifest_file)
    _run_status()


def _legacy_run_tab(S, manifest_file):
    st.markdown("研究用Runnerで新しいRegistrationを1件だけSlurmに投入します（CPU 4コア、6 GB、最長45分）。Production処理とは別です。")
    c1, c2 = st.columns(2)
    variant = c1.selectbox("方式", list(jobs.RUNNABLE), format_func=jobs.RUNNABLE.get, key=f"{KEY}-run-v")
    fixed = c2.selectbox("Fixed", list(S.modalities), format_func=lambda k: DIRS.get(k, k), key=f"{KEY}-run-f")
    cfg = {"sample": S.sample_id, "method": "A2" if variant == "A2" else "E0x", "variant": variant, "fixed": fixed}
    errs = jobs.validate(cfg, S)
    if errs:
        st.error("入力検証で実行できません: " + " / ".join(errs))
    hit = jobs.cached_run(cfg)
    if hit:
        st.success(f"同じ設定の完了済み結果があります（run {hit['run_id']}）。再計算せず『Explorer run』で表示できます。")
    else:
        hit = jobs.active_same(cfg)
        if hit:
            st.info(f"同じ設定のジョブが実行待ち・実行中です（run {hit['run_id']}, job {hit.get('slurm_job_id')}）。重複投入はできません。")
    if st.session_state.pop(f"{KEY}-run-clear", False):   # one confirmation per submission
        st.session_state[f"{KEY}-run-ok"] = False
    if f"{KEY}-run-msg" in st.session_state:
        kind, msg = st.session_state.pop(f"{KEY}-run-msg")
        getattr(st, kind)(msg)
    confirm = st.checkbox("Slurm（partition mib-dbia）にジョブを1件投入することを確認しました", key=f"{KEY}-run-ok")
    if st.button("Slurmに投入", disabled=bool(errs) or not confirm or bool(hit), key=f"{KEY}-run-go"):
        try:
            rd, jid = jobs.submit(cfg, manifest_file)
            st.session_state[f"{KEY}-run-msg"] = ("success", f"投入しました: job {jid}, run {rd.name}")
        except (RuntimeError, OSError, ValueError) as exc:
            st.session_state[f"{KEY}-run-msg"] = ("error", f"投入できませんでした: {exc}")
        st.session_state[f"{KEY}-run-clear"] = True
        st.rerun()


@st.fragment(run_every="10s")
def _run_status():
    runs = [jobs.refresh(r) for r in jobs.list_runs()[:10]]
    if not runs:
        st.caption("Explorer runはまだありません。")
        return
    df = pd.DataFrame([{"run": r["run_id"], "state": r["state"], "job": r.get("slurm_job_id"), "sample": r["config"].get("sample"),
                        "method": jobs.WORKFLOWS.get(r["config"].get("variant"), r["config"].get("variant")) if r["config"].get("engine") else r["config"].get("variant"),
                        "fixed": r["config"].get("fixed"), "verdict": r.get("verdict"),
                        "accepted": r.get("accepted"), "updated": r.get("updated_at"), "error": r.get("error")} for r in runs])
    st.dataframe(df, hide_index=True, width="stretch")
    pick = st.selectbox("ログを見るrun", [r["run_id"] for r in runs], key=f"{KEY}-run-log")
    r = next(x for x in runs if x["run_id"] == pick)
    prog = jobs.read_log(r["run_dir"], "progress.log")
    if prog:
        st.markdown("**処理段階（C1 → C2 → C3 → C4）**")
        st.code(prog[-2000:], language="text")
    st.code(jobs.read_log(r["run_dir"], "stderr.log")[-2500:] or "(stderr empty)", language="text")
    if r["state"] in ("queued", "running") and st.button("このジョブをキャンセル", key=f"{KEY}-run-cancel"):
        try:
            st.info(jobs.cancel(Path(r["run_dir"])))
        except (PermissionError, ValueError, OSError) as exc:
            st.error(str(exc))


def _workflow_d_handoff(rec):
    """Accepted, fully automatic engine results only: Workflow C-compatible artifact for Workflow D (affine only)."""
    from ..registration_core import workflow_d_bridge as bridge
    art = rec.artifact or {}
    if not art.get("schema", "").startswith("cellreg-registration-artifact/"):
        st.caption("Workflow Dへの受け渡し：研究Workflow C（共通エンジン）の結果だけが対象です。")
        return
    why = bridge.check_handoff(art)
    if why:
        st.warning("Workflow Dへは渡せません：" + " / ".join(why) + "。未採択の候補・注釈を使った結果・手動変換は、PREVIEW ONLYのままです。")
        return
    st.success("採択済み（完全自動）の結果です。Workflow Dが読み込める形式（Workflow C result artifact、affineのみ・fine stageなし）で書き出せます。")
    if st.button("Workflow D用のartifactを作成", key=f"{KEY}-wd-build-{rec.rid}"):
        try:
            S = catalog.load_manifest()[art["sample"]]
            he, iff = catalog.side_input(S, "HE"), catalog.side_input(S, "IF")
            st.session_state[f"{KEY}-wd-bytes-{rec.rid}"] = bridge.to_workflow_c_artifact(art, he, iff)
        except (ValueError, OSError, KeyError) as exc:
            st.error(f"作成できませんでした: {exc}")
    data = st.session_state.get(f"{KEY}-wd-bytes-{rec.rid}")
    if data:
        st.download_button("workflow_c_registration_result.zip（研究Workflow C・採択済み）", data, file_name=f"{art['sample']}_{art['run_id']}_workflow_c_registration_result.zip",
                           mime="application/zip", key=f"{KEY}-wd-dl-{rec.rid}")
        st.caption("Workflow Dの『Workflow C result artifact』アップロード欄で読み込めます。Workflow Dの処理自体は変更していません。")


def _artifact_tab(rec):
    _workflow_d_handoff(rec)
    st.caption("この画面からWorkflow Dのsessionへ自動で変換を渡すことはありません（採択済みの結果だけを、ファイルとして書き出せます）。")
    for f in rec.files:
        st.code(f, language="text")
    if rec.artifact:
        st.json(rec.artifact, expanded=False)
    else:
        try:
            st.json(json.loads(Path(rec.files[0]).read_text()), expanded=False)
        except (OSError, IndexError, json.JSONDecodeError):
            st.caption("artifact JSONなし")


def _candidate_map_tab(S, avail, rec, fixed_key):
    """Where on the fixed section are the candidate regions?  Stage diagnosis + candidate table + clickable map."""
    rui.stage_panel(rec)
    refui.feature_maps_panel(S, avail)
    with st.container(border=True):
        cur_ref = refui.reference_panel(S, rec, fixed_key)
    if not rec.candidates:
        st.info("この結果には候補リストが保存されていません。")
        return
    df = rui.candidate_table(rec, S.sample_id)
    st.markdown("**候補一覧**（行を選ぶと『Viewer』でその候補を表示します。どの候補もPREVIEW ONLYで、選んでも採択にはなりません）")
    ev = st.dataframe(df, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key=f"{KEY}-ctable-{rec.rid}")
    rows = list(getattr(getattr(ev, "selection", None), "rows", []) or [])
    last = f"{KEY}-ctable-last-{rec.rid}"
    if rows and st.session_state.get(last) != rows[0]:
        st.session_state[last] = rows[0]
        st.session_state[f"{KEY}-cand-{rec.rid}"] = df.iloc[rows[0]]["候補"]
        st.info(f"候補 {df.iloc[rows[0]]['候補']} を選びました。『Viewer』タブで重ね合わせを確認できます。")
    # map: whole fixed section + every candidate's mapped moving FOV, coloured by search source (click = select)
    mm = S.modality(S.other(fixed_key))
    me = mm.extent()
    if me is None:
        return
    polys = []
    for c in rec.candidates:
        Mc, tc = (c.M_final, c.t_final) if c.M_final is not None else (c.M_search, c.t_search)
        Mo, to, _ = results.oriented(Mc, tc, rec.fixed, fixed_key)
        polys.append((f"{c.cand_id}（{rui.results_source_label(c)}）", scene.apply(Mo, to, scene.footprint(me)), rui.SOURCE_COLORS.get(c.source, "#b39ddb"), c.cand_id))
    rp = refui.reference_poly(cur_ref, S, fixed_key)
    if rp is not None:
        polys.append(rp)
    with st.spinner("Fixed画像を準備しています..."):
        sc, _, _ = scene.build_scene(S, avail, fixed_key, np.eye(2), np.zeros(2), state_label="候補マップ（Fixed全体）",
                                     banners=[{"level": "info", "text": "候補マップ：色 = 探索方法（橙 = 画像探索、水色 = 核探索、紫 = 従来法、緑 = 注釈を使った探索、黄緑の破線 = 参照変換）"}],
                                     opts=scene.ImageOptions(target_um=3.2), candidate_polys=polys, show_vectors=False)
    sc["images"] = [L for L in sc["images"] if L["role"] == "fixed"]
    sc["points"] = [L for L in sc["points"] if L["id"] == "fixed_nuclei"]
    sc["polys"] = [L for L in sc["polys"] if L["id"].startswith("cand_") or L["id"] == "fov_fixed"]
    fe = S.modality(fixed_key).extent()
    allp = np.concatenate([np.asarray(p[1]) for p in polys] + ([scene.footprint(fe)] if fe else []))
    lo, hi = allp.min(0), allp.max(0); pad = 0.03 * float(max(hi - lo))
    sc["bounds"] = [lo[0] - pad, lo[1] - pad, hi[0] + pad, hi[1] + pad]
    sc["view_key"] = f"map-{S.sample_id}-{fixed_key}"
    ev2 = viewer.interactive(sc, 560, key=f"{KEY}-map-{S.sample_id}-{fixed_key}", view_key=sc["view_key"])
    if ev2 and isinstance(ev2, dict) and ev2.get("kind") == "pick":
        seen = st.session_state.setdefault(f"{KEY}-seen-nonce", set())
        if ev2.get("nonce") not in seen:
            seen.add(ev2.get("nonce"))
            st.session_state[f"{KEY}-pick-pending"] = (rec.rid, ev2["candidate"])
            st.rerun()
    st.caption("左の『クリック操作』で『候補をクリックで選択』にすると、地図上の候補をクリックして選べます。")


def _review_tab(S, rec, fixed_key, session_id):
    ctx = st.session_state.get(f"{KEY}-ctx") or {}
    if ctx.get("sample") != S.sample_id:
        ctx = {}
    st.markdown("Viewerで表示中の候補・変換を評価します。評価は座標・変換と一緒に保存され、Review Annotation（人の判断の記録）として扱います。")
    rui.review_form(S.sample_id, rec, ctx.get("candidate"), ctx, fixed_key)
    st.divider()
    st.markdown("**このサンプルの評価**")
    rui.review_list(S.sample_id)
    st.markdown("**このサンプルの注釈**（『新規実行』の注釈を使った探索で再利用できます）")
    rui.annotation_list(S.sample_id)


def show_registration_explorer() -> None:
    st.header("Registration Explorer（研究モード）")
    st.caption("対応する組織領域の探索・候補の比較・位置合わせの検証・修正・保存を行う研究用の画面です。Productionの処理には適用されません。"
               "推定した核の対応は、生物学的に同じ細胞であることを示すものではありません。")
    samples, mtime = _manifest()
    c1, c2, c3, c4 = st.columns([1.4, 1.4, 1, 1])
    sid = c1.selectbox("Sample", list(samples), key=f"{KEY}-sample")
    S = samples[sid]
    hide = c3.toggle("画像を使わない（Point-only表示）", key=f"{KEY}-hide")
    sim = c4.toggle("画像読み込み失敗を再現（テスト）", key=f"{KEY}-sim")
    avail = catalog.availability(S, hide_images=hide, simulate_image_failure=sim)
    fixed_key = c2.selectbox("Fixed / Moving", [k for k in ("HE", "IF") if k in S.modalities], format_func=DIRS.get, key=f"{KEY}-fixed")
    session_id = rui.session_panel(sid)
    try:
        rui.sync_runs()
    except (OSError, ValueError, KeyError) as exc:
        st.warning(f"実行記録の同期に失敗しました: {exc}")
    recs = _records(sid, mtime) + _store_records(sid)
    pp = st.session_state.pop(f"{KEY}-pick-pending", None)       # candidate clicked in a viewer (applied before widgets exist)
    if pp:
        st.session_state[f"{KEY}-cand-{pp[0]}"] = pp[1]
    _metadata_table(S, avail, recs)

    avail_m = {m: _method_availability(m, avail, recs) for m in METHOD_ORDER}
    m1, m2 = st.columns([1.3, 2])
    default = next((i for i, m in enumerate(METHOD_ORDER) if m in ("WF", "E0x") and avail_m[m][0]), 0)
    method = m1.selectbox("結果の種類", METHOD_ORDER,
                          format_func=lambda m: results.METHOD_LABELS[m] + ("" if avail_m[m][0] else f"（利用不可: {avail_m[m][1]}）"),
                          index=default, key=f"{KEY}-method")
    ok, why = avail_m[method]
    fallback_note = None
    if not ok:
        if method in NEEDS_IMAGES and any(r.method == "E0x" for r in recs):
            fallback_note = f"{results.METHOD_LABELS[method]} は利用できません（{why}）。画像が使えないため、核探索の結果を表示します。"
            method = "E0x"
        else:
            st.warning(f"{results.METHOD_LABELS[method]} は利用できません: {why}。『新規実行』タブで実行できます。")
            t_run, t_rec = st.tabs(["新規実行", "実験記録"])
            with t_run:
                _run_tab(S, catalog.manifest_path())
            with t_rec:
                rui.record_tab(sid, session_id)
            return
    if fallback_note:
        st.error(fallback_note)
    cand = [r for r in recs if r.method == method]
    if method == "WF":     # newest run first, then the requested direction first (stable sorts)
        cand.sort(key=lambda r: (r.artifact or {}).get("created_at", ""), reverse=True)
        cand.sort(key=lambda r: r.fixed != fixed_key)
    else:
        cand.sort(key=lambda r: (r.fixed != fixed_key, r.config))
    rid = m2.selectbox("結果", [r.rid for r in cand], format_func=lambda x: (lambda r: f"{r.config}（{r.moving}→{r.fixed}"
                                                                         + ("" if r.fixed == fixed_key else "、逆方向で計算")
                                                                         + f"） — {r.safety_label()}")(results.find(cand, x)), key=f"{KEY}-rid-{sid}-{method}-{fixed_key}")
    rec = results.find(cand, rid)
    effective = None
    if rec.fallback_source:
        effective = next((r for r in recs if r.source == rec.source and r.config == rec.fallback_source), None)
    _safety_panel(rec, effective)
    st.caption(f"内部ID: {results.INTERNAL_IDS.get(rec.method, rec.method)} / {rec.rid}")
    tab_labels = ["対応領域の探索", "Viewer（重ね合わせ・手動調整）", "評価・注釈", "方式比較", "Warp Preview", "新規実行", "実験記録", "Artifact"]
    show_rcm = rcm_ui.enabled()
    tabs = st.tabs(tab_labels + ([rcm_ui.TAB_LABEL] if show_rcm else []))
    with tabs[0]:
        _candidate_map_tab(S, avail, rec, fixed_key)
    with tabs[1]:
        _viewer_tab(S, avail, rec, effective, fixed_key)
    with tabs[2]:
        _review_tab(S, rec, fixed_key, session_id)
    with tabs[3]:
        _compare_tab(S, recs)
    with tabs[4]:
        _warp_tab(S, avail, rec, effective, fixed_key)
    with tabs[5]:
        _run_tab(S, catalog.manifest_path())
    with tabs[6]:
        rui.record_tab(sid, session_id)
    with tabs[7]:
        _artifact_tab(rec)
    if show_rcm:
        with tabs[8]:
            rcm_ui.render(S.sample_id)
