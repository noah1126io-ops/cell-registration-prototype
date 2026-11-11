"""Diagnostics for the global localization (2026-10-27): the structural maps the image search compares, the
researcher-specified reference transform, and automatic candidates vs that reference.

A reference transform is provisional (MANUAL_UNVERIFIED) and never ground truth: it only diagnoses the search.
The active reference of a sample is the newest 'note' annotation with payload role = reference_transform.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from . import catalog, results, scene
from . import research_ui as rui
from ..registration_core import reference as ref

KEY = "regx"
REF_COLOR = "#76ff03"
MAP_INFO = {"tissue": "tissue（従来）：HE = 光学密度 / IF = 全チャンネル最大値。IFでは核（Hoechst）が支配的",
            "nuclear": "nuclear（従来）：HE = ヘマトキシリン / IF = B−R。核の配置（隣接切片では一致しない）",
            "occ": "occ（2026-10-27）：核を除いた染色の占有率。HE = 光学密度 / IF = max(G, R)。組織 ≈ 1、間隙・背景 ≈ 0",
            "cleft": "cleft（2026-10-27）：間隙の網目（1 − occ の帯域通過）"}


# ----------------------------------------------------------------------------- feature maps
@st.cache_data(show_spinner=False, max_entries=8)
def _feature_maps(sample_id: str, key: str, image_path: str, mtime: float):
    from ..registration_core.features import Modality, extract
    S = catalog.load_manifest()[sample_id]
    si = catalog.side_input(S, key)
    fm = extract(Modality(si.frame(), np.asarray(si.points, float)), res_um=2.0)
    out = {}
    for k in MAP_INFO:
        m = fm.maps[k].astype(np.float32)
        lo, hi = np.percentile(m, [1, 99.5])
        out[k] = (np.clip((m - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)[::2, ::2]
    return out, fm.quality


def feature_maps_panel(S: catalog.SampleInfo, avail: dict):
    with st.expander("画像探索が実際に比較している画像（特徴マップ）"):
        keys = [k for k in ("HE", "IF") if S.modality(k).image_on_disk() and S.modality(k).pixel_um is not None]
        if len(keys) < 2:
            st.info("両方の画像がないため、画像の特徴マップはありません（核点群で探索します）。")
            return
        st.caption("画像探索は生の色や輝度を比べず、両モダリティを同じ意味の地図に変換してから比べます。"
                   "下の地図が似ていない場合、画像探索は人が見ている構造を比べられていません。表示は2 µm/pxの地図を間引いたものです。")
        if not st.checkbox("特徴マップを計算して表示する（数秒〜十数秒）", key=f"{KEY}-fmaps-{S.sample_id}"):
            return
        maps = {}
        for k in keys:
            p = S.modality(k).image_path
            with st.spinner(f"{k} の特徴マップを計算しています..."):
                maps[k] = _feature_maps(S.sample_id, k, p, Path(p).stat().st_mtime)
        for name, info in MAP_INFO.items():
            st.markdown(f"**{info}**")
            cols = st.columns(2)
            for c, k in zip(cols, keys):
                c.image(maps[k][0][name], caption=f"{S.modality(k).label} — {name}", width="stretch")
        st.markdown("**画像の品質判定**（読める ≠ 使える ≠ 比較できる）")
        st.table(pd.DataFrame([{"modality": k, **{q: maps[k][1].get(q) for q in ("tissue_fraction_fov", "nuclear_map_auc_own_nuclei_vs_tissue", "usable")}}
                               for k in keys]).set_index("modality"))


# ----------------------------------------------------------------------------- reference transform
def _oriented_tf(row, fixed_key):
    M, t = np.asarray(json.loads(row["M_json"]), float), np.asarray(json.loads(row["t_json"]), float)
    return results.oriented(M, t, row["fixed"], fixed_key)[:2]


def active_reference(sample_id: str, fixed_key: str):
    """Newest reference designation for the sample -> dict(M, t in the fixed_key frame, row, landmarks) or None."""
    S = rui.store()
    notes = [a for a in S.annotations(sample_id, kind="note") if (json.loads(a["payload_json"] or "{}").get("role") == "reference_transform")]
    if not notes:
        return None
    pay = json.loads(sorted(notes, key=lambda a: a["created_at"])[-1]["payload_json"])
    tid = pay.get("transform_id")
    row = S.transform(tid) if tid else None
    if row is None:
        return None
    M, t = _oriented_tf(row, fixed_key)
    det = json.loads(row["detail_json"] or "{}")
    lms = det.get("landmarks")
    if lms and row["fixed"] != fixed_key:
        lms = [{"fixed": p["moving"], "moving": p["fixed"]} for p in lms]
    return {"M": M, "t": t, "row": row, "landmarks": lms, "detail": det}


def designate_reference(sample_id: str, tid: str | None, fixed: str, moving: str, label: str = ""):
    sid = rui.ensure_session(sample_id)
    return rui.store().add_annotation(sample_id=sample_id, session_id=sid, kind="note", frame=f"{sample_id}:{fixed}",
                                      payload={"role": "reference_transform", "transform_id": tid, "fixed": fixed, "moving": moving,
                                               "statement": "researcher-specified provisional reference; not ground truth"},
                                      label=label or ("参照変換の指定" if tid else "参照変換の解除"))


def reference_panel(S: catalog.SampleInfo, rec, fixed_key: str):
    """Choose the reference transform and compare every candidate of the shown result with it."""
    st.markdown("**参照変換（研究者が指定した暫定の変換。Ground Truthではありません）**")
    rows = [r for r in rui.store().transforms(S.sample_id) if r["kind"] in ("manual", "landmark")]
    cur = active_reference(S.sample_id, fixed_key)
    if not rows:
        st.caption("手動調整した変換・対応点から求めた変換がまだありません。Viewerで手動調整を保存するか、対応点を指定してください。")
        return cur
    opts = [r["id"] for r in rows]
    lab = lambda i: next(f"{r['label'] or r['id']}（{'対応点' if r['kind'] == 'landmark' else '手動'}、{r['created_at'][:16]}）" for r in rows if r["id"] == i)  # noqa: E731
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    pick = c1.selectbox("参照に使う変換", opts, index=opts.index(cur["row"]["id"]) if cur and cur["row"]["id"] in opts else 0, format_func=lab,
                        key=f"{KEY}-refpick-{S.sample_id}")
    if c2.button("参照変換に指定", key=f"{KEY}-refset-{S.sample_id}", disabled=bool(cur and cur["row"]["id"] == pick)):
        r = rui.store().transform(pick)
        aid = designate_reference(S.sample_id, pick, r["fixed"], r["moving"])
        st.success(f"参照変換に指定しました（{aid}）。自動推定の結果は変わりません。")
        st.rerun()
    if cur is None:
        st.caption("参照変換はまだ指定されていません。")
        return None
    p = ref.params(cur["M"])
    st.caption(f"現在の参照：{cur['row']['label'] or cur['row']['id']}（{cur['row']['id']}）— scale {p['scale']:.4f}, 回転 {p['rotation_deg']:.2f}°, "
               f"{S.other(fixed_key)}→{fixed_key}。地図とViewerに黄緑の破線で表示します。")
    fit = (cur["detail"] or {}).get("fit")
    if fit:
        st.caption(f"対応点 {fit['n']} 組（採用 {fit['n_inliers']}）：残差 中央値 {fit['residual_median_um']:.1f} µm、"
                   f"leave-one-out 中央値 {fit['loo_median_um'] if fit['loo_median_um'] is not None else '—'} µm"
                   + (f"、held-out {fit['holdout']['median_um']:.1f} µm（{fit['holdout']['n_test']}点）" if fit.get("holdout") else ""))
    if rec is None or not rec.candidates:
        return cur
    me = S.modality(S.other(fixed_key)).extent()
    if me is None:
        return cur
    out = []
    for c in rec.candidates:
        for stage, (Mc, tc) in (("探索pose（C1）", (c.M_search, c.t_search)), ("ICP後（C2）", (c.M_final, c.t_final))):
            if Mc is None:
                continue
            Mo, to, _ = results.oriented(Mc, tc, rec.fixed, fixed_key)
            d = ref.compare(Mo, to, cur["M"], cur["t"], me, cur["landmarks"])
            out.append({"候補": c.cand_id, "段階": stage, "探索スコア": None if c.score_z is None else round(c.score_z, 2),
                        "参照からの距離 µm（中央値）": round(d["probe_median_um"], 1), "回転の差 °": round(d["d_rotation_deg"], 1),
                        "倍率比": round(d["scale_ratio"], 3), "対応点誤差 µm": None if "landmark_error_median_um" not in d else round(d["landmark_error_median_um"], 1),
                        "参照領域か": "はい（≤ 40 µm）" if d["probe_median_um"] <= 40 else ("近い（≤ 150 µm）" if d["probe_median_um"] <= 150 else "いいえ")})
    st.markdown("**自動推定の候補と参照変換の比較**（Localization指標。核の近接率に依存しない）")
    st.dataframe(pd.DataFrame(out), hide_index=True, width="stretch")
    hit = [r for r in out if r["参照からの距離 µm（中央値）"] <= 40]
    st.caption(("参照領域に一致する候補：" + ", ".join(sorted({r['候補'] for r in hit}))) if hit else
               "参照領域に一致する候補はありません（C1が参照領域を候補に残していない）。")
    return cur


def reference_poly(cur, S, fixed_key):
    me = S.modality(S.other(fixed_key)).extent()
    if cur is None or me is None:
        return None
    return ("参照変換（研究者指定・暫定）の領域", scene.apply(cur["M"], cur["t"], scene.footprint(me)), REF_COLOR, None, [10, 4])


# ----------------------------------------------------------------------------- landmark metadata
def image_fingerprint(path) -> dict | None:
    if not path or not Path(path).is_file():
        return None
    p = Path(path); s = p.stat()
    with open(p, "rb") as f:
        head = hashlib.sha256(f.read(1 << 20)).hexdigest()
    return {"path": str(p), "size": s.st_size, "mtime": s.st_mtime, "sha256_first_1MiB": head}


def landmark_records(S: catalog.SampleInfo, fixed_key: str, lms: list, meta: pd.DataFrame | None):
    """Full per-landmark record: ids, type, confidence, pixel + frame coordinates of both images, image fingerprints."""
    moving_key = S.other(fixed_key)
    fm, mm = S.modality(fixed_key), S.modality(moving_key)
    out = []
    for i, p in enumerate(lms):
        m = meta.iloc[i] if meta is not None and i < len(meta) else {}
        rec = {"landmark_id": f"L{i + 1}", "type": (m.get("種類") if hasattr(m, "get") else None) or "other",
               "confidence": (m.get("確信度") if hasattr(m, "get") else None) or "medium",
               "fixed": [float(x) for x in p["fixed"]], "moving": [float(x) for x in p["moving"]],
               "frame_units": fm.units, "physical_origin": "frame-local (each image's own (0,0)); common physical origin UNKNOWN"}
        for side, mod, xy in (("fixed", fm, p["fixed"]), ("moving", mm, p["moving"])):
            rec[f"{side}_px"] = [float(v) / mod.pixel_um for v in xy] if mod.pixel_um else None
        out.append(rec)
    return out, {fixed_key: image_fingerprint(fm.image_path), moving_key: image_fingerprint(mm.image_path)}


def landmark_editor(pk: str, n: int):
    """Type / confidence for each clicked pair (kept in session state until saved)."""
    types = list(ref.LANDMARK_TYPES); confs = list(ref.CONFIDENCE)
    base = st.session_state.get(f"{pk}-lmmeta")
    if base is None or len(base) != n:
        old = base if base is not None else pd.DataFrame(columns=["種類", "確信度"])
        base = pd.DataFrame({"種類": [old["種類"].iloc[i] if i < len(old) else "other" for i in range(n)],
                             "確信度": [old["確信度"].iloc[i] if i < len(old) else "medium" for i in range(n)]}, index=[f"L{i + 1}" for i in range(n)])
    ed = st.data_editor(base, key=f"{pk}-lmmeta-ed-{n}", width="stretch",
                        column_config={"種類": st.column_config.SelectboxColumn("種類", options=types, help=" / ".join(f"{k} = {v}" for k, v in ref.LANDMARK_TYPES.items())),
                                       "確信度": st.column_config.SelectboxColumn("確信度", options=confs, help="high / medium / low")})
    st.session_state[f"{pk}-lmmeta"] = ed
    return ed
