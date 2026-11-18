"""Registration Explorer research tab 「細胞対応レビュー（研究）」 (2026-11-17).

Disabled unless CELLREG_CELL_REVIEW=1.  Flow: global transform (from the Viewer / candidates) -> ROI -> automatic
correspondence (frozen Graph/UOT/B2) -> human review (support / exclude / reassign / add / unset, undo / redo) ->
local deformation (Affine / auto + RBF / reviewed + RBF / UOT conservative) -> H&E image warp (inverse map) -> export.

Storage: files under runs/cell_review/<sample>/ (auto results written once; reviews as new version files).  The research
store (research.sqlite) is not written.  Nothing is sent to Workflow C/D.  Direction is fixed: H&E (moving) -> IF (fixed).
All logic lives in src/registration_research/cell_review.py; this module only draws and dispatches.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from ..registration_research import cell_review as CR
from . import catalog, imaging, results, scene, viewer
from . import cell_review_viewer as crv

ENV = "CELLREG_CELL_REVIEW"
TAB_LABEL = "細胞対応レビュー（研究）"
KEY = "regx-crv"
CAT_STYLE = {   # effective state -> (label, colour)
    "auto": ("自動推定 matched", "#43a047"), "supported": ("支持", "#00e676"), "reassigned": ("対応先を変更", "#e040fb"),
    "added": ("追加", "#e040fb"), "excluded": ("除外", "#9e9e9e"), "undecided": ("未確定", "#ffee58"),
    "ambiguous": ("ambiguous（自動）", "#ffa726"), "unmatched": ("unmatched（自動）", "#ef5350")}
MAX_IMAGE_PX = 2400 * 2400


def enabled() -> bool:
    return os.environ.get(ENV, "") == "1"


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return "はい" if v else "いいえ"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{nd}f}"
    return str(v)


def input_condition(avail) -> str:
    n = sum(bool(avail.get(k, {}).get("image")) for k in ("HE", "IF"))
    return {2: "両側の画像がある（画像の組織構造を優先した探索）", 1: "片側の画像だけがある（片側画像＋点群の探索）",
            0: "点群のみ（点群の探索）"}[n]


# ----------------------------------------------------------------------------- global transform
def transform_options(S, rec, fixed_key):
    """[(key, label, M_he_to_if, t_he_to_if)] from the Viewer's displayed transform, the result, and its candidates."""
    out = []
    ctx = st.session_state.get("regx-ctx") or {}
    if ctx.get("sample") == S.sample_id and ctx.get("M") is not None and not str(ctx.get("label", "")).startswith("Before"):
        M, t = CR.to_he_to_if(ctx["M"], ctx["t"], ctx["fixed"])
        out.append(("viewer", f"Viewerで表示中の変換: {ctx.get('label')}" + ("（手動調整を含む・未検証）" if ctx.get("manual") else ""), M, t))
    if rec is not None and rec.M is not None:
        M, t, _ = results.oriented(rec.M, rec.t, rec.fixed, "IF")
        out.append(("final", f"結果の最終変換: {rec.config}（{rec.safety_label()}）", M, t))
    for c in (rec.candidates if rec is not None else []):
        Mc, tc = (c.M_final, c.t_final) if c.M_final is not None else (c.M_search, c.t_search)
        M, t, _ = results.oriented(Mc, tc, rec.fixed, "IF")
        out.append((f"cand:{c.cand_id}", f"候補 {c.cand_id}（#{c.rank}, {'ICP後' if c.M_final is not None else '探索pose'}）[PREVIEW]", M, t))
    return out


def _he_reader(mod):
    info = imaging.pyramid_info(mod.image_path)
    return {"read": lambda lvl, x0, y0, x1, y1: imaging.read_region(mod.image_path, lvl, x0, y0, x1, y1),
            "level_shapes": info.level_shapes, "downsamples": info.downsamples, "pixel_um": mod.pixel_um}


# ----------------------------------------------------------------------------- session helpers
def _auto(path):
    cache = st.session_state.setdefault(f"{KEY}-autos", {})
    if str(path) not in cache:
        cache[str(path)] = CR.load_auto(path)
    return cache[str(path)]


def _review_key(auto):
    return f"{KEY}-review-{auto.auto_id}"


def _review_hash(rv):
    return hashlib.sha1(json.dumps(sorted(rv.pairs().items())).encode()).hexdigest()[:12] if rv is not None else "none"


def effective_categories(auto, rv):
    """{he_index: category} for HE nuclei in the ROI (category keys of CAT_STYLE)."""
    stt = rv.state() if rv is not None else {h: {"if_index": f, "decision": "auto"} for h, f in auto.auto_pairs().items()}
    out = {}
    for h, s in zip(auto.table["he_index"].to_numpy(int), auto.table["status"].to_numpy(str)):
        d = stt.get(int(h))
        if d is None or d["decision"] == "none":
            out[int(h)] = s if s != "matched" else "auto"
        else:
            out[int(h)] = d["decision"]
    return out


# ----------------------------------------------------------------------------- main
def render(S, avail, rec, fixed_key) -> None:
    st.markdown("**細胞対応レビュー（研究用プレビュー）** — 推定した核対応を確認・修正し、その修正を点群とH&E画像の変形に反映します。")
    st.caption("方向は H&E（Moving）→ 蛍光（Fixed）に固定しています（Workflow D の he_to_if と同じ）。推定・手動の対応は生物学的に同じ細胞である"
               "ことの証明ではありません。結果は runs/cell_review/ にファイルとして保存し、研究ストアには書き込みません。Workflow D へは送りません。")
    if "HE" not in S.modalities or "IF" not in S.modalities:
        st.info("このサンプルには H&E と IF の両方の核点群が必要です。")
        return
    he_m, if_m = S.modality("HE"), S.modality("IF")
    try:
        he_pts, he_ids = catalog.load_nuclei(he_m)
        if_pts, if_ids = catalog.load_nuclei(if_m)
    except FileNotFoundError as exc:
        st.error(f"核座標を読み込めません: {exc}")
        return
    st.info(f"入力条件: **{input_condition(avail)}**。対応領域（Global変換）の探索・候補の比較・手動の回転/倍率/平行移動・ランドマーク・ROIは、"
            "既存の『対応領域の探索』『Viewer』タブで行い、その変換をここで選びます。")
    # ---- Phase 1: global transform + ROI
    with st.container(border=True):
        st.markdown("**1. Global変換と計算範囲（ROI）**")
        opts = transform_options(S, rec, fixed_key)
        if not opts:
            st.warning("使える変換がありません。Viewerで変換を表示するか、結果を選んでください。")
            return
        okey = st.selectbox("使う変換（H&E→IF）", [o[0] for o in opts], format_func=lambda k: next(o[1] for o in opts if o[0] == k), key=f"{KEY}-tf-{S.sample_id}")
        _, _, M, t = next(o for o in opts if o[0] == okey)
        p = results.params(M, t)
        st.caption(f"H&E→IF: scale {p['scale']:.4f}, 回転 {p['rotation_deg']:.2f}°, reflect {p['reflect']}, t=({p['tx']:.1f}, {p['ty']:.1f}) µm")
        mapped = CR.map_points(M, t, he_pts)
        try:
            droi = CR.default_roi(mapped, if_pts, if_m.extent())
        except CR.ReviewError as exc:
            st.error(str(exc))
            return
        pend = (st.session_state.get(f"regx-pending-{S.sample_id}-{fixed_key}") or {}).get("roi")
        c = st.columns([1, 1, 1, 1, 1.4])
        src = c[4].radio("ROI", ["overlap", "viewer", "manual"], format_func={"overlap": "重なり全体", "viewer": "Viewerで描いたROI",
                                                                                 "manual": "数値で指定"}.get, key=f"{KEY}-roisrc")
        if src == "viewer" and pend:
            poly = np.asarray(pend, float)
            if fixed_key == "HE":                          # the Viewer ROI is in the displayed fixed frame
                poly = CR.map_points(M, t, poly)
            roi0 = [float(poly[:, 0].min()), float(poly[:, 1].min()), float(poly[:, 0].max()), float(poly[:, 1].max())]
        else:
            if src == "viewer":
                st.caption("Viewerの『クリック操作 → 領域（ROI）を指定』で描いて送信すると使えます。今は重なり全体を使います。")
            roi0 = droi
        roi = [c[i].number_input(["x0", "y0", "x1", "y1"][i] + " (IF µm)", value=float(roi0[i]), step=10.0, disabled=src != "manual",
                                 key=f"{KEY}-roi{i}-{src}-{okey}") for i in range(4)]
        n_he = int(CR._inside(mapped, roi).sum())
        st.caption(f"ROI内: H&E核 {n_he} 個（変換後）、IF核 {int(CR._inside(if_pts, roi).sum())} 個。大きなROIは計算に数分かかります（Graph/UOT/B2）。")
        b1, b2 = st.columns([1, 2])
        if b1.button("自動推定を計算して保存", key=f"{KEY}-run", type="primary"):
            with st.spinner("凍結済み Graph/UOT/B2 で核対応を推定しています..."):
                try:
                    auto = CR.compute_auto(he_pts, he_ids, if_pts, if_ids, M, t, roi=roi, if_extent=if_m.extent(),
                                           meta={"sample_id": S.sample_id, "global_source": next(o[1] for o in opts if o[0] == okey),
                                                 "result_id": rec.rid if rec is not None else None})
                    d = CR.save_auto(auto, S.sample_id)
                    st.session_state[f"{KEY}-cur-{S.sample_id}"] = str(d)
                    st.success(f"保存しました: {d}（この自動推定は今後変更されません）")
                except CR.ReviewError as exc:
                    st.error(str(exc))
        autos = CR.list_autos(S.sample_id)
        if not autos:
            st.caption("まだ自動推定がありません。")
            return
        cur = st.session_state.get(f"{KEY}-cur-{S.sample_id}")
        names = [str(p) for p in autos]
        idx = names.index(cur) if cur in names else 0

        def _lab(pth):
            m = json.loads((CR.Path(pth) / "auto_meta.json").read_text())
            return f"{m['auto_id']}  {m['created_at'][:16]}  HE {m['n_he_roi']}  {m['status_counts']}  — {m.get('global_source', '')[:50]}"
        sel = b2.selectbox("保存済みの自動推定", names, index=idx, format_func=_lab, key=f"{KEY}-autosel-{S.sample_id}")
        st.session_state[f"{KEY}-cur-{S.sample_id}"] = sel
    try:
        auto = _auto(sel)
    except CR.ReviewError as exc:
        st.error(str(exc))
        return
    Ma, ta = np.asarray(auto.meta["global"]["M"]), np.asarray(auto.meta["global"]["t"])
    if not (np.allclose(Ma, M) and np.allclose(ta, t)):
        st.warning("選択中の自動推定は、上で選んだ変換とは別の変換で計算されています。以下は自動推定に保存された変換を使います。")
    mapped = CR.map_points(Ma, ta, he_pts)
    _review_section(S, auto, he_pts, he_ids, if_pts, if_ids, mapped, avail)


def _review_section(S, auto, he_pts, he_ids, if_pts, if_ids, mapped, avail):
    rk = _review_key(auto)
    with st.container(border=True):
        st.markdown("**2. 核対応の推定と手動レビュー**")
        meta = auto.meta
        st.caption(f"{meta['auto_id']}: ROI {[round(v, 1) for v in meta['roi_if_frame']]}、HE {meta['n_he_roi']} 個、候補のIF {meta['n_if_candidates']} 個。"
                   f" 状態: {meta['status_counts']}。自動推定で同じIF核に複数のH&E核が対応: {meta['conflicting_auto_pairs']} 件。"
                   f" UOT収束: {meta['uot_diagnostics']['converged']}。")
        with st.expander("スコアの定義"):
            st.table(pd.DataFrame([{"指標": k, "定義": v} for k, v in meta["score_definitions"].items()]).set_index("指標"))
        c1, c2, c3 = st.columns([2, 1, 1])
        revs = CR.list_reviews(S.sample_id, auto.auto_id)
        choice = c1.selectbox("レビュー", ["(新規)"] + [str(p) for p in revs],
                              format_func=lambda s: "新しいレビューを始める" if s == "(新規)" else s.split("/")[-2] + " / " + s.split("/")[-1],
                              key=f"{KEY}-revsel-{auto.auto_id}")
        if c2.button("このレビューを開く", key=f"{KEY}-revopen"):
            st.session_state[rk] = CR.Review.start(auto) if choice == "(新規)" else CR.load_review(choice)
        user = c3.text_input("使用者", value=getpass.getuser(), key=f"{KEY}-user")
        rv = st.session_state.get(rk)
        if rv is None:
            rv = st.session_state[rk] = CR.Review.start(auto)
        st.caption(f"レビュー {rv.review_id}（保存版 v{rv.version}）: 有効な変更 {rv.cursor} 件 / 履歴 {len(rv.history)} 件、対応ペア {len(rv.pairs())} 組、"
                   f"衝突 {len(rv.conflicts())} 件。自動推定のファイルは変更されません。")
        cats = effective_categories(auto, rv)
        sel_he = st.session_state.get(f"{KEY}-selhe-{auto.auto_id}")
        sel_if = st.session_state.get(f"{KEY}-selif-{auto.auto_id}")
        ev = _picker(S, auto, rv, cats, mapped, if_pts, avail, sel_he, sel_if)
        if ev and isinstance(ev, dict) and ev.get("kind") == "nucleus":
            seen = st.session_state.setdefault(f"{KEY}-nonce", set())
            if ev.get("nonce") not in seen:
                seen.add(ev.get("nonce"))
                if ev["role"] == "he":
                    st.session_state[f"{KEY}-selhe-{auto.auto_id}"] = int(ev["index"])
                    st.session_state.pop(f"{KEY}-selif-{auto.auto_id}", None)
                else:
                    st.session_state[f"{KEY}-selif-{auto.auto_id}"] = int(ev["index"])
                st.rerun()
        _nucleus_panel(auto, rv, he_ids, if_ids, mapped, if_pts, user)
        u1, u2, u3, u4 = st.columns(4)
        if u1.button("↶ 取り消す", disabled=rv.cursor == 0, key=f"{KEY}-undo"):
            rv.undo(user); st.rerun()
        if u2.button("↷ やり直す", disabled=rv.cursor >= len(rv.ops), key=f"{KEY}-redo"):
            rv.redo(user); st.rerun()
        if u3.button("レビューを保存（新しい版）", key=f"{KEY}-save"):
            p = rv.save(S.sample_id)
            st.success(f"保存しました: {p}（以前の版は残ります）")
        u4.download_button("レビューJSON", json.dumps(rv.to_json(), ensure_ascii=False, indent=1, default=CR._json_default),
                           file_name=f"{S.sample_id}_{rv.review_id}.json", mime="application/json", key=f"{KEY}-revdl")
        conf = rv.conflicts()
        if conf:
            st.warning("同じIF核に複数のH&E核が割り当てられています（衝突）: " + "; ".join(
                f"IF {CR._scalar(if_ids[f])} ← HE " + ", ".join(str(CR._scalar(he_ids[h])) for h in hs) for f, hs in list(conf.items())[:12])
                + (" …" if len(conf) > 12 else ""))
        with st.expander("核の一覧と変更履歴"):
            tab = auto.table.copy()
            tab["現在の状態"] = tab["he_index"].map(lambda h: CAT_STYLE[cats[int(h)]][0])
            pr = rv.pairs()
            tab["現在の対応先 IF id"] = tab["he_index"].map(lambda h: CR._scalar(if_ids[pr[int(h)]]) if int(h) in pr else None)
            f = st.multiselect("状態で絞り込み", [v[0] for v in CAT_STYLE.values()], key=f"{KEY}-filt")
            if f:
                tab = tab[tab["現在の状態"].isin(f)]
            st.dataframe(tab[["he_id", "status", "b2_match_prob", "b2_unmatched_prob", "top1_if_id", "top1_distance_um", "if_conflict_n",
                              "現在の状態", "現在の対応先 IF id", "n_he_within_30um", "n_if_within_30um"]], height=260, hide_index=True)
            if rv.history:
                st.dataframe(pd.DataFrame([{"#": h["seq"], "操作": h.get("action"), "内容": CR.OP_LABELS.get(h.get("op"), h.get("op", "")),
                                            "HE id": CR._scalar(he_ids[h["he_index"]]) if "he_index" in h else "",
                                            "IF id": CR._scalar(if_ids[h["if_index"]]) if h.get("if_index") is not None else "",
                                            "元の推定 IF id": CR._scalar(if_ids[h["auto"]["if_index"]]) if (h.get("auto") or {}).get("if_index") is not None else "",
                                            "使用者": h["user"], "日時": h["at"], "理由": h.get("reason", "")} for h in rv.history]), hide_index=True)
    _deformation_section(S, auto, rv, he_pts, he_ids, if_pts, if_ids, mapped, avail)


# Display modes shared with the Viewer tab: the Viewer's widget state (regx-hem / regx-ifm) is the single source of truth.
# Two widgets cannot share one key, so this tab has its own widgets that copy from / write back to the Viewer's keys.
HE_MODE_LABELS = {"rgb": "RGB", "nuclear": "核強調（反転グレー）"}          # same labels as the Viewer tab (page.py)
IF_MODE_LABELS = {"rgb": "表示用RGB", "nuclear": "核推定 B−R（シアン）"}
SHARED_MODE_KEYS = {"he": "regx-hem", "if": "regx-ifm"}


def _shared_mode(which, label, labels, col):
    shared, own = SHARED_MODE_KEYS[which], f"{KEY}-{which}m"
    st.session_state[own] = st.session_state.get(shared, st.session_state.get(own, "rgb"))   # default RGB, as in the Viewer

    def _write_back():
        st.session_state[shared] = st.session_state[own]
    return col.selectbox(label, list(labels), format_func=labels.get, key=own, on_change=_write_back)


def display_modes(col_he=st, col_if=st):
    """(he_mode, if_mode) shared with the Viewer tab; default RGB."""
    return (_shared_mode("he", "H&E表示", HE_MODE_LABELS, col_he), _shared_mode("if", "蛍光表示", IF_MODE_LABELS, col_if))


def _picker(S, auto, rv, cats, mapped, if_pts, avail, sel_he, sel_if):
    Ma, ta = np.asarray(auto.meta["global"]["M"]), np.asarray(auto.meta["global"]["t"])
    c_img, c_he, c_if = st.columns([1.2, 1, 1])
    show_img = c_img.toggle("画像を重ねる（概観）", value=any(avail[k]["image"] for k in ("HE", "IF")), key=f"{KEY}-img")
    he_mode, if_mode = display_modes(c_he, c_if)
    av = {k: dict(v) for k, v in avail.items()}
    if not show_img:
        for k in av:
            av[k]["image"], av[k]["image_reason"] = False, "hidden in review"
    try:
        sc, _, _ = scene.build_scene(S, av, "IF", Ma, ta, state_label="細胞対応レビュー", banners=[
            {"level": "warn", "text": "研究用プレビュー — 推定・手動の対応は同一細胞の証明ではありません"}],
            opts=scene.ImageOptions(target_um=1.6, he_mode=he_mode, if_mode=if_mode), show_vectors=False)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return None
    roi = auto.meta["roi_if_frame"]
    in_roi = auto.he_idx
    pts = [crv.pick_points_layer("if", "IF核（クリックで対応先に）", if_pts, np.arange(len(if_pts)), "if", "#90a4ae", radius=2.2)]
    for cat, (lab, col) in CAT_STYLE.items():
        hs = np.asarray([h for h in in_roi if cats[int(h)] == cat], int)
        if len(hs):
            pts.append(crv.pick_points_layer(f"he_{cat}", f"H&E核: {lab}", mapped[hs], hs, "he", col, radius=2.8))
    out_roi = np.setdiff1d(np.arange(len(mapped)), in_roi)
    if len(out_roi):
        pts.append(viewer.points_layer("he_out", "H&E核（ROI外）", mapped[out_roi], "#5d4037", visible=False, radius=1.5))
    polys = [p for p in sc["polys"] if p["id"] in ("fov_fixed", "fov_moving")]
    polys.append(viewer.poly_layer("roi", "計算範囲（ROI）", np.array([[roi[0], roi[1]], [roi[2], roi[1]], [roi[2], roi[3]], [roi[0], roi[3]]]),
                                   "#76ff03", dash=[6, 4]))
    st_ = rv.state()
    auto_pairs = auto.auto_pairs()
    groups = {"auto": [], "edited": []}
    for h, f in rv.pairs().items():
        groups["edited" if st_[h]["decision"] in ("reassigned", "added") else "auto"].append((h, f))
    for g, col, lab in (("auto", "#43a047", "対応線（自動・支持）"), ("edited", "#e040fb", "対応線（手動で変更・追加）")):
        if groups[g]:
            hs, fs = zip(*groups[g])
            polys.append(crv.segments_layer(f"pairs_{g}", lab, mapped[list(hs)], if_pts[list(fs)], col))
    conf = rv.conflicts()
    if conf:
        hs = [h for v in conf.values() for h in v]
        fs = [f for f, v in conf.items() for _ in v]
        polys.append(crv.segments_layer("pairs_conf", "衝突（同じIF核に複数）", mapped[hs], if_pts[fs], "#ff1744", width=2.4))
    excl = [h for h, s in st_.items() if s["decision"] == "excluded" and h in auto_pairs]
    if excl:
        polys.append(crv.segments_layer("pairs_excl", "除外した自動対応", mapped[excl], if_pts[[auto_pairs[h] for h in excl]], "#9e9e9e",
                                        width=0.8, visible=False))
    selected = {}
    if sel_he is not None:
        selected["he"] = mapped[sel_he].round(2).tolist()
        cands = auto.candidates[auto.candidates["he_index"] == sel_he]
        selected["candidates"] = cands[["if_x", "if_y"]].to_numpy(float).round(2).tolist()
    if sel_if is not None:
        selected["if"] = if_pts[sel_if].round(2).tolist()
    pad = 40.0
    sc.update({"points": pts, "polys": polys, "vectors": None, "selected": selected,
               "bounds": [roi[0] - pad, roi[1] - pad, roi[2] + pad, roi[3] + pad], "frame_name": f"{S.sample_id}:IF (fixed frame)",
               "frame_note": "IF（Fixed）frame の µm。H&E核はGlobal変換後の位置。黄色の輪 = 選択中のH&E核、白い輪 = その対応候補、水色の輪 = 選択中のIF核。"})
    return crv.interactive(sc, 640, key=f"{KEY}-viewer-{auto.auto_id}", view_key=f"crv-{auto.auto_id}")


def _nucleus_panel(auto, rv, he_ids, if_ids, mapped, if_pts, user):
    sel_he = st.session_state.get(f"{KEY}-selhe-{auto.auto_id}")
    sel_if = st.session_state.get(f"{KEY}-selif-{auto.auto_id}")
    c1, c2 = st.columns([1, 1])
    typed = c1.text_input("H&E核IDで選択", key=f"{KEY}-typehe", placeholder="例: 1234")
    if typed:
        hit = np.flatnonzero(np.asarray([str(CR._scalar(v)) for v in he_ids]) == typed.strip())
        if len(hit) and int(hit[0]) in set(auto.he_idx.tolist()):
            sel_he = int(hit[0]); st.session_state[f"{KEY}-selhe-{auto.auto_id}"] = sel_he
        else:
            c1.error("このIDのH&E核はROI内にありません（不正なIDまたはROI外）")
    typed_if = c2.text_input("対応先のIF核IDを指定", key=f"{KEY}-typeif", placeholder="例: 5678")
    if typed_if:
        hit = np.flatnonzero(np.asarray([str(CR._scalar(v)) for v in if_ids]) == typed_if.strip())
        if len(hit):
            sel_if = int(hit[0]); st.session_state[f"{KEY}-selif-{auto.auto_id}"] = sel_if
        else:
            c2.error("このIDのIF核はありません（不正なID）")
    if sel_he is None:
        st.caption("ビューアーで『H&E核を選択』にしてH&E核をクリックすると、対応候補と操作ボタンが表示されます。")
        return
    row = auto.table[auto.table["he_index"] == sel_he]
    if row.empty:
        st.warning("選択したH&E核はこの自動推定のROIにありません。")
        return
    r = row.iloc[0]
    cur = rv.state().get(sel_he, {"if_index": None, "decision": "none"})
    with st.container(border=True):
        st.markdown(f"**H&E核 {CR._scalar(he_ids[sel_he])}**（index {sel_he}）— 自動推定: **{r['status']}**、現在: "
                    f"**{CAT_STYLE.get(effective_categories(auto, rv)[sel_he], ('?',))[0]}**"
                    + (f"、現在の対応先 IF {CR._scalar(if_ids[cur['if_index']])}" if cur["if_index"] is not None else ""))
        st.caption(f"H&E座標 ({r['he_x']:.1f}, {r['he_y']:.1f}) µm → 変換後 ({r['mapped_x']:.1f}, {r['mapped_y']:.1f}) µm（IF frame）。"
                   f" 周辺30 µm内: H&E {r['n_he_within_30um']} 個、IF {r['n_if_within_30um']} 個。根拠: {r['adoption_reason']}")
        cands = auto.candidates[auto.candidates["he_index"] == sel_he].copy()
        cands["対応先(現在)"] = cands["if_index"] == cur["if_index"]
        st.dataframe(cands[["rank", "if_id", "distance_um", "mass_share", "graph_support", "position_graph_cost", "対応先(現在)"]], hide_index=True)
        reason = st.text_input("理由（履歴に保存）", key=f"{KEY}-reason")
        b = st.columns(5)

        def do(op, f=None):
            try:
                rv.apply(op, sel_he, f, user=user, reason=reason)
                st.rerun()
            except CR.ReviewError as exc:
                st.error(str(exc))
        if b[0].button("支持する", disabled=cur["if_index"] is None, key=f"{KEY}-sup"):
            do("support")
        if b[1].button("除外する", disabled=cur["if_index"] is None, key=f"{KEY}-exc"):
            do("exclude")
        if b[2].button("未確定にする", key=f"{KEY}-uns"):
            do("unset")
        pick = b[3].selectbox("候補に変更", [None] + cands["if_index"].tolist(),
                              format_func=lambda f: "候補を選ぶ" if f is None else f"#{int(cands.loc[cands.if_index == f, 'rank'].iloc[0])} IF {CR._scalar(if_ids[f])}",
                              key=f"{KEY}-candpick-{sel_he}")
        if pick is not None and b[3].button("この候補にする", key=f"{KEY}-candset"):
            do("reassign" if cur["if_index"] is not None else "add", int(pick))
        if sel_if is not None:
            d = float(np.linalg.norm(if_pts[sel_if] - mapped[sel_he]))
            if b[4].button(f"IF {CR._scalar(if_ids[sel_if])} を対応先にする（{d:.1f} µm）", key=f"{KEY}-ifset"):
                do("reassign" if cur["if_index"] is not None else "add", int(sel_if))
        else:
            b[4].caption("ビューアーで『対応先のIF核を選択』にしてIF核をクリックすると、任意のIF核を対応先にできます。")


# ----------------------------------------------------------------------------- deformation + image
def _fields(auto, rv, he_pts, if_pts, excl):
    cache = st.session_state.setdefault(f"{KEY}-fields", {})
    kb = (auto.auto_id, "base", excl)
    if kb not in cache:
        cache[kb] = CR.compute_fields(auto, None, he_pts, if_pts, methods=("A", "AUTO", "D"), exclude_conflicts=excl)
    kr = (auto.auto_id, rv.review_id, _review_hash(rv), excl)
    if kr not in cache:
        cache[kr] = CR.compute_fields(auto, rv, he_pts, if_pts, methods=("REVIEWED",), exclude_conflicts=excl)
    return {**cache[kb], **cache[kr]}


def _deformation_section(S, auto, rv, he_pts, he_ids, if_pts, if_ids, mapped, avail):
    with st.container(border=True):
        st.markdown("**3. 点群の局所変形（Global変換は固定）**")
        st.caption("アンカーが20組未満、またはアンカーが占有セル（100 µm）の20%未満しか覆わない場合は、変形せずAffineのみにします。"
                   "すべての変位場は既存の安全性検査（Jacobian・Foldover・過大変位・点の改善）を通ったときだけ適用されます（閾値は変更していません）。")
        excl = st.checkbox("衝突している対応をアンカーから除く（自動・手動の両方に適用）", key=f"{KEY}-excl")
        state_key = f"{KEY}-fieldsfor-{auto.auto_id}"
        cur_hash = (rv.review_id, _review_hash(rv), excl)
        go = st.button("変位場を計算 / 再計算（現在のレビューを反映）", key=f"{KEY}-fgo", type="primary")
        if go:
            with st.spinner("変位場を計算しています（UOT conservative を含む初回は時間がかかります）..."):
                _fields(auto, rv, he_pts, if_pts, excl)
            st.session_state[state_key] = cur_hash
        used = st.session_state.get(state_key)
        if used is None:
            st.caption("ボタンを押すと計算します。")
            return
        if used != cur_hash:
            st.warning("レビューが変更されています。上のボタンで再計算すると、手動修正後の変位場に反映されます（下は前回の計算結果です）。")
            rid, h, ex = used
            fields = {**st.session_state[f"{KEY}-fields"][(auto.auto_id, "base", ex)], **st.session_state[f"{KEY}-fields"][(auto.auto_id, rid, h, ex)]}
        else:
            fields = _fields(auto, rv, he_pts, if_pts, excl)
        rows = []
        for m in CR.FIELD_METHODS:
            s = fields[m].summary()
            rows.append({"方式": s["label"], "適用": "適用" if s["applied"] else "不適用（Affineのみ）", "理由": s["reason"] or "",
                         "アンカー数": _fmt(s.get("anchors_n")), "アンカー被覆": _fmt(s.get("anchor_coverage_cells"), 2),
                         "Jacobian min": _fmt(s.get("jacobian_min")), "Jacobian max": _fmt(s.get("jacobian_max")),
                         "Foldover": _fmt(s.get("foldover_fraction"), 4), "最大変位 (µm)": _fmt(s.get("displacement_max_um"), 1),
                         "p95変位 (µm)": _fmt(s.get("displacement_p95_um"), 1)})
        st.table(pd.DataFrame(rows).set_index("方式"))
        st.caption("Jacobian・変位は、適用されなかった場合も『試した変位場』の値です。")
        diff = CR.displacement_change(fields["AUTO"], fields["REVIEWED"], mapped[auto.he_idx])
        st.caption(f"手動修正による点の位置の変化（自動＋RBF → 手動修正後＋RBF、ROI内のH&E核 {diff['n']} 個）: 中央値 {diff['median_um']:.2f} µm、"
                   f"95% {diff['p95_um']:.2f} µm、最大 {diff['max_um']:.2f} µm、1 µm以上動いた核 {diff['moved_gt_1um']} 個。")
        m = st.radio("表示する方式", list(CR.FIELD_METHODS), format_func=CR.FIELD_LABELS.get, horizontal=True, key=f"{KEY}-fm")
        fr = fields[m]
        after = fr.apply_points(mapped)
        sel = auto.he_idx
        roi = auto.meta["roi_if_frame"]
        a_, b_ = mapped[sel], after[sel]
        dd = np.linalg.norm(b_ - a_, axis=1)
        scn = {"images": [], "points": [viewer.points_layer("if", "IF核", if_pts, "#90a4ae", radius=2.0),
                                        viewer.points_layer("he_aff", "H&E核（Affine後）", a_, "#ff4fa3", visible=False, radius=2.0),
                                        viewer.points_layer("he_def", f"H&E核（{CR.FIELD_LABELS[m]}）", b_, "#ffeb3b", radius=2.4)],
               "polys": [viewer.poly_layer("roi", "ROI", np.array([[roi[0], roi[1]], [roi[2], roi[1]], [roi[2], roi[3]], [roi[0], roi[3]]]), "#76ff03", dash=[6, 4])],
               "vectors": viewer.vectors_layer("vec", "変位ベクトル（Affine後 → 変形後）", a_, b_, dd, visible=True),
               "bounds": [roi[0] - 40, roi[1] - 40, roi[2] + 40, roi[3] + 40], "units": "µm", "frame_name": f"{S.sample_id}:IF (fixed frame)",
               "frame_note": "変位ベクトルの色は大きさ（≤3 / ≤5 / >5 µm）。ここでの『残留』は変形量の意味です。",
               "banners": [{"level": "ok" if fr.applied else "warn", "text": ("適用（研究用プレビュー）" if fr.applied else f"不適用: {fr.reason}")}],
               "default_mode": "overlay"}
        components.html(viewer.render(scn, 560), height=580)
    _image_section(S, auto, rv, fields, he_pts, he_ids, if_pts, if_ids, avail)


def _image_section(S, auto, rv, fields, he_pts, he_ids, if_pts, if_ids, avail):
    with st.container(border=True):
        st.markdown("**4. H&E画像への変形の適用（逆写像）と出力**")
        he_m, if_m = S.modality("HE"), S.modality("IF")
        Ma, ta = np.asarray(auto.meta["global"]["M"]), np.asarray(auto.meta["global"]["t"])
        roi = list(auto.meta["roi_if_frame"])
        images, recs = {}, {}
        if not avail["HE"]["image"]:
            st.info(f"H&E画像が使えません（{avail['HE']['image_reason']}）。変換後の点群と変位場だけを出力します。")
        else:
            c1, c2 = st.columns(2)
            dst = c1.selectbox("出力解像度", [2.0, 1.0, 0.5], format_func=lambda v: f"{v} µm/px", key=f"{KEY}-dst")
            m = c2.selectbox("非剛体の方式", ["REVIEWED", "AUTO", "D"], format_func=CR.FIELD_LABELS.get, key=f"{KEY}-imgm")
            w, h = (roi[2] - roi[0]) / dst, (roi[3] - roi[1]) / dst
            if w * h > MAX_IMAGE_PX:
                st.warning(f"ROIが大きいため（{int(w)}×{int(h)} px）、出力解像度を下げてください。")
            elif st.button("画像を変形する", key=f"{KEY}-imggo"):
                with st.spinner("ROIのタイルだけを読み込み、逆写像で再標本化しています..."):
                    rd = _he_reader(he_m)
                    I, z = np.eye(2), np.zeros(2)
                    for name, (MM, tt, fr) in {"he_before": (I, z, None), "he_affine": (Ma, ta, None), f"he_nonrigid_{m}": (Ma, ta, fields[m])}.items():
                        img, valid, rec = CR.warp_he_roi(rd, MM, tt, roi, dst, fr)
                        images[name] = np.dstack([imaging.to_display_rgb(img, he_m.kind, "rgb"), (valid * 255).astype(np.uint8)])
                        recs[name] = rec
                    if avail["IF"]["image"]:
                        img, valid, rec = CR.warp_he_roi(_he_reader(if_m), np.eye(2), np.zeros(2), roi, dst, None)
                        images["if_roi"] = np.dstack([imaging.to_display_rgb(img, if_m.kind, "rgb"), (valid * 255).astype(np.uint8)])
                        recs["if_roi"] = rec
                st.session_state[f"{KEY}-imgs-{auto.auto_id}"] = (images, recs, dst, m, _review_hash(rv))
            got = st.session_state.get(f"{KEY}-imgs-{auto.auto_id}")
            if got:
                images, recs, dst, m, h_used = got
                if h_used != _review_hash(rv) and m == "REVIEWED":
                    st.warning("画像は以前のレビューで変形したものです。再計算してから『画像を変形する』を押してください。")
                nr = recs.get(f"he_nonrigid_{m}", {})
                if nr.get("status") == "APPLIED":
                    st.success(f"非剛体の変形を適用しました（{CR.FIELD_LABELS[m]}、逆写像の最大残差 {nr['inverse_residual_max_um']:.3f} µm）。研究用プレビューです。")
                else:
                    st.warning(f"非剛体の変形は適用されていません: {nr.get('status')}。下の『Nonrigid』画像は Affine と同じか、表示しません。")
                view = st.radio("H&E画像", [k for k in images if k.startswith("he_")], horizontal=True, key=f"{KEY}-imgview",
                                format_func={"he_before": "Before", "he_affine": "Affine", f"he_nonrigid_{m}": "Nonrigid"}.get)
                if view == f"he_nonrigid_{m}" and nr.get("status") != "APPLIED":
                    st.info("適用されていないため、Nonrigid画像は表示しません。")
                else:
                    c = [roi[0], roi[1]]
                    pix, pdef = CR.nuclei_pixel_positions(he_pts, Ma, ta, fields[m] if view.startswith("he_nonrigid") else None, roi, dst)
                    lay = [viewer.image_layer("he", f"H&E ({view})", imaging.encode_png(images[view]), images[view].shape[1], images[view].shape[0],
                                              dst * np.eye(2), c, "moving", mime="image/png")]
                    if "if_roi" in images:
                        lay.insert(0, viewer.image_layer("if", "IF ROI", imaging.encode_png(images["if_roi"]), images["if_roi"].shape[1],
                                                         images["if_roi"].shape[0], dst * np.eye(2), c, "fixed", mime="image/png"))
                    pts = [viewer.points_layer("if", "IF核", if_pts, "#40e0ff", visible=False, radius=2.0)]
                    if view != "he_before":
                        pts.append(viewer.points_layer("he", "H&E核（同じ変換を点に適用）", pdef[auto.he_idx], "#ff4fa3", radius=2.2))
                    scn = {"images": lay, "points": pts, "polys": [], "vectors": None, "bounds": [roi[0], roi[1], roi[2], roi[3]], "units": "µm",
                           "frame_name": f"{S.sample_id}:IF (fixed frame)",
                           "frame_note": "画像は逆写像（IFの画素 → 変位場の逆 → Affineの逆 → H&Eの画素）、核は順写像。両者が重なれば座標の扱いが一致しています。",
                           "banners": [{"level": "warn", "text": "研究用プレビュー（Productionへの正式適用ではありません）"}], "default_mode": "overlay"}
                    components.html(viewer.render(scn, 600), height=620)
                with st.expander("再標本化の記録"):
                    st.json(recs)
        if st.button("結果をファイルに出力（CSV・変位場・画像・安全性・設定・履歴）", key=f"{KEY}-export"):
            ex = CR.export_bundle(S.sample_id, auto, rv, fields, he_pts, he_ids, if_pts, if_ids,
                                  images={k: v for k, v in images.items()}, image_records=recs,
                                  settings={"exclude_conflicts": st.session_state.get(f"{KEY}-excl", False)})
            st.session_state[f"{KEY}-export-{auto.auto_id}"] = str(ex)
            st.success(f"出力しました: {ex}")
        ex = st.session_state.get(f"{KEY}-export-{auto.auto_id}")
        if ex:
            st.download_button("出力をZIPでダウンロード", CR.bundle_zip(CR.Path(ex)), file_name=CR.Path(ex).name + ".zip", mime="application/zip",
                               key=f"{KEY}-zip")
