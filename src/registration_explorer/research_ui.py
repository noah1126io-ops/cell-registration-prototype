"""Research-session UI pieces of the Registration Explorer: session panel, candidate map, structured reviews,
annotations, engine runs (automatic / assisted) and the experiment record (history, comparison, export).

Everything saved here goes to the persistent research store (src/research_store.py); nothing changes a saved
artifact or an acceptance decision.  Reviews are Review Annotations, not ground truth.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from .. import research_store as rs
from . import catalog, jobs, results

KEY = "regx"
SOURCE_COLORS = {"image": "#ff9800", "points": "#40c4ff", "a2": "#b388ff", "init": "#69f0ae", "": "#b39ddb"}
STAGE_TEXT = {"C1": "C1 対応領域の探索", "C2": "C2 精密化（ICP）", "C3": "C3 検証（核配置・一貫性）"}


@st.cache_resource(show_spinner=False)
def store() -> rs.Store:
    return rs.Store()


# ----------------------------------------------------------------------------- session
def session_panel(sample_id: str):
    """Select / create the experiment session that receives everything saved on this page."""
    S = store()
    rows = S.sessions(sample_id)
    k = f"{KEY}-session-{sample_id}"
    pending = st.session_state.pop(f"{KEY}-session-pending-{sample_id}", None)   # created by ensure_session in a previous run
    if pending:
        st.session_state[k] = pending
    with st.container(border=True):
        c1, c2, c3 = st.columns([2.2, 2, 1])
        ids = [r["id"] for r in rows]
        if ids and st.session_state.get(k) not in ids:
            st.session_state[k] = ids[0]          # restore the latest session (survives reloads / restarts)
        sid = c1.selectbox("実験セッション（保存先）", ids or ["（なし）"],
                           format_func=lambda i: next((f"{r['name']} — {r['created_at'][:16]} ({r['created_by']})" for r in rows if r["id"] == i), "（まだありません。右で作成）"),
                           key=k, disabled=not ids)
        name = c2.text_input("新しいセッション名", key=f"{KEY}-newses-{sample_id}", placeholder="例: 8_6 候補の見直し 10/22")
        if c3.button("作成", key=f"{KEY}-newses-btn-{sample_id}", disabled=not name.strip()):
            new = S.create_session(name.strip(), sample_id)
            st.session_state[k] = new
            st.rerun()
        if ids:
            st.caption(f"保存先：{S.path}（SQLite、追記のみ）。評価・注釈・手動変換・実行はこのセッションに記録され、再読込・再起動後も残ります。")
    return sid if ids else None


def ensure_session(sample_id: str) -> str:
    k = f"{KEY}-session-{sample_id}"
    sid = st.session_state.get(f"{KEY}-session-pending-{sample_id}") or st.session_state.get(k)
    if sid and store().session(sid):
        return sid
    from datetime import datetime
    sid = store().create_session(f"自動作成 {datetime.now():%Y-%m-%d %H:%M}", sample_id)
    st.session_state[f"{KEY}-session-pending-{sample_id}"] = sid    # the selectbox already exists in this run: apply next run
    return sid


# ----------------------------------------------------------------------------- candidate map
def candidate_table(rec: results.ResultRecord, sample_id: str):
    revs = store().reviews(sample_id, latest_only=True)
    rows = []
    for c in rec.candidates:
        m = c.metrics or {}
        rv = next((r for r in revs if r["candidate_id"] == c.cand_id and r["run_id"] == _run_id(rec)), None)
        ov = (c.overlap or {}).get("overlap_area_mm2")
        rows.append({"候補": c.cand_id, "探索方法": results_source_label(c), "順位": c.rank, "回転 (°)": round(c.theta_deg, 1),
                     "倍率": round(c.scale, 4), "平行移動": f"({c.t_um[0]:.0f}, {c.t_um[1]:.0f})" if c.t_um is not None else "—",
                     "推定重なり (mm²)": round(ov, 3) if ov is not None else None, "探索スコア": None if c.score_z is None else round(c.score_z, 2),
                     "1位との差": None if c.gap_to_top is None else round(c.gap_to_top, 3),
                     "ICPの移動 (µm)": None if m.get("icp_median_shift_um") is None else round(m["icp_median_shift_um"], 1),
                     "≤3 µm": None if m.get("le_3um") is None else round(m["le_3um"], 3),
                     "偶然一致": None if m.get("null_le_3um") is None else round(m["null_le_3um"], 3),
                     "核配置の基準": {True: "満たす", False: "満たさない", None: "—"}.get(c.geometric_pass if c.geometric_pass is not None else m.get("geometric_pass")),
                     "注意": ", ".join(c.flags), "レビュー": rs.JUDGEMENTS.get(rv["judgement"], "") if rv else ""})
    return pd.DataFrame(rows)


def results_source_label(c):
    from ..registration_core import SOURCE_LABELS
    return SOURCE_LABELS.get(c.source, c.note or "—")


def _run_id(rec):
    rid = rec if isinstance(rec, str) else rec.rid
    return rid.split(":", 1)[1] if rid.startswith("wf:") else rid


def stage_panel(rec: results.ResultRecord):
    """Where did it fail?  Localization vs refinement vs validation, per search source (engine runs)."""
    art = rec.artifact or {}
    d = art.get("decision")
    if not d:
        st.caption("この結果には処理段階ごとの記録がありません（2026-10-22以前の形式）。")
        return
    lab = rec.safety_label()
    kind = "success" if d["accepted"] else ("info" if d.get("accepted_tissue_level") else "warning")
    getattr(st, kind)(f"**{results.SAFETY_TEXT.get(lab, lab)}** — {d.get('reason', '')}")
    rel = art.get("section_relation") or {}
    adjacent = rel.get("value") == "adjacent"
    if rel:
        secs = " / ".join(f"{k} {v}" for k, v in (rel.get("sections") or {}).items())
        st.caption(f"切片の関係: **{rel.get('text', rel.get('value'))}**" + (f"（{secs}、{'ファイル名から推定' if rel.get('source') == 'filename' else '設定'}）" if secs or rel.get("source") else ""))
    if d.get("precision_note"):
        st.caption("精度の目安: " + d["precision_note"])
    rows = []
    loc = art.get("localization", {})
    for src, p in (d.get("per_source") or {}).items():
        L = loc.get(src, {})
        rows.append({"探索": L.get("label", src), "C1 探索の判定": f"{p['verdict']}（margin {p['margin']:.3f}）" if isinstance(p.get("margin"), (int, float)) else p["verdict"],
                     "C1 の意味": p.get("verdict_text", ""),
                     "C2/C3 核配置（最上位）": "対象外（隣接切片：核は別の細胞）" if adjacent else ("基準を満たす" if p["geometry_ok"] else "基準を満たさない"),
                     **({"C3 象限ごとの再探索": ("—" if p.get("quadrant_ok") is None else ("同じ位置を再現" if p["quadrant_ok"] else "再現しない"))} if adjacent else
                        {"探索poseの時点で核配置が合うか": "合う" if p.get("search_pose_geometry_ok") else "合わない"}),
                     "この探索での採択": ("組織レベルで位置決定" if adjacent else "採択") if p["accepted_by_rule"] else "未採択",
                     "失敗した段階": STAGE_TEXT.get(p.get("failure_stage"), "—"), "使用した特徴": ", ".join(L.get("layers") or []),
                     "探索数 / 時間": f"{L.get('n_poses', '—')} pose / {L.get('runtime_s', 0):.0f} s" if L.get("runtime_s") else "—"})
    st.table(pd.DataFrame(rows).set_index("探索"))
    if d.get("failure_stage"):
        st.info(f"止まった段階（主の探索）: **{STAGE_TEXT.get(d['failure_stage'])}** — {d.get('failure_stage_text') or ''}")
    top_img = next((c for c in (art.get("candidates") or []) if c.get("source") == "image" and c.get("rank") == 0), None)
    qc = ((top_img or {}).get("validation") or {}).get("quadrant_consistency")
    if qc:
        st.markdown(f"**象限ごとの再探索**（IF視野を2×2に分け、各象限だけで位置を探し直す。{qc['rule']['min_agree']}象限以上が "
                    f"{qc['rule']['max_shift_um']:.0f} µm・{qc['rule']['max_dtheta_deg']:.0f}° 以内なら合格）: {qc['n_agree']} / 4 象限が一致")
        st.table(pd.DataFrame([{"象限": ["左上", "右上", "左下", "右下"][q["quadrant"]],
                                "位置の差 (µm)": round(q["shift_um"], 1) if q.get("ok") else "—",
                                "回転の差 (°)": round(q["d_theta_deg"], 1) if q.get("ok") else "—",
                                "z": round(q["z"], 1) if q.get("ok") else "—", "備考": q.get("note", "")} for q in qc["quadrants"]]).set_index("象限"))
    ims = art.get("image_status") or {}
    st.caption("画像の利用: " + ", ".join(ims.get("flags") or []) + ("（" + " / ".join(ims.get("reasons") or []) + "）" if ims.get("reasons") else ""))
    # image vs point top candidates
    tops = {c.source: c for c in rec.candidates if c.rank == 0}
    if "image" in tops and "points" in tops:
        a, b = tops["image"], tops["points"]
        me = np.asarray((art["input"]["moving"].get("extent") or [1000, 1000]), float)
        probe = np.array([[0, 0], [me[0], 0], [me[0], me[1]], [0, me[1]], me / 2])
        d_um = float(np.median(np.linalg.norm(probe @ np.asarray(a.M_final).T + a.t_final - (probe @ np.asarray(b.M_final).T + b.t_final), axis=1)))
        dr = (a.theta_deg - b.theta_deg + 180) % 360 - 180
        msg = (f"画像探索の1位と核探索の1位の差：位置 {d_um:.0f} µm（視野の四隅と中心の中央値）、回転 {dr:.1f}°、倍率比 {a.scale / b.scale:.3f}。"
               + ("**同じ領域を指しています。**" if d_um < 40 else "**別の領域を指しています。**"))
        st.markdown(msg)


# ----------------------------------------------------------------------------- reviews / annotations
def review_form(sample_id, rec, cand_id, transform_ctx: dict, frame_key: str):
    """Structured review of the candidate / transform currently shown."""
    st.markdown(f"**評価する対象**：{rec.config if rec else '—'} / 候補 {cand_id or '最終変換'}（{frame_key} Fixed）")
    with st.form(f"{KEY}-review-form", clear_on_submit=True):
        j = st.radio("候補領域の評価", list(rs.JUDGEMENTS), format_func=rs.JUDGEMENTS.get, horizontal=True, index=2)
        probs = st.multiselect("問題の種類（複数選択可）", list(rs.PROBLEMS), format_func=rs.PROBLEMS.get)
        com = st.text_area("コメント（任意）", placeholder="例: 右下の腺管の配置が合わない。回転は近いが倍率が小さすぎる")
        ok = st.form_submit_button("評価を保存")
    if ok:
        sid = ensure_session(sample_id)
        rid = store().add_review(sample_id=sample_id, session_id=sid, run_id=_run_id(rec) if rec else None, candidate_id=cand_id,
                                 transform_id=transform_ctx.get("transform_id"), judgement=j, problems=probs, comment=com,
                                 view={k: v for k, v in transform_ctx.items() if k != "transform_id"})
        st.success(f"評価を保存しました（{rid}）。人による評価はReview Annotationとして記録され、正解（ground truth）としては扱いません。")


def annotation_list(sample_id, *, session_only=False, session_id=None):
    rows = store().annotations(sample_id, session_id if session_only else None)
    if not rows:
        st.caption("注釈はまだありません。")
        return []
    df = pd.DataFrame([{"ID": r["id"], "種類": rs.ANNOTATION_KINDS.get(r["kind"], r["kind"]), "名前": r["label"], "frame": r["frame"],
                        "作成": r["created_at"][:16], "作成者": r["created_by"], "セッション": r["session_id"]} for r in rows])
    st.table(df.drop(columns=["セッション"]).set_index("ID"))
    return rows


def review_list(sample_id, session_id=None):
    rows = store().reviews(sample_id, session_id)
    if not rows:
        st.caption("評価はまだありません。")
        return
    st.table(pd.DataFrame([{"日時": r["created_at"][:16], "評価": rs.JUDGEMENTS[r["judgement"]],
                            "問題": ", ".join(rs.PROBLEMS.get(p, p) for p in json.loads(r["problems_json"])), "コメント": r["comment"],
                            "実行": r["run_id"], "候補": r["candidate_id"], "作成者": r["created_by"]} for r in rows]).set_index("日時"))


def save_annotation_from_transform(sample_id, kind, M, t, fixed, moving, label, related):
    sid = ensure_session(sample_id)
    return store().add_annotation(sample_id=sample_id, session_id=sid, kind=kind, frame=f"{sample_id}:{fixed}",
                                  payload={"M": np.asarray(M).tolist(), "t": np.asarray(t).tolist(), "fixed": fixed, "moving": moving, **related},
                                  label=label, run_id=related.get("run_id"), candidate_id=related.get("candidate_id"))


# ----------------------------------------------------------------------------- engine runs
def run_panel(S: catalog.SampleInfo, manifest_file: Path):
    st.markdown("研究Workflow C（共通エンジン `src/registration_core`）を1件Slurmに投入します。処理は **C1 対応領域の探索 → C2 精密化 → C3 検証 → C4 記録** の順です。"
                "画像が両側にあり使える品質なら画像探索を先に行い、核探索の結果と並べて記録します。")
    c1, c2, c3 = st.columns([2.4, 1.2, 1])
    wf = c1.selectbox("処理", list(jobs.WORKFLOWS), format_func=jobs.WORKFLOWS.get, key=f"{KEY}-wf-kind")
    fixed = c2.selectbox("Fixed", list(S.modalities), format_func=lambda k: {"HE": "H&E Fixed / 蛍光 Moving", "IF": "蛍光 Fixed / H&E Moving"}.get(k, k), key=f"{KEY}-wf-fixed")
    use_img = c3.toggle("画像を使う", value=True, key=f"{KEY}-wf-img", disabled=wf in ("points", "a2"))
    feat = st.selectbox("画像探索で比べる画像特徴", list(jobs.IMAGE_FEATURES), format_func=jobs.IMAGE_FEATURES.get, key=f"{KEY}-wf-feat",
                        disabled=wf in ("points", "a2") or not use_img)
    ann_ids, cons = [], None
    if wf == "assisted":
        rows = store().annotations(S.sample_id)
        usable = [r for r in rows if r["kind"] in ("search_roi", "roi", "exclusion", "manual_init", "landmarks")]
        ann_ids = st.multiselect("使う注釈（明示的に選んだものだけが探索に反映されます）", [r["id"] for r in usable],
                                 format_func=lambda i: next(f"{rs.ANNOTATION_KINDS[r['kind']]}: {r['label'] or r['id']} ({r['created_at'][:16]})" for r in usable if r["id"] == i),
                                 key=f"{KEY}-wf-anns")
        st.caption("注釈を使った実行は『assisted』として記録し、完全自動の性能評価には含めません。結果は自動採択されません。")
        if ann_ids:
            try:
                cons = rs.constraints_from_annotations(store(), ann_ids, fixed, S.other(fixed))
            except (ValueError, KeyError) as exc:
                st.error(f"注釈を制約に変換できません: {exc}")
                ann_ids = []
    loc = "auto" if wf == "assisted" else wf
    cfg = {"engine": jobs.ENGINE, "sample": S.sample_id, "fixed": fixed, "localization": loc, "use_images": bool(use_img and wf not in ("points", "a2")),
           "mode": "assisted" if wf == "assisted" else "automatic", "annotation_ids": ann_ids,
           "constraints": None if cons is None else {k: v for k, v in cons.__dict__.items()}, "method": "WF", "variant": wf,
           "image_feature": feat}
    errs = jobs.validate(cfg, S)
    if errs:
        st.error("実行できません: " + " / ".join(errs))
    hit = jobs.cached_run(cfg) or None
    if hit is None:
        for r in jobs.list_runs():
            if r["config"].get("key") == jobs.config_key(cfg) and r.get("state") == "completed" and (Path(r["run_dir"]) / "artifact.json").is_file():
                hit = r; break
    if hit:
        st.success(f"同じ設定の完了済み結果があります（run {hit['run_id']}）。『研究Workflow C』の結果一覧から表示できます。")
    act = jobs.active_same(cfg)
    if act:
        st.info(f"同じ設定のジョブが実行中です（run {act['run_id']}）。")
    if st.session_state.pop(f"{KEY}-wf-clear", False):
        st.session_state[f"{KEY}-wf-ok"] = False
    if f"{KEY}-wf-msg" in st.session_state:
        kind, msg = st.session_state.pop(f"{KEY}-wf-msg")
        getattr(st, kind)(msg)
    ok = st.checkbox(f"Slurm（mib-dbia、CPU 4コア・{jobs.ENGINE_SLURM['mem']}・最長{jobs.ENGINE_SLURM['time']}）にジョブを1件投入することを確認しました",
                     key=f"{KEY}-wf-ok")
    if st.button("研究Workflow Cを投入", disabled=bool(errs) or not ok or bool(hit) or bool(act), key=f"{KEY}-wf-go"):
        try:
            sid = ensure_session(S.sample_id)
            rd, jid = jobs.submit(cfg, manifest_file)
            store().register_run(run_id=rd.name, session_id=sid, sample_id=S.sample_id, mode=cfg["mode"], localization=loc, fixed=fixed,
                                 moving=S.other(fixed), run_dir=rd, params=cfg, constraints=cfg["constraints"], annotation_ids=ann_ids,
                                 slurm_job_id=jid)
            st.session_state[f"{KEY}-wf-msg"] = ("success", f"投入しました: job {jid}, run {rd.name}（セッションに記録）")
        except (RuntimeError, OSError, ValueError) as exc:
            st.session_state[f"{KEY}-wf-msg"] = ("error", f"投入できませんでした: {exc}")
        st.session_state[f"{KEY}-wf-clear"] = True
        st.rerun()


def sync_runs():
    """Copy finished engine runs into the store (state + artifact summary + the automatic transform)."""
    S = store()
    for r in jobs.list_runs():
        cfg = r["config"]
        if cfg.get("engine") != jobs.ENGINE:
            continue
        row = S.run(r["run_id"])
        if row is None:
            continue
        art = Path(r["run_dir"]) / "artifact.json"
        if r.get("state") == "completed" and art.is_file() and not row["artifact_path"]:
            S.update_run(r["run_id"], state="completed", artifact_path=art)
            a = json.loads(art.read_text())
            tr = a.get("transform")
            if tr:
                S.add_transform(sample_id=a["sample"], session_id=row["session_id"], run_id=r["run_id"], candidate_id=tr["candidate"], kind="automatic",
                                fixed=a["input"]["fixed"]["key"], moving=a["input"]["moving"]["key"], M=tr["forward"]["M"], t=tr["forward"]["t"],
                                status=a["decision"]["status"], label=f"{a['config']['localization']} {a['mode']}", detail={"accepted": a["decision"]["accepted"]})
        elif r.get("state") in ("failed",) and row["state"] != "failed":
            S.update_run(r["run_id"], state="failed")
        elif r.get("state") != row["state"] and r.get("state") in ("queued", "running"):
            S.update_run(r["run_id"], state=r["state"])


# ----------------------------------------------------------------------------- experiment record
def record_tab(sample_id: str, session_id: str | None):
    S = store()
    scope = st.radio("表示範囲", ["このセッション", "このサンプルのすべて"], horizontal=True, key=f"{KEY}-rec-scope")
    sid = session_id if scope == "このセッション" else None
    runs = S.runs(sample_id, sid)
    st.markdown("**実行（Registration runs）**")
    if runs:
        rows = []
        for r in runs:
            m = {}
            if r["artifact_path"] and Path(r["artifact_path"]).is_file():
                a = json.loads(Path(r["artifact_path"]).read_text())
                sel = next((c for c in a["candidates"] if c["id"] == a["decision"]["selected_candidate"]), None)
                if sel:
                    v = sel["validation"]
                    f = ((v.get("if_frame") or {}).get("final") or v.get("fixed_frame") or {})
                    m = {"NN median (µm)": f.get("nn_median_um"), "≤3 µm": f.get("le_3um")}
            rows.append({"run": r["id"], "モード": {"automatic": "自動", "assisted": "注釈使用"}.get(r["mode"], r["mode"]), "探索": r["localization"],
                         "Fixed": r["fixed"], "状態": r["state"], "判定": r["status"], "採択": {1: "採択", 0: "未採択"}.get(r["accepted"], "—"),
                         "選ばれた候補": r["selected_candidate"], "止まった段階": r["failure_stage"], **m, "注釈": r["annotation_ids_json"],
                         "作成": r["created_at"][:16], "artifact sha256": (r["artifact_sha256"] or "")[:12]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.caption("記録された実行はありません。")
    st.markdown("**変換（自動・手動・対応点）** — 追記のみ。元の自動推定は上書きされません。")
    tfs = S.transforms(sample_id, sid)
    if tfs:
        st.dataframe(pd.DataFrame([{"ID": t["id"], "種類": t["kind"], "状態": t["status"], "名前": t["label"], "Fixed": t["fixed"],
                                    "元の変換": t["parent_id"], "実行": t["run_id"], "候補": t["candidate_id"], "作成": t["created_at"][:16],
                                    "作成者": t["created_by"]} for t in tfs]), hide_index=True, width="stretch")
    else:
        st.caption("保存された変換はありません。")
    st.markdown("**候補の評価（Review Annotation）**")
    review_list(sample_id, sid)
    st.markdown("**注釈（ROI・対応点・除外・初期位置）**")
    annotation_list(sample_id, session_only=sid is not None, session_id=sid)
    with st.expander("操作履歴（event log）"):
        ev = S.events(sample_id, sid, limit=300)
        st.dataframe(pd.DataFrame([{"日時": e["ts"], "操作": e["action"], "内容": e["payload_json"], "作成者": e["actor"]} for e in ev]),
                     hide_index=True, width="stretch")
    data = S.export_json(sample_id, sid)
    c1, c2 = st.columns(2)
    c1.download_button("JSONでエクスポート", json.dumps(data, ensure_ascii=False, indent=1), file_name=f"{sample_id}_{sid or 'all'}_research_record.json",
                       mime="application/json", key=f"{KEY}-exp-json")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in S.export_csv(sample_id, sid).items():
            z.writestr(name, text)
    c2.download_button("CSV（zip）でエクスポート", buf.getvalue(), file_name=f"{sample_id}_{sid or 'all'}_research_record_csv.zip",
                       mime="application/zip", key=f"{KEY}-exp-csv")
    st.caption(f"DB: {S.path} / schema v{S.schema_version()} / integrity_check: {S.integrity_check()}")
