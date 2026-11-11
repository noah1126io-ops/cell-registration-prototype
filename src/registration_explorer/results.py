"""Result adapter: saved research artifacts -> one ResultRecord shape for the explorer.

Sources (read-only):
  oir-20261019           TransformArtifact JSON + mode_comparison.csv row + run cache (search candidates)
  e0x-ablation-20261020  ablation cache (X0-X3; top candidates with ICP + evaluation)
  explorer-runs          runs started from the explorer (same JSON shape as the ablation cache)
  stored-transform       a transform recorded in an earlier analysis (no verdict)

A record's transform maps MOVING world -> FIXED world (frame-local units).  Display in the other direction
uses the inverse and is labelled as derived.  Only records with accepted=True are formal results; everything
else is PREVIEW ONLY and is never handed to Workflow D.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# user-facing names (internal keys stay in details / logs)
METHOD_LABELS = {"WF": "研究Workflow C：対応領域の探索 → 精密化 → 検証", "A2": "核密度による対応領域探索（従来法）",
                 "E0x": "視野サイズの違いに対応した核探索", "E1": "片側画像を使った補助解析", "E2": "両画像からの対応領域探索",
                 "E3": "画像と核配置の統合探索", "ABL": "核探索の構成要素の比較（2026-10-20 ablation）", "RUN": "Explorerから実行した結果（旧Runner）",
                 "NC": "陰性対照（核の並べ替え）", "MAN": "手動調整した変換（未検証）"}
INTERNAL_IDS = {"WF": "registration_core run_workflow (C1-C4)", "A2": "A2 / E0", "E0x": "E0x", "E1": "E1", "E2": "E2", "E3": "E3",
                "ABL": "X0-X3", "RUN": "explorer runner", "NC": "NCshuf", "MAN": "manual"}
SAFETY_TEXT = {"UNIQUE / Accepted": "採択：探索で一意に決まり、核配置の整合も事前の基準を満たした",
               "AMBIGUOUS": "曖昧：上位の候補どうしの差が小さく、対応領域が1つに決まらない（未採択）",
               "LOW_PEAK": "ピークが弱い：最上位の候補でも探索スコアが基準に届かない（未採択）",
               "FAIL": "不合格：探索は一意でも、核配置の整合が偶然一致と区別できない（未採択）",
               "POINT_ONLY_FALLBACK": "画像探索が基準を満たさず、核探索の結果を採用した",
               "IMAGE_UNUSABLE": "画像が位置合わせに使えない（品質・座標の問題）。核探索に切り替え",
               "UNVERIFIED": "判定の記録がない（未検証）", "MANUAL (UNVERIFIED)": "手動で調整した変換（判定を経ていない）",
               "CONFLICT": "矛盾：画像探索と核探索がそれぞれ基準を満たしたが、別の領域を指している（未採択）",
               "NOT_ACCEPTED": "未採択：どの探索も事前の基準を満たさない",
               "ASSISTED_REVIEW_REQUIRED": "注釈を使った探索の結果（自動採択しない・要レビュー）",
               "TISSUE_LEVEL": "組織レベルで位置決定：隣接切片のため核は対応しない。画像探索が一意で、象限ごとの再探索も同じ位置を再現した"
                               "（細胞レベルの採択ではない・Workflow Dへは渡さない）"}
IMAGE_METHODS = ("E1", "E2", "E3")
ACCEPT_RULE = "UNIQUE search verdict AND <=3 um proximity >= 0.25 AND >= 3x mirror null (prespecified 2026-10-19; unchanged)"


@dataclass
class Candidate:
    cand_id: str
    rank: int
    theta_deg: float
    scale: float
    chirality: str
    t_um: list
    score_z: float | None
    ncc: float | None
    gap_to_top: float | None
    M_search: np.ndarray
    t_search: np.ndarray
    M_final: np.ndarray | None = None
    t_final: np.ndarray | None = None
    metrics: dict = field(default_factory=dict)
    overlap: dict = field(default_factory=dict)
    note: str = ""
    source: str = ""                 # engine runs: image | points | a2 | init
    flags: list = field(default_factory=list)
    geometric_pass: bool | None = None


@dataclass
class ResultRecord:
    rid: str
    sample: str
    method: str                      # key of METHOD_LABELS
    config: str
    fixed: str
    moving: str
    M: np.ndarray | None             # moving world -> fixed world
    t: np.ndarray | None
    verdict: str
    margin: float | None
    accepted: bool | None
    status_flags: list
    failure_reasons: list
    fallback_source: str | None
    metrics: dict
    candidates: list
    source: str
    files: list
    inputs: str = "points"
    artifact: dict | None = None
    note: str = ""

    @property
    def label(self):
        return f"{METHOD_LABELS.get(self.method, self.method)} | {self.config} | {self.moving}->{self.fixed}"

    def safety_label(self) -> str:
        for st in ("CONFLICT", "NOT_ACCEPTED", "ASSISTED_REVIEW_REQUIRED", "TISSUE_LEVEL"):
            if st in self.status_flags and not self.accepted:
                return st
        if "MANUAL_UNVERIFIED" in self.status_flags:
            return "MANUAL (UNVERIFIED)"
        if "IMAGE_INCOMPATIBLE" in self.status_flags:
            return "IMAGE_UNUSABLE"
        if self.accepted:
            return "UNIQUE / Accepted"
        if "POINT_ONLY_FALLBACK" in self.status_flags:
            return "POINT_ONLY_FALLBACK"
        if self.verdict in ("AMBIGUOUS", "LOW_PEAK"):
            return self.verdict
        if self.accepted is None:
            return "UNVERIFIED"
        return "FAIL"


def _f(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _params(M, t):
    M = np.asarray(M, float)
    det = float(np.linalg.det(M))
    C = np.diag([1.0, -1.0 if det < 0 else 1.0])
    s = math.sqrt(abs(det))
    R = M @ C / max(s, 1e-12)
    return {"scale": s, "rotation_deg": math.degrees(math.atan2(R[1, 0], R[0, 0])), "reflect": det < 0,
            "tx": float(t[0]), "ty": float(t[1]), "anisotropy": float(np.linalg.svd(M)[1][0] / np.linalg.svd(M)[1][1])}


# ----------------------------------------------------------------------------- oir-20261019
def _oir_method(config: str) -> str:
    if config.startswith("NCshuf"):
        return "NC"
    head = config.split("_")[0]
    for k in ("E0x", "E0", "E1", "E2", "E3"):
        if head.startswith(k):
            return "A2" if k == "E0" else k
    return config


def _oir_records(sample, src) -> list[ResultRecord]:
    import pandas as pd
    art_dir, cache_dir = Path(src["artifacts_dir"]), Path(src["cache_dir"])
    rows = {}
    if Path(src["metrics_csv"]).is_file():
        df = pd.read_csv(src["metrics_csv"])
        rows = {r["config"]: r for _, r in df[df["sample"] == sample].iterrows()}
    out = []
    for p in sorted(art_dir.glob(f"{sample}_*.json")):
        config = p.stem[len(sample) + 1:]
        a = json.loads(p.read_text())
        row = rows.get(config, {})
        run_path = cache_dir / f"real_{sample}_{config}.json"
        run = json.loads(run_path.read_text()) if run_path.is_file() else {}
        fixed = a["target_frame"]["modality"]; moving = a["source_frame"]["modality"]
        fw = a.get("forward_transform") or {}
        M = np.asarray(fw["M"], float) if fw.get("M") else None
        t = np.asarray(fw["t"], float) if fw.get("t") else None
        fsrc = row.get("final_transform_source") if isinstance(row.get("final_transform_source"), str) else None
        metrics = {
            "nn_median_um": _f(row.get("nn_median_um")), "nn_p95_um": _f(row.get("nn_p95_um")), "le_3um": _f(row.get("le_3um")),
            "le_1um": _f(row.get("le_1um")), "le_2um": _f(row.get("le_2um")), "le_5um": _f(row.get("le_5um")),
            "null_le_3um": _f(row.get("null_le_3um")), "inlier_over_null": _f(row.get("inlier_over_null")), "n_eval": _f(row.get("n_if_eval")),
            "heldout_le_3um": _f(row.get("heldout_le_3um_mean")),
            "scale_he_to_if": _f(row.get("scale_he_to_if")), "rotation_he_to_if_deg": _f(row.get("rotation_deg_he_to_if")),
            "img_tissue_dice": _f(row.get("img_tissue_dice")), "img_nuclear_corr": _f(row.get("img_nuclear_corr")),
            "img_tissue_independent": row.get("img_tissue_independent_of_search"), "img_nuclear_independent": row.get("img_nuclear_independent_of_search"),
            "ref_median_um": _f(row.get("ref_median_um")), "runtime_s": _f(row.get("runtime_s")), "peak_mem_mb": _f(row.get("peak_mem_mb")),
            "icp_median_shift_um": _f(row.get("icp_median_shift_um")),
            "population": "IF nuclei inside the mapped H&E footprint (10 um inset), evaluated in the IF frame",
        }
        if M is not None:
            T_he_if = (M, t) if fixed == "IF" else _inv(M, t)
            metrics["translation_he_to_if_um"] = [float(T_he_if[1][0]), float(T_he_if[1][1])]
        cands = []
        for i, c in enumerate(((run.get("search") or {}).get("fine") or [])):
            Ms = c["scale"] * _rot(c["theta"]) @ (np.diag([1.0, -1.0]) if c["chirality"] == "reflect" else np.eye(2))
            top_score = _f(run["search"]["fine"][0].get("score"))
            sc = _f(c.get("score"))
            gap = 0.0 if i == 0 else ((top_score - sc) / abs(top_score) if (top_score and sc is not None) else None)
            cands.append(Candidate(f"{config}#c{i}", i, c["theta"], c["scale"], c["chirality"], c["t"], sc,
                                   _f((list((c.get("layer_ncc") or {}).values()) or [None])[0]), gap,
                                   Ms, np.asarray(c["t"], float), note="search pose only (ICP not stored for non-top candidates in 2026-10-19 run)"))
        verdict = (a.get("confidence") or {}).get("search_verdict") or row.get("search_verdict") or "UNKNOWN"
        out.append(ResultRecord(
            rid=f"oir:{sample}:{config}", sample=sample, method=_oir_method(config), config=config, fixed=fixed, moving=moving, M=M, t=t,
            verdict=str(verdict), margin=_f((a.get("confidence") or {}).get("search_margin")), accepted=bool(a.get("accepted")),
            status_flags=list(a.get("image_status") or []), failure_reasons=list(a.get("failure_reasons") or []),
            fallback_source=(fsrc if fsrc and fsrc != config else None), metrics=metrics, candidates=cands, source="oir-20261019",
            files=[str(p)] + ([str(run_path)] if run else []), inputs=str(row.get("inputs", "points")), artifact=a))
    return out


def _rot(th):
    a = math.radians(th)
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def _inv(M, t):
    Mi = np.linalg.inv(np.asarray(M, float))
    return Mi, -Mi @ np.asarray(t, float)


# ----------------------------------------------------------------------------- ablation / explorer runs
def record_from_variant_json(rec: dict, rid: str, source: str, config: str, method="ABL", files=()) -> ResultRecord:
    cands = []
    for c in rec.get("candidates", []):
        Ts = c["T_search"]; Tf = c.get("T_final")
        ev = c.get("eval_if_frame") or {}
        fin = ev.get("final") or {}
        fe = c.get("final_eval") or {}
        m = {"nn_median_um": _f(fin.get("nn_median_um")), "le_3um": _f(fin.get("le_3um")), "null_le_3um": _f((ev.get("null") or {}).get("le_3um")),
             "inlier_over_null": _f(ev.get("inlier_over_null")), "n_eval": _f(ev.get("n_if_eval")),
             "geometric_pass": ev.get("geometric_pass") if ev else fe.get("geometric_pass"),
             "fixed_frame_le_3um": _f(fe.get("le_3um")), "fixed_frame_null_le_3um": _f(fe.get("null_le_3um")),
             "img_tissue_dice": _f((ev.get("image") or {}).get("tissue_dice")), "img_nuclear_corr": _f((ev.get("image") or {}).get("nuclear_corr")),
             "ref_median_um": _f((ev.get("vs_reference_um") or {}).get("median")),
             "icp_median_shift_um": _f(((c.get("icp") or {}).get("pose_change") or {}).get("median_shift_um")),
             "search_le_3um": _f((c.get("search_eval") or {}).get("le_3um"))}
        cands.append(Candidate(f"{config}#c{c['rank']}", c["rank"], c["theta_deg"], c["scale"], c["chirality"], c["t_um"], _f(c.get("score_z")),
                               _f(c.get("ncc")), _f(c.get("relative_gap_to_top")), np.asarray(Ts["M"]), np.asarray(Ts["t"]),
                               np.asarray(Tf["M"]) if Tf else None, np.asarray(Tf["t"]) if Tf else None, m, c.get("overlap") or {},
                               note=c.get("icp_note", "")))
    top = cands[0] if cands else None
    accepted = bool(rec.get("accepted"))
    metrics = dict(top.metrics) if top else {}
    metrics.update({"runtime_s": _f(rec.get("runtime_s")), "peak_mem_mb": _f(rec.get("peak_mem_mb")),
                    "population": "IF nuclei inside the mapped H&E footprint (10 um inset), evaluated in the IF frame"})
    if top is not None and top.M_final is not None:
        T_he_if = (top.M_final, top.t_final) if rec["fixed"] == "IF" else _inv(top.M_final, top.t_final)
        p = _params(*T_he_if)
        metrics.update({"scale_he_to_if": p["scale"], "rotation_he_to_if_deg": p["rotation_deg"], "translation_he_to_if_um": [p["tx"], p["ty"]]})
    reasons = []
    if not accepted:
        if rec.get("verdict") != "UNIQUE":
            reasons.append(f"search verdict {rec.get('verdict')} (margin {rec.get('margin')})")
        if top is not None and not (top.metrics.get("geometric_pass")):
            reasons.append("nuclear geometry inconsistent (prespecified acceptance not met)")
    return ResultRecord(rid=rid, sample=rec["sample"], method=method, config=config, fixed=rec["fixed"], moving=rec["moving"],
                        M=top.M_final if top is not None else None, t=top.t_final if top is not None else None,
                        verdict=str(rec.get("verdict")), margin=_f(rec.get("margin")), accepted=accepted, status_flags=[],
                        failure_reasons=reasons, fallback_source=None, metrics=metrics, candidates=cands, source=source, files=list(files),
                        note=f"canvas={rec.get('canvas')} icp={rec.get('icp')}")


def _ablation_records(sample, src):
    out = []
    for p in sorted(Path(src["cache_dir"]).glob(f"abl_{sample}_*.json")):
        rec = json.loads(p.read_text())
        cfg = f"{rec['variant']}_{rec['fixed']}fix"
        out.append(record_from_variant_json(rec, f"abl:{sample}:{cfg}", "e0x-ablation-20261020", cfg, files=[str(p)]))
    return out


def _metrics_from_validation(v):
    ff = v.get("fixed_frame") or {}
    ifr = v.get("if_frame") or {}
    fin, nul = ifr.get("final") or {}, ifr.get("null") or {}
    img = ifr.get("image") or {}
    m = {"nn_median_um": _f(fin.get("nn_median_um")), "le_3um": _f(fin.get("le_3um")), "null_le_3um": _f(nul.get("le_3um")),
         "inlier_over_null": _f(ifr.get("inlier_over_null")), "n_eval": _f(ifr.get("n_if_eval")), "geometric_pass": ifr.get("geometric_pass"),
         "fixed_frame_nn_median_um": _f(ff.get("nn_median_um")), "fixed_frame_le_3um": _f(ff.get("le_3um")),
         "fixed_frame_null_le_3um": _f(ff.get("null_le_3um")), "fixed_frame_geometric_pass": ff.get("geometric_pass"),
         "search_pose_le_3um": _f((v.get("search_pose_fixed_frame") or {}).get("le_3um")),
         "img_tissue_dice": _f(img.get("tissue_dice")), "img_nuclear_corr": _f(img.get("nuclear_corr")),
         "population": ("IF frame: IF nuclei inside the mapped H&E footprint (10 um inset)" if ifr else
                        "fixed frame: fixed nuclei inside the mapped moving footprint")}
    if not ifr:
        m.update({"nn_median_um": m["fixed_frame_nn_median_um"], "le_3um": m["fixed_frame_le_3um"], "null_le_3um": m["fixed_frame_null_le_3um"],
                  "geometric_pass": ff.get("geometric_pass"), "n_eval": _f(ff.get("n_eval"))})
    if (v.get("a2_detector") or {}):
        d = v["a2_detector"]
        m.update({"le_3um": _f(d.get("inlier3")), "null_le_3um": _f(d.get("inlier3_null")), "inlier_over_null": _f(d.get("inlier_over_null")),
                  "geometric_pass": d.get("detector_status") == "PASS", "population": "A2 detector (moving nuclei in the fixed hull)"})
    return m


def record_from_engine_artifact(art: dict, rid: str, files=()) -> ResultRecord:
    """cellreg-registration-artifact/1 (src/registration_core) -> ResultRecord with all candidates of all sources."""
    fixed, moving = art["input"]["fixed"]["key"], art["input"]["moving"]["key"]
    d = art["decision"]
    cands = []
    for c in art["candidates"]:
        s = c["search"]
        ov = (c["validation"].get("overlap") or {})
        cands.append(Candidate(c["id"], c["rank"], s["theta_deg"], s["scale"], s["chirality"], s["t"], _f(s.get("score")),
                               _f((list((s.get("layer_ncc") or {}).values()) or [None])[0]), _f(s.get("relative_gap_to_top")),
                               np.asarray(c["T_search"]["M"]), np.asarray(c["T_search"]["t"]), np.asarray(c["T_final"]["M"]), np.asarray(c["T_final"]["t"]),
                               {**_metrics_from_validation(c["validation"]), "icp_median_shift_um": _f(((c.get("refine") or {}).get("pose_change") or {}).get("median_shift_um"))},
                               {"overlap_area_mm2": ov.get("overlap_area_mm2"), "overlap_polygon_fixed_um": ov.get("overlap_polygon_fixed")},
                               note=c.get("source_label", ""), source=c["source"], flags=list(c.get("flags") or []), geometric_pass=c.get("geometric_pass")))
    sel = next((c for c in cands if c.cand_id == d.get("selected_candidate")), None)
    tr = art.get("transform") or {}
    fw = tr.get("forward") or {}
    flags = [d["status"]]          # image usability is shown from art["image_status"], not as a safety flag
    if d["status"] == "POINT_ONLY_FALLBACK":
        flags.append("POINT_ONLY_FALLBACK")
    loc = art.get("localization") or {}
    prim = "image" if "image" in loc else next(iter(loc), None)
    verdict = loc.get(prim, {}).get("verdict", "UNKNOWN") if prim else "UNKNOWN"
    metrics = dict(sel.metrics) if sel else {}
    metrics.update({"runtime_s": _f(art.get("runtime_s")), "peak_mem_mb": _f(art.get("peak_mem_mb"))})
    if sel is not None and sel.M_final is not None:
        T_he_if = (sel.M_final, sel.t_final) if fixed == "IF" else _inv(sel.M_final, sel.t_final)
        p = _params(*T_he_if)
        metrics.update({"scale_he_to_if": p["scale"], "rotation_he_to_if_deg": p["rotation_deg"], "translation_he_to_if_um": [p["tx"], p["ty"]]})
    reasons = [d.get("reason", "")]
    if d.get("failure_stage"):
        reasons.append(f"失敗した段階: {d['failure_stage']} {d.get('failure_stage_text') or ''}")
    label = {"automatic": "", "assisted": "［注釈使用］"}.get(art.get("mode"), "")
    return ResultRecord(rid=rid, sample=art["sample"], method="WF", config=f"{label}{art['config']['localization']}_{fixed}fix@{art['run_id']}",
                        fixed=fixed, moving=moving, M=np.asarray(fw["M"]) if fw else None, t=np.asarray(fw["t"]) if fw else None,
                        verdict=verdict, margin=_f(loc.get(prim, {}).get("margin")) if prim else None, accepted=bool(d.get("accepted")),
                        status_flags=flags, failure_reasons=reasons, fallback_source=None, metrics=metrics, candidates=cands,
                        source="registration_core", files=list(files), artifact=art,
                        note=f"mode={art.get('mode')} selected={d.get('selected_candidate')}")


def _run_records(sample, src):
    out = []
    root = Path(src["runs_dir"])
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        art = d / "artifact.json"
        if art.is_file():
            try:
                a = json.loads(art.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if a.get("sample") == sample and a.get("schema", "").startswith("cellreg-registration-artifact/"):
                out.append(record_from_engine_artifact(a, f"wf:{d.name}", files=[str(art)]))
            continue
        res = d / "result.json"
        if not res.is_file():
            continue
        try:
            rec = json.loads(res.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if rec.get("sample") != sample:
            continue
        cfg = f"{rec.get('variant', 'X3')}_{rec['fixed']}fix@{d.name}"
        r = record_from_variant_json(rec, f"run:{d.name}", "explorer-runs", cfg, method="RUN", files=[str(res)])
        r.note += f" | run {d.name}"
        out.append(r)
    return out


def _stored_records(sample, src):
    M, t = np.asarray(src["M"], float), np.asarray(src["t"], float)
    return [ResultRecord(rid=f"stored:{sample}:{src['method']}", sample=sample, method="A2", config=src["label"], fixed=src["fixed"],
                         moving=src["moving"], M=M, t=t, verdict=src.get("verdict", "UNKNOWN"), margin=None, accepted=src.get("accepted"),
                         status_flags=[], failure_reasons=[src.get("note", "")], fallback_source=None,
                         metrics={"population": "not recorded; see UI-recomputed diagnostics"}, candidates=[], source="stored-transform",
                         files=[src.get("source_file", "")])]


def _manual_records(sample):
    from . import manual
    out = []
    for m in manual.list_saved(sample):
        try:
            fw = m["forward_transform"]
            M, t = np.asarray(fw["M"], float), np.asarray(fw["t"], float)
            if M.shape != (2, 2) or t.shape != (2,) or m["fixed"] == m["moving"]:
                raise ValueError("bad transform shape or direction")
        except (KeyError, TypeError, ValueError):
            continue   # malformed file: skipped rather than shown as a usable transform
        base = m.get("base") or {}
        adj = m.get("adjustment") or {}
        out.append(ResultRecord(
            rid=f"manual:{sample}:{m['id']}", sample=sample, method="MAN", config=f"{m.get('name') or m['id']}", fixed=m["fixed"],
            moving=m["moving"], M=M, t=t, verdict="MANUAL", margin=None, accepted=False,
            status_flags=[manual.STATUS], failure_reasons=["manual adjustment by the user; not verified by the prespecified acceptance rule"],
            fallback_source=None, metrics={"population": "not evaluated by the research pipeline; see UI-recomputed diagnostics"},
            candidates=[], source="manual", files=[m["_path"]], artifact=m,
            note=(f"base {base.get('label', '?')}; rot {adj.get('rotation_deg')}°, scale {adj.get('scale')}, "
                  f"d=({adj.get('dx')}, {adj.get('dy')}); by {m.get('created_by')} {m.get('created_at')}")))
    return out


def load_records(sample_info) -> list[ResultRecord]:
    out = []
    for src in sample_info.result_sources:
        kind = src.get("type")
        try:
            if kind == "oir-20261019":
                out += _oir_records(sample_info.sample_id, src)
            elif kind == "e0x-ablation-20261020":
                out += _ablation_records(sample_info.sample_id, src)
            elif kind == "explorer-runs":
                out += _run_records(sample_info.sample_id, src)
            elif kind == "stored-transform":
                out += _stored_records(sample_info.sample_id, src)
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            out.append(ResultRecord(rid=f"error:{kind}", sample=sample_info.sample_id, method="ERR", config=f"{kind} unreadable", fixed="HE",
                                    moving="IF", M=None, t=None, verdict="UNREADABLE", margin=None, accepted=False, status_flags=[],
                                    failure_reasons=[f"{type(exc).__name__}: {exc}"], fallback_source=None, metrics={}, candidates=[],
                                    source=str(kind), files=[]))
    out += _manual_records(sample_info.sample_id)
    return out


def find(records, rid):
    return next((r for r in records if r.rid == rid), None)


def oriented(M, t, rec_fixed: str, display_fixed: str):
    """Transform (moving->fixed) re-expressed for the requested display direction; returns (M, t, derived_flag)."""
    if not isinstance(rec_fixed, str) or not isinstance(display_fixed, str):
        raise TypeError("oriented() expects modality keys ('HE' / 'IF') for rec_fixed and display_fixed")
    if rec_fixed == display_fixed:
        return np.asarray(M, float), np.asarray(t, float), False
    Mi, ti = _inv(M, t)
    return Mi, ti, True


def params(M, t):
    return _params(M, t)
