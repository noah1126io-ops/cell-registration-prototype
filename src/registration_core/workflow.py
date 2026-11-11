"""Research Workflow C: staged registration (C1 localize -> C2 refine -> C3 validate -> C4 artifact).

One engine for every caller (Slurm runner, Registration Explorer, Workflow C research mode).  The search, ICP and
evaluation functions are the vendored research code (search.py / pipeline.py / a2.py, unchanged); this module only
orchestrates them and records every stage.

C1  Global region localization.  If both images are present AND usable, an image search (structural maps, no raw
    intensity comparison) is run first; the point search (E0x) is always run as well when points are available, and the
    two candidate lists are kept separately (image candidates never replace point candidates).  Without usable images
    the point search alone is used.  ``localization="a2"`` runs the legacy A2 instead.
C2  Candidate refinement: overlap-restricted ICP (cropped_icp) for every candidate of every source; the size of the move
    is recorded and a candidate that ICP moves to another region (> distinct_um) is flagged ``relocated``.
C3  Validation with the PRESPECIFIED rule (unchanged): a source is accepted only if its search verdict is UNIQUE and its
    top candidate passes the nuclear-geometry check (<=3 um >= 0.25 and >= 3x mirror null) in the fixed frame AND, when
    both frames are HE/IF, in the IF frame.  Two accepted sources that disagree -> CONFLICT (not accepted).
    Runs that used user annotations (constraints) are ASSISTED and are never accepted automatically.
    Adjacent sections (2026-11-10): when HE and IF are different (adjacent) sections their nuclei are different cells, so
    the nuclear rule cannot pass even at the true pose (analysis landmark-reference-failure-analysis-20261110).  The
    nuclear check is still recorded but not used; nuclear ICP is diagnostic only (the search pose is kept); the image
    top candidate gets an independent quadrant-consistency check instead (each quarter of the IF field re-localised on
    its own).  Passing gives TISSUE_LEVEL: region located at tissue level, accepted stays False (never exported as a
    cell-level registration).  Same-section pairs are unchanged.
C4  Artifact: candidates, decision, transforms, image status, inputs fingerprint, software version, timings.

Transform convention: fixed_world = M @ moving_world + t (frame-local world units, see frames.py).
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import time
import tracemalloc
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import a2 as A2
from . import metrics as mt
from .features import FeatureMaps, Modality, extract
from .frames import Affine2D, Frame, json_safe
from .pipeline import ACCEPT, TRUST, RunResult, Side, build_layers, cropped_icp, evaluate, footprint, pose_change
from .search import Canvas, Candidate, PoseSearch, SearchConfig, pose_matrix

SCHEMA = "cellreg-registration-artifact/1"
CORE_VERSION = "registration_core 1.2 (2026-11-10)"   # 1.1: occ+cleft image search; 1.2: adjacent-section C2/C3 (TISSUE_LEVEL)
REPO = Path(__file__).resolve().parents[2]

# what the UI shows instead of the internal method keys
SOURCE_LABELS = {
    "image": "両画像からの対応領域探索（組織・核の構造map）",
    "points": "視野サイズの違いに対応した核探索",
    "a2": "核密度による対応領域探索（従来法）",
    "init": "指定した初期位置の周辺探索（注釈を使用）",
}
VERDICT_TEXT = {
    "UNIQUE": "一意：最上位の候補が2位より十分に高い（margin ≥ 0.10）",
    "AMBIGUOUS": "曖昧：上位の候補どうしのスコア差が小さく、1つに決まらない（margin < 0.10）",
    "LOW_PEAK": "ピークが弱い：最上位でもスコアが基準（z）に届かない",
    "NO_CANDIDATE": "候補なし：重なりを持つ配置が見つからない",
    "LOCAL": "局所探索：指定した初期位置の周辺だけを調べたので、一意性は評価していない",
}
SECTION_RELATION_TEXT = {"same": "同一切片（核が同じ細胞に対応する）", "adjacent": "隣接切片（核は別の細胞。組織構造だけが対応する）",
                         "unknown": "不明（同一切片として扱う）"}
# prespecified 2026-11-10 (PRESPECIFIED_RULE.md of the study): >= 3 of 4 quadrants within 30 um and 2 deg of the candidate
QUADRANT_RULE = {"min_agree": 3, "max_shift_um": 30.0, "max_dtheta_deg": 2.0, "half_deg": 4.0, "step_deg": 1.0,
                 "scales": (0.97, 1.0, 1.03), "window_um": 100.0}


# ----------------------------------------------------------------------------- inputs
@dataclass
class SideInput:
    """One modality of a sample in its own frame-local world (um when pixel_um is known)."""
    key: str                         # "HE" | "IF" | other
    points: np.ndarray               # (n, 2) nuclei, frame world units
    extent: tuple | None             # imaged extent (w, h) in world units; None = UNKNOWN
    frame_name: str
    pixel_um: float | None = None
    image_path: str | None = None
    image_kind: str = "points_only"  # brightfield_rgb | fluorescence_display_rgb | points_only
    shape_px: tuple | None = None
    pyramid_downsamples: tuple = ()
    nuclei_source: str = ""

    def frame(self) -> Frame:
        mod = {"brightfield_rgb": "HE", "fluorescence_display_rgb": "IF"}.get(self.image_kind, self.key)
        return Frame(self.frame_name, mod, self.pixel_um, self.shape_px, image_path=self.image_path,
                     pyramid_downsamples=tuple(self.pyramid_downsamples))


@dataclass
class Constraints:
    """User annotations used to steer a run (makes the run ASSISTED; never used silently)."""
    exclude_poses: list = field(default_factory=list)      # [{"M": 2x2, "t": 2, "label": str}] fixed<-moving poses to drop
    search_roi: list | None = None                          # polygon in the fixed frame (search restricted to it)
    init: dict | None = None                                # {"M", "t"} initial transform (manual / landmarks)
    landmarks: list = field(default_factory=list)          # [{"fixed": [x,y], "moving": [x,y]}]
    annotation_ids: list = field(default_factory=list)     # provenance (research store ids)

    def used(self) -> bool:
        return bool(self.exclude_poses or self.search_roi or self.init or self.landmarks)


@dataclass
class WorkflowConfig:
    localization: str = "auto"       # auto (image first if usable, plus points) | points | image | a2
    image_feature: str = "occ+cleft"           # 2026-10-27 (was tissue+nuclear: IF tissue = all-channel max, nuclei-dominated)
    image_combine: str = "zsum"
    feature_res_um: float = 2.0
    n_candidates: int = 6
    search: dict = field(default_factory=dict)      # SearchConfig overrides
    run_points_with_images: bool = True             # keep the point search alongside the image search
    local_half_deg: float = 10.0
    local_step_deg: float = 1.0
    local_scales: tuple = (0.95, 0.975, 1.0, 1.025, 1.05)
    local_window_um: float = 150.0
    section_relation: str = "auto"   # auto (from the image file names) | same | adjacent | unknown


# ----------------------------------------------------------------------------- helpers
def fingerprint(side: SideInput) -> dict:
    h = hashlib.sha256(np.ascontiguousarray(np.asarray(side.points, np.float64)).tobytes()).hexdigest()
    img = None
    if side.image_path and Path(side.image_path).is_file():
        st = Path(side.image_path).stat()
        img = {"path": side.image_path, "size": st.st_size, "mtime": st.st_mtime}
    return {"points_sha256": h, "n_points": int(len(side.points)), "extent": list(side.extent) if side.extent else None,
            "pixel_um": side.pixel_um, "image": img, "nuclei_source": side.nuclei_source}


def software_version() -> dict:
    def git(*a):
        try:
            return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"core": CORE_VERSION, "git_head": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--", "src")),
            "python": platform.python_version(), "numpy": np.__version__}


def similarity_from_landmarks(fixed_xy, moving_xy):
    """Umeyama similarity (no reflection) moving -> fixed from >= 2 pairs; returns (M, t, residuals)."""
    P, Q = np.asarray(moving_xy, float), np.asarray(fixed_xy, float)
    if len(P) < 2:
        raise ValueError("at least 2 landmark pairs are needed")
    mp, mq = P.mean(0), Q.mean(0)
    X, Y = P - mp, Q - mq
    U, S, Vt = np.linalg.svd(Y.T @ X / len(P))
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt)) or 1.0])
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / max((X ** 2).sum() / len(P), 1e-12))
    M = s * R
    t = mq - M @ mp
    res = np.linalg.norm(P @ M.T + t - Q, axis=1)
    return M, t, res


def infer_section_relation(sample_id: str, fixed_in: SideInput, moving_in: SideInput, requested: str = "auto") -> dict:
    """Same vs adjacent section from the file names (e.g. sample 031-8_6, HE '031-8_6_HE_reference_031-7_XY06' -> 031-7 vs
    031-8 = adjacent; '031-6_2_HE_031-6_XY02' -> same).  An explicit setting always wins; unparseable -> unknown."""
    if requested != "auto":
        return {"value": requested, "source": "config", "text": SECTION_RELATION_TEXT.get(requested, requested)}
    import re
    m = re.match(r"(\d+-\d+)_", sample_id + "_")
    sections = {}
    for side in (fixed_in, moving_in):
        name = Path(side.image_path).name if side.image_path else ""
        h = re.search(r"_HE_(?:reference_)?(\d+-\d+)_", name)
        if h:
            sections[side.key] = h.group(1)
        elif m and name.startswith(sample_id):
            sections[side.key] = m.group(1)
    if len(sections) < 2:
        return {"value": "unknown", "source": "filename", "sections": sections, "text": SECTION_RELATION_TEXT["unknown"]}
    val = "same" if len(set(sections.values())) == 1 else "adjacent"
    return {"value": val, "source": "filename", "sections": sections, "text": SECTION_RELATION_TEXT[val]}


def _crop_maps(fm: FeatureMaps, x0, y0, w, h):
    r = fm.res_um
    sl = (slice(int(round(y0 / r)), int(round((y0 + h) / r))), slice(int(round(x0 / r)), int(round((x0 + w) / r))))
    return FeatureMaps(fm.frame, fm.modality, r, {k: v[sl] for k, v in fm.maps.items()}, fm.fov[sl], fm.quality, 0.0, fm.source_level)


def quadrant_consistency(M, t, fixed: Side, moving: Side, feature: str, scfg: SearchConfig, rule=QUADRANT_RULE):
    """Independent check of an image candidate: each quarter of the moving field is re-localised alone (local image search
    around the candidate, fixed side cropped to the neighbourhood).  A true region is reproduced by every quarter of the
    tissue; a wrong optimum of the full-field score is not.  Returns per-quadrant shifts and the prespecified pass flag."""
    t_start = time.perf_counter()
    M, t = np.asarray(M, float), np.asarray(t, float)
    me, fe = np.asarray(moving.extent, float), np.asarray(fixed.extent, float)
    qe = me / 2
    wcfg = WorkflowConfig(local_half_deg=rule["half_deg"], local_step_deg=rule["step_deg"], local_scales=tuple(rule["scales"]),
                          local_window_um=rule["window_um"])
    th_c = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    out = []
    for qi, o in enumerate(np.array([[0, 0], [qe[0], 0], [0, qe[1]], qe])):
        mq = Side(moving.key, moving.points, tuple(qe), _crop_maps(moving.feats, o[0], o[1], *qe))
        fp = (np.array([[0, 0], [qe[0], 0], [0, qe[1]], qe]) + o) @ M.T + t
        pad = rule["window_um"] + 0.5 * float(np.linalg.norm(qe)) + 20.0
        f0 = np.floor(np.clip(fp.min(0) - pad, 0, fe) / fixed.feats.res_um) * fixed.feats.res_um
        f1 = np.clip(fp.max(0) + pad, 0, fe)
        rec = {"quadrant": qi, "origin_moving": o.tolist()}
        if np.any(f1 - f0 < qe * 0.5):
            out.append({**rec, "ok": False, "note": "quadrant maps mostly outside the fixed field"}); continue
        fq = Side(fixed.key, fixed.points, tuple(f1 - f0), _crop_maps(fixed.feats, f0[0], f0[1], *(f1 - f0)))
        layers, _ = build_layers("E2", fq, mq, scfg.res_fine, feature)
        res = _local_search(layers, fq, mq, scfg, M, M @ o + t - f0, wcfg)
        top = res["top"]
        if top is None:
            out.append({**rec, "ok": False, "note": "no peak"}); continue
        Mq, tq = top.A(), top.t + f0 - top.A() @ o
        cq = o + qe / 2
        out.append({**rec, "ok": True, "shift_um": float(np.linalg.norm(Mq @ cq + tq - (M @ cq + t))),
                    "d_theta_deg": float((top.theta - th_c + 180) % 360 - 180), "scale_ratio": float(top.scale / math.sqrt(abs(np.linalg.det(M)))),
                    "z": float(top.score)})
    agree = [q for q in out if q.get("ok") and q["shift_um"] <= rule["max_shift_um"] and abs(q["d_theta_deg"]) <= rule["max_dtheta_deg"]]
    return {"quadrants": out, "n_agree": len(agree), "pass": len(agree) >= rule["min_agree"], "rule": json_safe(rule),
            "runtime_s": time.perf_counter() - t_start}


def _probe_distance(Ma, ta, Mb, tb, probe):
    return float(np.median(np.linalg.norm(probe @ np.asarray(Ma).T + ta - (probe @ np.asarray(Mb).T + tb), axis=1)))


def _poly_mask_points(points, poly):
    return mt.inside_poly(points, np.asarray(poly, float), 0.0) if poly is not None else np.ones(len(points), bool)


def image_status(fixed: SideInput, moving: SideInput, res_um=2.0):
    """'Image file exists' is not 'image usable'.  Returns (status dict, feats_fixed, feats_moving)."""
    st = {"flags": [], "reasons": [], "quality": {}}
    for s in (fixed, moving):
        if s.image_path and not Path(s.image_path).is_file():
            st["flags"].append("IMAGE_READ_FAILED")
            st["reasons"].append(f"{s.key}: image file not found / unreadable ({s.image_path})")
    have = [s for s in (fixed, moving) if s.image_path and Path(s.image_path).is_file()]
    if not have:
        st["flags"].append("NO_IMAGE"); st["reasons"].append("画像なし")
        return st, None, None
    st["flags"].append("IMAGE_AVAILABLE" if len(have) == 2 else "ONE_IMAGE_ONLY")
    feats = {}
    for s in have:
        if s.pixel_um is None or not s.pyramid_downsamples:
            st["reasons"].append(f"{s.key}: pixel spacing / pyramid UNKNOWN")
            continue
        try:
            fm = extract(Modality(s.frame(), np.asarray(s.points, float)), res_um=res_um)
        except Exception as exc:  # noqa: BLE001 - unreadable image -> point-only, recorded
            st["reasons"].append(f"{s.key}: image read failed ({type(exc).__name__}: {exc})")
            continue
        st["quality"][s.key] = fm.quality
        feats[s.key] = fm
        if not fm.quality.get("usable"):
            st["reasons"].append(f"{s.key}: " + "; ".join(fm.quality.get("reasons") or ["quality check failed"]))
    both = fixed.key in feats and moving.key in feats
    usable = both and all(feats[k].quality.get("usable") for k in (fixed.key, moving.key))
    if len(have) < 2:
        st["reasons"].append("片側の画像しかないため、画像どうしの照合は行わない（核点群で探索）")
    st["flags"].append("IMAGE_USABLE" if usable else ("IMAGE_INCOMPATIBLE" if len(have) == 2 else "IMAGE_PAIR_UNAVAILABLE"))
    st["usable_for_image_search"] = bool(usable)
    return st, feats.get(fixed.key), feats.get(moving.key)


# ----------------------------------------------------------------------------- C1 localization
def _search_record(out, source, labels_used, cfg_used):
    def cand(c: Candidate, rank):
        top = out["fine"][0].score if out["fine"] else c.score
        return {"source": source, "rank": rank, "chirality": c.chirality, "theta_deg": float(c.theta), "scale": float(c.scale),
                "t": [float(x) for x in c.t], "score": float(c.score), "layer_ncc": c.layer_ncc, "layer_z": c.layer_z,
                "relative_gap_to_top": (top - c.score) / abs(top) if top else None, "M_search": pose_matrix(c.chirality, c.theta, c.scale).tolist()}
    return {"source": source, "label": SOURCE_LABELS[source], "verdict": out["verdict"], "verdict_text": VERDICT_TEXT.get(out["verdict"], ""),
            "margin": out["margin"], "n_poses": out["n_poses"], "runtime_s": out["runtime_s"], "layers": labels_used, "config": cfg_used,
            "candidates": [cand(c, i) for i, c in enumerate(out["fine"])], "coarse_top": [c.as_dict() for c in out["coarse_top"][:10]]}


def _run_pose_search(layers, fixed: Side, moving: Side, cfg: SearchConfig):
    ps = PoseSearch(layers, fixed.extent, moving.extent, np.asarray(fixed.extent) / 2.0, np.asarray(moving.extent) / 2.0, cfg)
    return ps.run()


def _local_search(layers, fixed: Side, moving: Side, cfg: SearchConfig, init_M, init_t, wcfg: WorkflowConfig):
    """Fine-resolution search only around a given pose (manual / landmark initialisation)."""
    M = np.asarray(init_M, float)
    det = float(np.linalg.det(M))
    chir = "reflect" if det < 0 else "proper"
    C = np.diag([1.0, -1.0]) if chir == "reflect" else np.eye(2)
    sR = M @ C
    s0 = math.sqrt(abs(np.linalg.det(sR)))
    th0 = math.degrees(math.atan2(sR[1, 0], sR[0, 0]))
    ps = PoseSearch(layers, fixed.extent, moving.extent, np.asarray(fixed.extent) / 2.0, np.asarray(moving.extent) / 2.0, cfg)
    t_start = time.perf_counter()
    cv = Canvas(fixed.extent, moving.extent, ps.fc, cfg.res_fine, s0 * max(wcfg.local_scales) * 1.1)
    cache, cands = {}, []
    mc = ps.mc
    target = M @ mc + np.asarray(init_t, float)          # where the moving centre lands under the initial pose
    for th in np.arange(th0 - wcfg.local_half_deg, th0 + wcfg.local_half_deg + 1e-9, wcfg.local_step_deg):
        for f in wcfg.local_scales:
            maps, t0 = ps._maps(cv, chir, th, s0 * f, cache)
            if maps is None:
                continue
            t_init = target - pose_matrix(chir, th, s0 * f) @ mc
            pk = ps._peaks(cv, maps, t0, chir, th, s0 * f, 1, window=(t_init, wcfg.local_window_um), stage="local")
            cands += pk
    d = ps.distinct(cands)
    top = d[0] if d else None
    margin = (top.score - d[1].score) / abs(top.score) if (top and len(d) > 1 and top.score) else np.inf
    return {"top": top, "second": d[1] if len(d) > 1 else None, "margin": float(margin), "verdict": "LOCAL" if top else "NO_CANDIDATE",
            "fine": d, "coarse_top": [], "n_poses": ps.n_poses, "runtime_s": time.perf_counter() - t_start}


def _verdict(fine, cfg: SearchConfig, n_layers):
    """Same rule as PoseSearch.run (prespecified): NO_CANDIDATE / AMBIGUOUS / LOW_PEAK / UNIQUE."""
    top = fine[0] if fine else None
    second = fine[1] if len(fine) > 1 else None
    margin = (top.score - second.score) / abs(top.score) if (top and second and top.score != 0) else np.inf
    if top is None:
        return "NO_CANDIDATE", float(margin)
    if margin < cfg.ambiguous_margin:
        return "AMBIGUOUS", float(margin)
    if top.score < cfg.min_peak_z * (n_layers if cfg.combine in ("zsum", "rerank") else 1):
        return "LOW_PEAK", float(margin)
    return "UNIQUE", float(margin)


def _apply_exclusions(out, exclude, probe, cfg: SearchConfig, n_layers):
    """Drop candidates matching user-excluded poses, then re-apply the unchanged verdict rule to what remains."""
    if not exclude:
        return out, []
    keep, dropped = [], []
    for c in out["fine"]:
        hit = next((e for e in exclude if _probe_distance(c.A(), c.t, e["M"], e["t"], probe) < cfg.distinct_um), None)
        (dropped if hit else keep).append((c, hit))
    fine = [c for c, _ in keep]
    verdict, margin = _verdict(fine, cfg, n_layers)
    out = {**out, "fine": fine, "top": fine[0] if fine else None, "second": fine[1] if len(fine) > 1 else None, "margin": margin, "verdict": verdict}
    return out, [{"theta_deg": c.theta, "scale": c.scale, "t": c.t.tolist(), "excluded_by": (h or {}).get("label", "")} for c, h in dropped]


def localize(fixed_in: SideInput, moving_in: SideInput, cfg: WorkflowConfig, cons: Constraints | None, log):
    """C1: returns (searches dict, image status, feats)."""
    cons = cons or Constraints()
    searches, notes = {}, []
    st, ff, mf = image_status(fixed_in, moving_in, cfg.feature_res_um) if cfg.localization in ("auto", "image") else ({"flags": ["IMAGES_NOT_REQUESTED"], "reasons": [], "quality": {}, "usable_for_image_search": False}, None, None)
    if fixed_in.extent is None or moving_in.extent is None:
        if cfg.localization != "a2":
            raise ValueError("imaged extent (pixel spacing x image size) is UNKNOWN on one side; the partial-overlap search needs it "
                             "(use localization='a2' for point clouds without a physical frame)")
    fixed = Side(fixed_in.key, np.asarray(fixed_in.points, float), tuple(fixed_in.extent) if fixed_in.extent else None, ff)
    moving = Side(moving_in.key, np.asarray(moving_in.points, float), tuple(moving_in.extent) if moving_in.extent else None, mf)
    scfg = SearchConfig(**cfg.search)
    probe = (np.asarray(moving.extent) / 2.0 + np.array([[0, 0], [200, 0], [0, 200], [-200, 0], [0, -200]], float)) if moving.extent else moving.points[:: max(1, len(moving.points) // 50)]
    # ROI restriction of the fixed side (search only; validation always uses the full data)
    fixed_search = fixed
    if cons.search_roi:
        sel = _poly_mask_points(fixed.points, cons.search_roi)
        if sel.sum() < 20:
            raise ValueError(f"search ROI contains only {int(sel.sum())} fixed nuclei (need >= 20)")
        feats = fixed.feats
        if feats is not None:
            H, W = feats.fov.shape
            yy, xx = np.mgrid[0:H, 0:W]
            roi_px = mt.inside_poly(np.column_stack([xx.ravel(), yy.ravel()]) * feats.res_um, np.asarray(cons.search_roi, float), 0.0).reshape(H, W)
            feats = FeatureMaps(feats.frame, feats.modality, feats.res_um, feats.maps, feats.fov * roi_px.astype(np.float32), feats.quality, feats.runtime_s, feats.source_level)
        fixed_search = Side(fixed.key, fixed.points[sel], fixed.extent, feats)
        notes.append(f"search restricted to the ROI ({int(sel.sum())} / {len(fixed.points)} fixed nuclei)")
    init = None
    if cons.landmarks:
        M0, t0, res = similarity_from_landmarks([l["fixed"] for l in cons.landmarks], [l["moving"] for l in cons.landmarks])
        init = {"M": M0, "t": t0, "from": "landmarks", "landmark_residual_um": res.tolist()}
    if cons.init:
        init = {"M": np.asarray(cons.init["M"], float), "t": np.asarray(cons.init["t"], float), "from": cons.init.get("from", "manual")}
    if cfg.localization == "a2":
        log("C1: legacy A2 (point density, fixed-bbox canvas)")
        t0 = time.perf_counter()
        res = A2.method_a2(moving.points, fixed.points)
        det = A2.detector(res.M, res.t, moving.points, fixed.points, basin_margin=res.diagnostics.get("basin_margin", np.nan))
        cands = []
        for rank, (obj, M, t, hyp) in enumerate(res.refined[:cfg.n_candidates]):
            cands.append({"source": "a2", "rank": rank, "chirality": hyp["chirality"], "theta_deg": hyp["theta"], "scale": hyp["scale"],
                          "t": list(hyp["translation"]), "score": hyp.get("score"), "M_search": pose_matrix(hyp["chirality"], hyp["theta"], hyp["scale"]).tolist(),
                          "a2_refined": {"M": np.asarray(M).tolist(), "t": np.asarray(t).tolist(), "objective": obj}})
        verdict = {"PASS": "UNIQUE", "AMBIGUOUS": "AMBIGUOUS"}.get(det["detector_status"], "LOW_PEAK")
        searches["a2"] = {"source": "a2", "label": SOURCE_LABELS["a2"], "verdict": verdict, "verdict_text": VERDICT_TEXT.get(verdict, ""),
                          "margin": res.diagnostics.get("basin_margin"), "detector": det, "runtime_s": time.perf_counter() - t0,
                          "candidates": cands, "note": "A2 detector PASS -> UNIQUE (prespecified 2026-10-14)"}
        return searches, st, (fixed, moving), probe, init, notes
    if init is not None:
        log(f"C1: local search around the {init['from']} initial pose (ASSISTED)")
        layers, used = build_layers("E0x", fixed_search, moving, scfg.res_fine)
        if st.get("usable_for_image_search"):
            il, used_i = build_layers("E2", fixed_search, moving, scfg.res_fine, cfg.image_feature)
            layers, used = il + layers, used_i + used
        out = _local_search(layers, fixed_search, moving, SearchConfig(**{**cfg.search, "combine": cfg.image_combine}), init["M"], init["t"], cfg)
        searches["init"] = _search_record(out, "init", used or ["points"], {"init_from": init["from"], "half_deg": cfg.local_half_deg,
                                                                            "scales": list(cfg.local_scales), "window_um": cfg.local_window_um})
        searches["init"]["init"] = json_safe({k: v for k, v in init.items()})
        return searches, st, (fixed, moving), probe, init, notes
    if cfg.localization in ("auto", "image") and st.get("usable_for_image_search"):
        log("C1: image search (structural maps: " + cfg.image_feature + ")")
        layers, used = build_layers("E2", fixed_search, moving, scfg.res_fine, cfg.image_feature)
        icfg = SearchConfig(**{**cfg.search, "combine": cfg.image_combine})
        out = _run_pose_search(layers, fixed_search, moving, icfg)
        out, dropped = _apply_exclusions(out, cons.exclude_poses, probe, icfg, len(layers))
        searches["image"] = _search_record(out, "image", used, asdict(icfg))
        searches["image"]["excluded"] = dropped
    elif cfg.localization == "image":
        notes.append("image search requested but images are not usable: " + "; ".join(st.get("reasons", [])))
    if cfg.localization == "points" or (cfg.localization == "auto" and (cfg.run_points_with_images or "image" not in searches)):
        log("C1: point search (union canvas, nuclei density)")
        layers, used = build_layers("E0x", fixed_search, moving, scfg.res_fine)
        out = _run_pose_search(layers, fixed_search, moving, scfg)
        out, dropped = _apply_exclusions(out, cons.exclude_poses, probe, scfg, len(layers))
        searches["points"] = _search_record(out, "points", ["points"], asdict(scfg))
        searches["points"]["excluded"] = dropped
    return searches, st, (fixed, moving), probe, init, notes


# ----------------------------------------------------------------------------- C2 / C3
def geometric_eval(T: Affine2D, moving: Side, fixed: Side):
    """Fixed-frame nuclear geometry (same definition as the 2026-10-20 ablation)."""
    mw = T.apply(moving.points)
    mov_fp = T.apply(footprint(moving.extent))
    fix_mask = mt.inside_poly(fixed.points, mov_fp, 10.0) & mt.inside_poly(fixed.points, footprint(fixed.extent), 0.0)
    m_in = mt.inside_poly(mw, footprint(fixed.extent), -10.0)
    if fix_mask.sum() < 5 or m_in.sum() < 5:
        return {"n_eval": int(fix_mask.sum()), "le_3um": 0.0, "null_le_3um": 0.0, "inlier_over_null": 0.0, "nn_median_um": None,
                "geometric_pass": False, "note": "too few nuclei in the overlap"}
    pr = mt.proximity(mw[m_in], fixed.points, fix_mask)
    nl = mt.mirror_null(mw[m_in], fixed.points, fix_mask)
    ion = pr["le_3um"] / max(nl["le_3um"], 1e-6)
    return {"n_eval": int(fix_mask.sum()), **pr, "null_le_3um": nl["le_3um"], "inlier_over_null": ion,
            "geometric_pass": bool(pr["le_3um"] >= ACCEPT["min_inlier"] and ion >= ACCEPT["min_inlier_over_null"])}


def overlap_stats(T: Affine2D, moving: Side, fixed: Side):
    mov_fp = T.apply(footprint(moving.extent))
    fe = np.asarray(fixed.extent)
    res = 4.0
    xs, ys = np.arange(res / 2, fe[0], res), np.arange(res / 2, fe[1], res)
    g = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    area_mm2 = mt.inside_poly(g, mov_fp, 0.0).sum() * res * res / 1e6
    return {"overlap_area_mm2": float(area_mm2), "overlap_frac_of_moving": float(area_mm2 / (np.prod(moving.extent) / 1e6)),
            "n_fixed_in_overlap": int(mt.inside_poly(fixed.points, mov_fp, 0.0).sum()),
            "n_moving_in_overlap": int(mt.inside_poly(T.apply(moving.points), footprint(fixed.extent), 0.0).sum()),
            "overlap_polygon_fixed": mov_fp.tolist()}


def _if_frame_eval(T: Affine2D, T_search: Affine2D, fixed: Side, moving: Side, fixed_key, he_full, if_full, used):
    if {fixed.key, moving.key} != {"HE", "IF"}:
        return None
    he = fixed if fixed.key == "HE" else moving
    iff = fixed if fixed.key == "IF" else moving
    R = RunResult("", "", "", fixed.key, moving.key, T, T_search, image_features_used=used)
    return evaluate(R, Side("HE", he.points, he.extent, None), Side("IF", iff.points, iff.extent, None), he_full, if_full)


def refine_and_validate(searches, fixed: Side, moving: Side, cfg: WorkflowConfig, log, names, relation="same"):
    """C2 + C3 for every candidate of every source (relation='adjacent': nuclear ICP diagnostic only, quadrant check)."""
    fname, mname = names
    probe = footprint(moving.extent) if moving.extent else moving.points[::max(1, len(moving.points) // 50)]
    if src_needs := [k for k in searches if k != "a2" and (fixed.extent is None or moving.extent is None)]:
        raise ValueError(f"{src_needs}: imaged extents are needed")
    he_full = fixed.feats if fixed.key == "HE" else moving.feats
    if_full = fixed.feats if fixed.key == "IF" else moving.feats
    out = []
    for src, rec in searches.items():
        used = [l for l in rec.get("layers", []) if l != "points"]
        for c in rec["candidates"][:cfg.n_candidates]:
            cid = f"{src}#{c['rank']}"
            Ms = np.asarray(c["M_search"], float)
            ts = np.asarray(c["t"], float)
            Ts = Affine2D(Ms, ts, mname, fname)
            item = {"id": cid, "source": src, "source_label": SOURCE_LABELS[src], "rank": c["rank"], "search": c, "T_search": Ts.to_dict(),
                    "flags": []}
            if src == "a2":
                ref = c["a2_refined"]
                Tf = Affine2D(np.asarray(ref["M"]), np.asarray(ref["t"]), mname, fname)
                item["refine"] = {"method": "A2 production refine (all points)", "objective": ref["objective"],
                                  "pose_change": pose_change(Ts, Tf, probe), "ran": True}
            else:
                try:
                    ref = cropped_icp(Ms, ts, c["chirality"], moving.points, fixed.points, moving.extent, fixed.extent)
                except ValueError:
                    ref = None
                if ref is None:
                    Tf = Ts
                    item["refine"] = {"method": "overlap-restricted ICP", "ran": False, "note": "too few nuclei inside the other footprint"}
                    item["flags"].append("ICP_NOT_RUN")
                else:
                    T_icp = Affine2D(ref["M"], ref["t"], mname, fname)
                    ch = pose_change(Ts, T_icp, probe)
                    item["refine"] = {"method": "overlap-restricted ICP (similarity -> affine)", "ran": True, "pose_change": ch,
                                      "n_moving_used": ref["n_moving_used"], "n_fixed_used": ref["n_fixed_used"], "objective": ref["objective"]}
                    Tf = T_icp
                    if relation == "adjacent":
                        item["flags"].append("ICP_DIAGNOSTIC_ONLY")
                        item["refine"]["icp_not_used"] = "adjacent sections: nuclei are different cells, the search pose is kept (2026-11-10)"
                        Tf = Ts
                    elif ch["median_shift_um"] > SearchConfig().distinct_um:
                        item["flags"].append("RELOCATED_BY_ICP")
                    if relation != "adjacent" and src == "image" and (abs(ch["scale_change"]) > TRUST["max_scale_change"] or abs(ch["rot_change_deg"]) > TRUST["max_rot_change_deg"]
                                           or ch["median_shift_um"] > TRUST["max_shift_um"]):
                        item["flags"].append("ICP_LEFT_TRUST_REGION")
                        item["refine"]["icp_rejected"] = "ICP left the image-search trust region; the search pose is kept (2026-10-19 rule)"
                        Tf = Ts
            item["T_final"] = Tf.to_dict()
            if fixed.extent is None or moving.extent is None:     # no physical frame: only the A2 detector can judge
                det = searches["a2"].get("detector") if c["rank"] == 0 else None
                item["validation"] = {"a2_detector": det, "note": "imaged extent UNKNOWN: footprint-based checks skipped; "
                                                                  "A2's prespecified detector is used for the top candidate only"}
                item["geometric_pass"] = bool(det and det.get("detector_status") == "PASS")
                out.append(item)
                continue
            item["validation"] = {"search_pose_fixed_frame": geometric_eval(Ts, moving, fixed), "fixed_frame": geometric_eval(Tf, moving, fixed),
                                  "overlap": overlap_stats(Tf, moving, fixed)}
            ife = _if_frame_eval(Tf, Ts, fixed, moving, fixed.key, he_full, if_full, used)
            if ife is not None:
                item["validation"]["if_frame"] = ife
            v = item["validation"]
            item["geometric_pass"] = bool(v["fixed_frame"]["geometric_pass"] and (ife is None or ife.get("geometric_pass")))
            if relation == "adjacent":
                v["nuclear_check_applicable"] = False
                if src == "image" and c["rank"] == 0 and fixed.feats is not None and moving.feats is not None:
                    log(f"C3: {cid}: quadrant-consistency check (adjacent sections)")
                    v["quadrant_consistency"] = quadrant_consistency(Ts.M, Ts.t, fixed, moving, cfg.image_feature,
                                                                     SearchConfig(**{**cfg.search, "combine": cfg.image_combine}))
            out.append(item)
        log(f"C2/C3: {src}: {len(rec['candidates'][:cfg.n_candidates])} candidates refined and validated")
    return out


def decide(searches, cands, probe, mode, assisted, relation="same"):
    """C3 decision with the prespecified rule; returns decision dict."""
    if relation == "adjacent":
        return _decide_adjacent(searches, cands, mode, assisted)
    tol = SearchConfig().distinct_um
    per = {}
    for src, rec in searches.items():
        top = next((c for c in cands if c["source"] == src and c["rank"] == 0), None)
        ok_search = rec["verdict"] == "UNIQUE"
        ok_geom = bool(top and top["geometric_pass"])
        stage = None
        if top is None:
            stage = "C1"
        elif not ok_search:
            stage = "C1"
        elif "ICP_NOT_RUN" in top["flags"] or "RELOCATED_BY_ICP" in top["flags"]:
            stage = "C2" if not ok_geom else None
        elif not ok_geom:
            stage = "C3"
        per[src] = {"verdict": rec["verdict"], "verdict_text": rec.get("verdict_text"), "margin": rec.get("margin"), "top": top["id"] if top else None,
                    "search_ok": ok_search, "geometry_ok": ok_geom, "accepted_by_rule": bool(ok_search and ok_geom), "failure_stage": stage,
                    "search_pose_geometry_ok": bool(top and (top["validation"].get("search_pose_fixed_frame") or {}).get("geometric_pass"))}
    acc = [s for s, p in per.items() if p["accepted_by_rule"]]
    status, selected, reason = None, None, ""
    if len(acc) >= 2:
        a, b = [next(c for c in cands if c["id"] == per[s]["top"]) for s in acc[:2]]
        d = _probe_distance(a["T_final"]["M"], a["T_final"]["t"], b["T_final"]["M"], b["T_final"]["t"], probe)
        if d < tol:
            status, selected, reason = "ACCEPTED", per[acc[0]]["top"], f"{' と '.join(acc)} が同じ領域で採択基準を満たした（差 {d:.1f} µm）"
        else:
            status, selected, reason = "CONFLICT", per[acc[0]]["top"], f"{acc[0]} と {acc[1]} がそれぞれ基準を満たしたが、異なる領域を指している（差 {d:.0f} µm）"
    elif len(acc) == 1:
        s = acc[0]
        status, selected = ("ACCEPTED" if s in ("image", "a2") or "image" not in searches else "POINT_ONLY_FALLBACK"), per[s]["top"]
        reason = f"{SOURCE_LABELS[s]}が採択基準を満たした" + ("（画像探索は基準を満たさず、核探索の結果を採用）" if status == "POINT_ONLY_FALLBACK" else "")
    else:
        prim = "image" if "image" in searches else next(iter(searches), None)
        status = "NOT_ACCEPTED"
        selected = per[prim]["top"] if prim else None
        reason = "どの探索も採択基準を満たさない"
    accepted = status in ("ACCEPTED", "POINT_ONLY_FALLBACK") and not assisted
    if assisted:
        reason = "注釈（制約）を使った探索のため自動採択しない（要レビュー）。" + reason
        status = "ASSISTED_REVIEW_REQUIRED"
    # the stage reported is that of the primary search (image when it ran); the minimum over sources hid a C3 failure of
    # the image search behind the ambiguous point search (031-8_6, 2026-10-27)
    prim = "image" if "image" in per else next(iter(per), None)
    failure_stage = None if accepted else ("C3" if status == "CONFLICT" else (per[prim]["failure_stage"] if prim else None))
    return {"status": status, "accepted": bool(accepted), "selected_candidate": selected, "reason": reason, "mode": mode,
            "per_source": per, "failure_stage": failure_stage,
            "failure_stage_text": {"C1": "対応領域の探索（候補が一意に決まらない / ピークが弱い / 候補なし）",
                                   "C2": "精密化（ICPが実行できない、または別の領域へ大きく移動した）",
                                   "C3": "検証（核配置の整合が偶然一致と区別できない、または探索どうしが矛盾）"}.get(failure_stage),
            "rule": "prespecified 2026-10-19: search verdict UNIQUE AND <=3 um proximity >= 0.25 AND >= 3x mirror null "
                    "(fixed frame, and IF frame when HE/IF); unchanged"}


def _decide_adjacent(searches, cands, mode, assisted):
    """Adjacent sections (prespecified 2026-11-10): image search UNIQUE AND quadrant consistency -> TISSUE_LEVEL (accepted
    stays False).  The point search cannot localise different cells and is recorded only."""
    per = {}
    for src, rec in searches.items():
        top = next((c for c in cands if c["source"] == src and c["rank"] == 0), None)
        ok_search = rec["verdict"] == "UNIQUE"
        q = (top or {}).get("validation", {}).get("quadrant_consistency")
        ok_q = bool(q and q["pass"])
        stage = "C1" if (top is None or not ok_search) else (None if ok_q else "C3")
        per[src] = {"verdict": rec["verdict"], "verdict_text": rec.get("verdict_text"), "margin": rec.get("margin"), "top": top["id"] if top else None,
                    "search_ok": ok_search, "geometry_ok": False, "quadrant_ok": ok_q if src == "image" else None,
                    "accepted_by_rule": bool(src == "image" and ok_search and ok_q), "failure_stage": stage if src == "image" else None,
                    "note": None if src == "image" else "adjacent sections: the point search compares different cells (recorded only)"}
    img = per.get("image")
    selected = img["top"] if img else next((p["top"] for p in per.values()), None)
    if img and img["accepted_by_rule"]:
        status = "TISSUE_LEVEL"
        q = next(c for c in cands if c["id"] == img["top"])["validation"]["quadrant_consistency"]
        reason = (f"隣接切片：画像探索が一意で、IF視野の4象限中{q['n_agree']}象限が同じ位置を独立に再現した"
                  "（組織レベルで位置決定。核は別の細胞なので細胞レベルの採択はしない）")
    else:
        status = "NOT_ACCEPTED"
        reason = ("隣接切片：画像探索が使えない" if not img else
                  "隣接切片：画像探索が一意に決まらない" if not img["search_ok"] else "隣接切片：象限ごとの再探索が同じ位置を再現しない")
    if assisted:
        reason = "注釈（制約）を使った探索のため自動判定しない（要レビュー）。" + reason
        status = "ASSISTED_REVIEW_REQUIRED"
    fs = None if status == "TISSUE_LEVEL" else (img["failure_stage"] if img else "C1")
    return {"status": status, "accepted": False, "accepted_tissue_level": status == "TISSUE_LEVEL", "selected_candidate": selected,
            "reason": reason, "mode": mode, "per_source": per, "failure_stage": fs,
            "failure_stage_text": {"C1": "対応領域の探索（候補が一意に決まらない / ピークが弱い / 候補なし）",
                                   "C3": "検証（象限ごとの再探索が同じ位置を再現しない）"}.get(fs),
            "precision_note": "組織レベル（隣接切片。8_6のランドマーク精度は LOO 中央値 約24 µm）" if status == "TISSUE_LEVEL" else None,
            "rule": "prespecified 2026-11-10 (adjacent sections): image search verdict UNIQUE AND quadrant consistency "
                    ">= 3/4 quadrants within 30 um and 2 deg -> TISSUE_LEVEL (accepted=False); nuclear rule recorded, not applicable"}


# ----------------------------------------------------------------------------- C4 entry point
def run_workflow(sample_id: str, fixed_in: SideInput, moving_in: SideInput, cfg: WorkflowConfig | None = None,
                 constraints: Constraints | None = None, log=print, run_id: str | None = None) -> dict:
    cfg = cfg or WorkflowConfig()
    cons = constraints or Constraints()
    assisted = cons.used()
    mode = "assisted" if assisted else "automatic"
    run_id = run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    tracemalloc.start(); t0 = time.perf_counter()
    fname, mname = f"{sample_id}:{fixed_in.key}", f"{sample_id}:{moving_in.key}"
    relation = infer_section_relation(sample_id, fixed_in, moving_in, cfg.section_relation)
    log(f"section relation: {relation['value']} ({relation['source']})")
    searches, st, (fixed, moving), probe, init, notes = localize(fixed_in, moving_in, cfg, cons, log)
    cands = refine_and_validate(searches, fixed, moving, cfg, log, (fname, mname), relation["value"])
    if init is not None and cons.landmarks:
        for c in cands:
            Tf = Affine2D.from_dict(c["T_final"])
            r = np.linalg.norm(Tf.apply(np.asarray([l["moving"] for l in cons.landmarks], float)) - np.asarray([l["fixed"] for l in cons.landmarks], float), axis=1)
            c["validation"]["landmark_residual_um"] = {"median": float(np.median(r)), "max": float(r.max()), "n": int(len(r))}
    probe_d = footprint(moving.extent) if moving.extent else moving.points[::max(1, len(moving.points) // 50)]
    dec = decide(searches, cands, probe_d, mode, assisted, relation["value"])
    sel = next((c for c in cands if c["id"] == dec["selected_candidate"]), None)
    T = Affine2D.from_dict(sel["T_final"]) if sel else None
    art = {
        "schema": SCHEMA, "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(), "software": software_version(),
        "sample": sample_id, "mode": mode,
        "input": {"fixed": {"key": fixed_in.key, "frame": fname, **fingerprint(fixed_in)}, "moving": {"key": moving_in.key, "frame": mname, **fingerprint(moving_in)}},
        "frames": {"fixed": fixed_in.frame().to_dict(), "moving": moving_in.frame().to_dict(),
                   "convention": "fixed_world = M @ moving_world + t; frame-local world (um when pixel spacing is known); image resampling uses the inverse"},
        "section_relation": relation, "config": json_safe(asdict(cfg)), "constraints": json_safe(asdict(cons)), "image_status": json_safe(st), "notes": notes,
        "localization": {k: {kk: vv for kk, vv in v.items() if kk != "candidates"} | {"n_candidates": len(v["candidates"])} for k, v in searches.items()},
        "candidates": cands, "decision": dec,
        "transform": ({"forward": T.to_dict(), "inverse": T.inverse().to_dict(), "candidate": sel["id"],
                       "accepted": dec["accepted"], "direction": f"{moving_in.key} world -> {fixed_in.key} world"} if T else None),
        "runtime_s": time.perf_counter() - t0, "peak_mem_mb": tracemalloc.get_traced_memory()[1] / 2 ** 20,
    }
    tracemalloc.stop()
    return json_safe(art)


def write_artifact(art: dict, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(art, indent=1, ensure_ascii=False))
    tmp.replace(path)
    return path
