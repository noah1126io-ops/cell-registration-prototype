"""Cell-correspondence review -> point deformation -> H&E image warp (research preview only, 2026-11-17).

Data flow (direction fixed: H&E = moving, fluorescence (IF) = fixed, as in Workflow D ``he_to_if``):

  HE nuclei (HE frame) --global affine M,t (from the Explorer Viewer / candidates / manual)--> IF frame
     -> ROI selection (IF frame)  -> frozen B0/B1/B3 + B2 (cell_correspondence.run_pipeline, unchanged)
     -> AutoResult  (immutable: written once, sha256 recorded)
     -> Review      (separate append-only edit log: support / exclude / reassign / add / unset, undo / redo)
     -> anchors (auto = B2 matched;  reviewed = auto + edits)
     -> local field (linear RBF of UOT conservative, deformation_experiment._rbf_from_anchors; D = validation_lib.run_uot)
        every field through the unchanged validation_lib.common_audit + MIN_ANCHORS (+ an anchor-coverage gate)
     -> point output  x_IF = A(x_HE) + d(A(x_HE))   (d sampled bilinearly from the grid, same sampler for the image;
                      outside the grid d fades linearly to 0 over TAPER_UM)
     -> image output  target pixel p (IF frame) -> q = d-inverse(p) (fixed point, residual-checked) -> x_HE = A^-1(q)
                      -> HE pyramid level, only the tiles under the ROI are decoded.

Nothing here feeds Workflow D.  A reviewed pair is a human judgement, not proof of biological identity.
"""
from __future__ import annotations

import datetime as _dt
import getpass
import hashlib
import io
import json
import math
import os
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from src.registration_evaluation import sample_displacement_field

from . import cell_correspondence as CC
from . import correspondence_anchors as CAn
from . import deformation_experiment as DE
from .legacy import REPO, frozen_b2, modules

SCHEMA_AUTO = "cellreg-cell-review-auto/1"
SCHEMA_REVIEW = "cellreg-cell-review-edits/1"
DIRECTION = "HE->IF"                    # moving = HE, fixed = IF
ROI_MARGIN_UM = 30.0                    # IF nuclei this far outside the ROI stay candidates for HE nuclei near the border
TOP_K = 5                               # candidates shown per HE nucleus
NEIGH_UM = 30.0                         # "surrounding nuclei" radius for the per-nucleus context
GRID_UM = 4.0                           # field grid spacing (same as the 11-13 study)
MIN_ANCHOR_COVERAGE = 0.20              # NEW research gate (stricter only): anchors must touch >= 20 % of occupied 100 um cells
INVERSE_TOL_UM = 0.05                   # |q + d(q) - p| for the image inverse map
INVERSE_MAX_ITER = 60
TAPER_UM = 50.0                         # outside the field grid the border displacement fades linearly to 0 over this distance
FIELD_METHODS = ("A", "AUTO", "REVIEWED", "D")
FIELD_LABELS = {"A": "Affineのみ", "AUTO": "自動推定の核対応（B2）＋linear RBF", "REVIEWED": "手動修正後の核対応＋linear RBF",
                "D": "UOT conservative（既存）"}
OPS = ("support", "exclude", "reassign", "add", "unset")
OP_LABELS = {"support": "支持", "exclude": "除外", "reassign": "対応先を変更", "add": "対応ペアを追加", "unset": "未確定にする"}
DEFAULT_ROOT = REPO / "runs" / "cell_review"      # runs/ is git-ignored; research.sqlite is never touched


class ReviewError(ValueError):
    """Invalid edit (unknown ID, wrong state, ...)."""


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def git_info() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def map_points(M, t, p):
    return np.asarray(p, float).reshape(-1, 2) @ np.asarray(M, float).T + np.asarray(t, float)


def inverse_affine(M, t):
    Mi = np.linalg.inv(np.asarray(M, float))
    return Mi, -Mi @ np.asarray(t, float)


def to_he_to_if(M, t, fixed_key: str):
    """The Viewer stores moving->fixed for the displayed direction; return HE->IF."""
    if fixed_key == "IF":
        return np.asarray(M, float), np.asarray(t, float)
    return inverse_affine(M, t)


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=_json_default).encode()).hexdigest()


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def points_hash(p) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.round(np.asarray(p, float), 4)).tobytes()).hexdigest()[:16]


# ----------------------------------------------------------------------------- Phase 2: automatic correspondence
def default_roi(he_mapped, if_pts, if_extent=None):
    """Bounding box (IF frame) of the overlap of the mapped HE nuclei and the IF nuclei / IF field."""
    lo = np.maximum(he_mapped.min(0), if_pts.min(0))
    hi = np.minimum(he_mapped.max(0), if_pts.max(0))
    if if_extent is not None:
        lo, hi = np.maximum(lo, 0.0), np.minimum(hi, np.asarray(if_extent, float))
    if np.any(hi <= lo):
        raise ReviewError("HEとIFの核が重なる範囲がありません（Global変換を確認してください）")
    return [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]


def _inside(p, roi, margin=0.0):
    x0, y0, x1, y1 = roi
    return (p[:, 0] >= x0 - margin) & (p[:, 0] <= x1 + margin) & (p[:, 1] >= y0 - margin) & (p[:, 1] <= y1 + margin)


@dataclass
class AutoResult:
    meta: dict
    table: pd.DataFrame            # one row per HE nucleus in the ROI
    candidates: pd.DataFrame       # long format: he_index, rank, if_index, ...
    he_idx: np.ndarray             # global HE indices evaluated (moving subset)
    if_idx: np.ndarray             # global IF indices offered as candidates (fixed subset)

    @property
    def auto_id(self) -> str:
        return self.meta["auto_id"]

    def auto_pairs(self) -> dict:
        """{he_index: if_index} for B2 'matched' rows (the automatic anchors)."""
        m = self.table["status"].to_numpy(str) == "matched"
        return {int(h): int(f) for h, f in zip(self.table["he_index"].to_numpy(int)[m], self.table["top1_if_index"].to_numpy(int)[m])}


def compute_auto(he_pts, he_ids, if_pts, if_ids, M, t, *, roi=None, if_extent=None, margin_um=ROI_MARGIN_UM, top_k=TOP_K,
                 meta=None) -> AutoResult:
    """Frozen Graph/UOT/B2 correspondence for the HE nuclei that the global transform puts inside the ROI (IF frame)."""
    he_pts, if_pts = np.asarray(he_pts, float), np.asarray(if_pts, float)
    he_ids, if_ids = np.asarray(he_ids), np.asarray(if_ids)
    if len(he_ids) != len(he_pts) or len(if_ids) != len(if_pts):
        raise ReviewError("核IDと座標の数が一致しません")
    M, t = np.asarray(M, float), np.asarray(t, float)
    mapped = map_points(M, t, he_pts)
    roi = [float(v) for v in (roi if roi is not None else default_roi(mapped, if_pts, if_extent))]
    he_sel = np.flatnonzero(_inside(mapped, roi))
    if_sel = np.flatnonzero(_inside(if_pts, roi, margin_um))
    if len(he_sel) < CC.MIN_POINTS or len(if_sel) < CC.MIN_POINTS:
        raise ReviewError(f"ROI内の核が少なすぎます（HE {len(he_sel)}、IF {len(if_sel)}。どちらも{CC.MIN_POINTS}個以上が必要）")
    fixed, moving = if_pts[if_sel], mapped[he_sel]
    res = CC.run_pipeline(fixed, moving, None, 0, "cell-review")
    E, _, _ = modules()
    b2 = res["frames"]["B2"]
    ctx = res["context"]
    b1 = res["runs"][E.METHOD_B1]
    ranked = np.asarray(b1.result.ranked_indices, int)[:, :top_k]
    plan = np.asarray(b1.plan, float)
    row_mass = plan.sum(1)
    cand_idx = np.asarray(ctx.candidates.indices, int)
    cost_pg = np.asarray(E.selected_cost(ctx, ("position", "graph")), float)
    gsup = np.asarray(ctx.graph_support, float)
    th = frozen_b2()
    status = b2["status"].to_numpy(str)
    pm, pu = b2["calibrated_confidence"].to_numpy(float), b2["predicted_unmatched_probability"].to_numpy(float)
    top1_local = b2["top1_fixed_index"].to_numpy(int)
    rows, crows = [], []
    tree_if = cKDTree(fixed)
    tree_he = cKDTree(moving)
    for k in range(len(he_sel)):
        g_he = int(he_sel[k])
        if status[k] == "matched":
            why = f"matched: P(match)={pm[k]:.3f} ≥ {th['match_threshold']:.4f}（B2、合成データで学習した凍結モデル）"
        elif status[k] == "unmatched":
            why = f"unmatched: P(match)={pm[k]:.3f} < {th['match_threshold']:.4f} かつ P(unmatched)={pu[k]:.3f} ≥ {th['unmatched_threshold']:.4f}"
        else:
            why = f"ambiguous: P(match)={pm[k]:.3f} < {th['match_threshold']:.4f} かつ P(unmatched)={pu[k]:.3f} < {th['unmatched_threshold']:.4f}"
        f1 = int(top1_local[k])
        rows.append({
            "he_index": g_he, "he_id": he_ids[g_he].item() if hasattr(he_ids[g_he], "item") else he_ids[g_he],
            "he_x": he_pts[g_he, 0], "he_y": he_pts[g_he, 1], "mapped_x": moving[k, 0], "mapped_y": moving[k, 1],
            "status": status[k], "b2_match_prob": pm[k], "b2_unmatched_prob": pu[k],
            "top1_if_index": int(if_sel[f1]), "top1_if_id": if_ids[if_sel[f1]].item() if hasattr(if_ids[if_sel[f1]], "item") else if_ids[if_sel[f1]],
            "top1_if_x": fixed[f1, 0], "top1_if_y": fixed[f1, 1], "top1_distance_um": float(np.linalg.norm(fixed[f1] - moving[k])),
            "top1_graph_support": float(b2["top1_graph_support"].iloc[k]), "top1_mass_share": float(b2["normalized_top1_mass"].iloc[k]),
            "n_he_within_30um": int(len(tree_he.query_ball_point(moving[k], NEIGH_UM)) - 1),
            "n_if_within_30um": int(len(tree_if.query_ball_point(moving[k], NEIGH_UM))),
            "adoption_reason": why,
        })
        for r, f in enumerate(ranked[k]):
            slot = np.flatnonzero(cand_idx[k] == f)
            s = int(slot[0]) if len(slot) else -1
            crows.append({"he_index": g_he, "rank": r + 1, "if_index": int(if_sel[f]),
                          "if_id": if_ids[if_sel[f]].item() if hasattr(if_ids[if_sel[f]], "item") else if_ids[if_sel[f]],
                          "if_x": fixed[f, 0], "if_y": fixed[f, 1], "distance_um": float(np.linalg.norm(fixed[f] - moving[k])),
                          "mass_share": float(plan[k, f] / row_mass[k]) if row_mass[k] > 0 else 0.0,
                          "graph_support": float(gsup[k, s]) if s >= 0 else float("nan"),
                          "position_graph_cost": float(cost_pg[k, s]) if s >= 0 else float("nan")})
    table = pd.DataFrame(rows)
    table["if_conflict_n"] = 0
    mt = table["status"] == "matched"
    cnt = table.loc[mt, "top1_if_index"].value_counts()
    table.loc[mt, "if_conflict_n"] = table.loc[mt, "top1_if_index"].map(cnt).fillna(0).astype(int) - 1
    cands = pd.DataFrame(crows)
    u = res["uot"]["B1"]
    meta = {
        "schema": SCHEMA_AUTO, "auto_id": None, "created_at": now_iso(), "direction": DIRECTION,
        "global": {"M": M.tolist(), "t": t.tolist()}, "roi_if_frame": roi, "roi_margin_um": margin_um, "top_k": top_k,
        "n_he_roi": int(len(he_sel)), "n_if_candidates": int(len(if_sel)),
        "status_counts": table["status"].value_counts().to_dict(),
        "conflicting_auto_pairs": int((table["if_conflict_n"] > 0).sum()),
        "method": "B2 = B1 (relative-cost Graph-UOT) + frozen adoption models (2026-10-14, synthetic Train fit, Validation thresholds)",
        "b2_source": th["source"], "b2_match_threshold": th["match_threshold"], "b2_unmatched_threshold": th["unmatched_threshold"],
        "score_definitions": {
            "b2_match_prob": "凍結B2の一致モデルの出力（合成データで学習。実データでは較正されていない。同一細胞の確率ではない）",
            "b2_unmatched_prob": "凍結B2の未対応モデルの出力（同上）",
            "mass_share": "B1 UOTの輸送計画で、その行の輸送量のうちこの候補に割り当てられた割合（同一細胞の確率ではない）",
            "graph_support": "近傍核の相対配置の一致度（2026-10-08の graph support）",
            "position_graph_cost": "位置＋graphのコスト（小さいほど良い）",
            "rank": "B1 UOTの輸送量の順位"},
        "uot_diagnostics": {"converged": bool(u["solver_converged"]), "iterations": u["solver_iterations"],
                            "mass_ratio_to_source": u["total_mass_ratio_to_source"], "row_entropy_median": u["normalized_row_entropy_median"]},
        "runtime_s": res["runtime_s"], "code": git_info(),
        "inputs": {"he_points_sha": points_hash(he_pts), "if_points_sha": points_hash(if_pts), "n_he": int(len(he_pts)), "n_if": int(len(if_pts))},
        **(meta or {}),
    }
    meta["auto_id"] = "auto-" + _sha({k: meta[k] for k in ("global", "roi_if_frame", "inputs", "created_at")})[:12]
    return AutoResult(meta, table, cands, he_sel, if_sel)


# ----------------------------------------------------------------------------- persistence (files, never research.sqlite)
def sample_dir(sample_id: str, root: Path | None = None) -> Path:
    return Path(root or os.environ.get("CELLREG_CELL_REVIEW_ROOT") or DEFAULT_ROOT) / sample_id


def save_auto(auto: AutoResult, sample_id: str, root: Path | None = None) -> Path:
    """Write once.  An existing auto result is never overwritten."""
    d = sample_dir(sample_id, root) / "auto" / auto.auto_id
    if d.exists():
        raise FileExistsError(f"{d} already exists (auto results are immutable)")
    d.mkdir(parents=True)
    auto.table.to_csv(d / "auto_correspondence.csv", index=False)
    auto.candidates.to_csv(d / "auto_candidates.csv", index=False)
    np.savez(d / "subsets.npz", he_idx=auto.he_idx, if_idx=auto.if_idx)
    meta = dict(auto.meta)
    meta["files_sha256"] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(d.iterdir())}
    (d / "auto_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False, default=_json_default))
    auto.meta = meta
    return d


def load_auto(path: Path) -> AutoResult:
    path = Path(path)
    meta = json.loads((path / "auto_meta.json").read_text())
    for name, h in meta.get("files_sha256", {}).items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != h:
            raise ReviewError(f"{path / name} は保存後に変更されています（sha256 不一致）")
    z = np.load(path / "subsets.npz")
    return AutoResult(meta, pd.read_csv(path / "auto_correspondence.csv"), pd.read_csv(path / "auto_candidates.csv"), z["he_idx"], z["if_idx"])


def list_autos(sample_id: str, root: Path | None = None) -> list[Path]:
    d = sample_dir(sample_id, root) / "auto"
    return sorted((p for p in d.glob("auto-*") if (p / "auto_meta.json").is_file()), key=lambda p: p.stat().st_mtime, reverse=True)


# ----------------------------------------------------------------------------- Phase 3: review (separate from the auto result)
@dataclass
class Review:
    """Edits on top of an AutoResult.  ops[:cursor] are in effect; history is append-only (every action incl. undo/redo)."""
    auto_id: str
    auto_sha: str
    n_he: int
    n_if: int
    auto_pairs: dict                       # {he_index: if_index} (frozen copy of the automatic anchors)
    review_id: str = field(default_factory=lambda: "rev-" + uuid.uuid4().hex[:10])
    created_at: str = field(default_factory=now_iso)
    ops: list = field(default_factory=list)
    cursor: int = 0
    history: list = field(default_factory=list)
    version: int = 0

    @classmethod
    def start(cls, auto: AutoResult):
        return cls(auto.auto_id, _sha(auto.meta.get("files_sha256") or auto.meta), int(auto.meta["inputs"]["n_he"]),
                   int(auto.meta["inputs"]["n_if"]), auto.auto_pairs())

    # -- state
    def state(self) -> dict:
        """{he_index: {"if_index": int | None, "decision": str}} for every HE nucleus touched (auto pairs + edits)."""
        s = {h: {"if_index": f, "decision": "auto"} for h, f in self.auto_pairs.items()}
        for op in self.ops[:self.cursor]:
            h = op["he_index"]
            cur = s.get(h, {"if_index": None, "decision": "none"})
            if op["op"] == "support":
                s[h] = {"if_index": cur["if_index"], "decision": "supported"}
            elif op["op"] == "exclude":
                s[h] = {"if_index": None, "decision": "excluded"}
            elif op["op"] == "unset":
                s[h] = {"if_index": None, "decision": "undecided"}
            elif op["op"] in ("reassign", "add"):
                s[h] = {"if_index": op["if_index"], "decision": "reassigned" if op["op"] == "reassign" else "added"}
        return s

    def pairs(self) -> dict:
        return {h: v["if_index"] for h, v in self.state().items() if v["if_index"] is not None}

    def conflicts(self) -> dict:
        """{if_index: [he_index, ...]} for IF nuclei assigned to more than one HE nucleus."""
        inv = {}
        for h, f in self.pairs().items():
            inv.setdefault(f, []).append(h)
        return {f: sorted(hs) for f, hs in inv.items() if len(hs) > 1}

    # -- edits
    def _check_ids(self, he_index, if_index=None):
        if not (isinstance(he_index, (int, np.integer)) and 0 <= int(he_index) < self.n_he):
            raise ReviewError(f"不正なHE核インデックス: {he_index}")
        if if_index is not None and not (isinstance(if_index, (int, np.integer)) and 0 <= int(if_index) < self.n_if):
            raise ReviewError(f"不正なIF核インデックス: {if_index}")

    def apply(self, op: str, he_index: int, if_index: int | None = None, *, user: str | None = None, reason: str = "") -> dict:
        if op not in OPS:
            raise ReviewError(f"不明な操作: {op}")
        self._check_ids(he_index, if_index)
        he_index = int(he_index)
        cur = self.state().get(he_index, {"if_index": None, "decision": "none"})
        if op == "support" and cur["if_index"] is None:
            raise ReviewError("支持する対応がありません（この核には現在の対応先がない）。『対応ペアを追加』を使ってください")
        if op in ("reassign", "add"):
            if if_index is None:
                raise ReviewError("対応先のIF核を指定してください")
            if_index = int(if_index)
            if cur["if_index"] == if_index:
                raise ReviewError("現在の対応先と同じです")
            if op == "add" and cur["if_index"] is not None:
                op = "reassign"                                   # the nucleus already has a partner: this is a change
            if op == "reassign" and cur["if_index"] is None:
                op = "add"
        else:
            if_index = None
        if op in ("exclude", "unset") and cur["if_index"] is None and cur["decision"] in ("excluded", "undecided"):
            raise ReviewError("すでに対応先がありません")
        rec = {"seq": len(self.history) + 1, "op": op, "he_index": he_index, "if_index": if_index, "before": cur,
               "auto": {"if_index": self.auto_pairs.get(he_index)}, "user": user or getpass.getuser(), "at": now_iso(),
               "reason": reason or ""}
        dropped = self.ops[self.cursor:]
        self.ops = self.ops[:self.cursor] + [rec]
        self.cursor = len(self.ops)
        self.history.append({**rec, "action": "apply", "discarded_redo": [d["seq"] for d in dropped]})
        return rec

    def undo(self, user=None):
        if self.cursor == 0:
            raise ReviewError("取り消す変更がありません")
        self.cursor -= 1
        self.history.append({"seq": len(self.history) + 1, "action": "undo", "target_seq": self.ops[self.cursor]["seq"],
                             "user": user or getpass.getuser(), "at": now_iso()})

    def redo(self, user=None):
        if self.cursor >= len(self.ops):
            raise ReviewError("やり直す変更がありません")
        self.history.append({"seq": len(self.history) + 1, "action": "redo", "target_seq": self.ops[self.cursor]["seq"],
                             "user": user or getpass.getuser(), "at": now_iso()})
        self.cursor += 1

    # -- persistence: each save is a new version file; earlier versions are kept
    def to_json(self) -> dict:
        return {"schema": SCHEMA_REVIEW, "review_id": self.review_id, "auto_id": self.auto_id, "auto_sha": self.auto_sha,
                "created_at": self.created_at, "saved_at": now_iso(), "version": self.version, "n_he": self.n_he, "n_if": self.n_if,
                "auto_pairs": {str(k): v for k, v in self.auto_pairs.items()}, "ops": self.ops, "cursor": self.cursor,
                "history": self.history, "effective_pairs_n": len(self.pairs()), "conflicts": {str(k): v for k, v in self.conflicts().items()},
                "note": "手動の採択は生物学的に同じ細胞であることの証明ではない。自動推定の結果（auto_id）は変更していない。"}

    @classmethod
    def from_json(cls, d: dict):
        if d.get("schema") != SCHEMA_REVIEW:
            raise ReviewError("レビューファイルの形式が違います")
        return cls(d["auto_id"], d["auto_sha"], int(d["n_he"]), int(d["n_if"]), {int(k): int(v) for k, v in d["auto_pairs"].items()},
                   review_id=d["review_id"], created_at=d["created_at"], ops=d["ops"], cursor=int(d["cursor"]), history=d["history"],
                   version=int(d["version"]))

    def save(self, sample_id: str, root: Path | None = None) -> Path:
        d = sample_dir(sample_id, root) / "reviews" / self.review_id
        d.mkdir(parents=True, exist_ok=True)
        self.version += 1
        p = d / f"v{self.version:04d}.json"
        if p.exists():
            raise FileExistsError(p)
        p.write_text(json.dumps(self.to_json(), indent=1, ensure_ascii=False, default=_json_default))
        return p


def load_review(path: Path) -> Review:
    return Review.from_json(json.loads(Path(path).read_text()))


def list_reviews(sample_id: str, auto_id: str, root: Path | None = None) -> list[Path]:
    out = []
    for d in (sample_dir(sample_id, root) / "reviews").glob("rev-*"):
        vs = sorted(d.glob("v*.json"))
        if vs and json.loads(vs[-1].read_text()).get("auto_id") == auto_id:
            out.append(vs[-1])
    return sorted(out, key=lambda p: p.stat().st_mtime, reverse=True)


def pairs_frame(pairs: dict, he_ids, if_ids, he_mapped, if_pts, state=None) -> pd.DataFrame:
    rows = []
    for h, f in sorted(pairs.items()):
        rows.append({"he_index": h, "he_id": _scalar(he_ids[h]), "if_index": f, "if_id": _scalar(if_ids[f]),
                     "mapped_x": he_mapped[h, 0], "mapped_y": he_mapped[h, 1], "if_x": if_pts[f, 0], "if_y": if_pts[f, 1],
                     "distance_um": float(np.linalg.norm(if_pts[f] - he_mapped[h])),
                     "decision": (state or {}).get(h, {}).get("decision", "auto")})
    return pd.DataFrame(rows, columns=["he_index", "he_id", "if_index", "if_id", "mapped_x", "mapped_y", "if_x", "if_y", "distance_um", "decision"])


def _scalar(v):
    return v.item() if hasattr(v, "item") else v


# ----------------------------------------------------------------------------- Phase 4: local deformation
def anchors_from_pairs(name, pairs: dict, he_mapped, if_pts, exclude_conflicts=False) -> CAn.Anchors:
    items = sorted(pairs.items())
    if exclude_conflicts:
        cnt = pd.Series([f for _, f in items]).value_counts() if items else pd.Series(dtype=int)
        items = [(h, f) for h, f in items if cnt.get(f, 0) == 1]
    hi = np.asarray([h for h, _ in items], int)
    fi = np.asarray([f for _, f in items], int)
    pos = he_mapped[hi] if len(hi) else np.zeros((0, 2))
    disp = (if_pts[fi] - he_mapped[hi]) if len(hi) else np.zeros((0, 2))
    return CAn.Anchors(name, hi, fi, pos, disp, np.ones(len(hi)))


@dataclass
class FieldResult:
    method: str
    applied: bool
    reason: str | None
    field_x: np.ndarray             # applied field (zeros when not applied)
    field_y: np.ndarray
    attempted_x: np.ndarray
    attempted_y: np.ndarray
    bounds: tuple                   # (min_x, min_y, max_x, max_y) of the grid, IF frame
    spacing: float
    audit: dict
    anchors: dict

    def displacement(self, pts, attempted=False):
        fx, fy = (self.attempted_x, self.attempted_y) if attempted else (self.field_x, self.field_y)
        return sample_displacement_field(np.asarray(pts, float).reshape(-1, 2), fx, fy, bounds=self.bounds, spacing=self.spacing)

    def inside(self, pts):
        x0, y0, x1, y1 = self.bounds
        p = np.asarray(pts, float).reshape(-1, 2)
        return (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)

    def forward_displacement(self, pts):
        """The displacement used for points AND (inverted) for images.

        Inside the (audited) grid: the bilinear field.  Outside: the border value fading linearly to 0 over TAPER_UM, so the
        map stays continuous (a hard cut would make the image inverse oscillate at the grid edge).  No nucleus of the ROI lies
        outside the grid (it pads the evaluated nuclei by 20 um); the taper gradient is reported in summary()."""
        p = np.asarray(pts, float).reshape(-1, 2)
        d = np.zeros_like(p)
        if not self.applied or not len(p):
            return d
        x0, y0, x1, y1 = self.bounds
        c = np.column_stack([np.clip(p[:, 0], x0, x1), np.clip(p[:, 1], y0, y1)])
        out = np.linalg.norm(p - c, axis=1)
        w = np.clip(1.0 - out / TAPER_UM, 0.0, 1.0)
        m = w > 0
        if m.any():
            d[m] = self.displacement(c[m]) * w[m, None]
        return d

    def taper_gradient_max(self) -> float:
        b = np.r_[self.field_x[0], self.field_x[-1], self.field_x[:, 0], self.field_x[:, -1]]
        bb = np.r_[self.field_y[0], self.field_y[-1], self.field_y[:, 0], self.field_y[:, -1]]
        return float(np.max(np.hypot(b, bb)) / TAPER_UM) if self.applied else 0.0

    def apply_points(self, mapped):
        """Deformed IF-frame positions x + d(x) (d = forward_displacement)."""
        p = np.asarray(mapped, float).reshape(-1, 2)
        return p + self.forward_displacement(p)

    def summary(self) -> dict:
        a = self.audit or {}
        keep = ("common_audit_pass", "hard_safety_pass", "topology_percentiles_pass", "point_improvement_gate_pass", "rejection_reason",
                "jacobian_min", "jacobian_p05", "jacobian_median", "jacobian_p95", "jacobian_max", "foldover_fraction",
                "displacement_max_um", "displacement_p95_um", "displacement_median_um", "finite")
        return {"method": self.method, "label": FIELD_LABELS.get(self.method, self.method), "applied": self.applied, "reason": self.reason,
                "taper_gradient_max": self.taper_gradient_max(),
                **{k: _json_default(a[k]) if isinstance(a.get(k), (np.generic, np.ndarray)) else a.get(k) for k in keep}, **self.anchors}


def _wrap(method, r: dict, gx, gy, spacing, anchors_info) -> FieldResult:
    bounds = (float(gx[0, 0]), float(gy[0, 0]), float(gx[0, -1]), float(gy[-1, 0]))
    ax = r.get("attempted_field_x", r["field_x"])
    ay = r.get("attempted_field_y", r["field_y"])
    return FieldResult(method, bool(r["applied"]), r.get("rejection_reason"), np.asarray(r["field_x"], float), np.asarray(r["field_y"], float),
                       np.asarray(ax, float), np.asarray(ay, float), bounds, float(spacing), r.get("audit") or {}, anchors_info)


def compute_fields(auto: AutoResult, review: Review | None, he_pts, if_pts, *, methods=FIELD_METHODS, spacing=GRID_UM,
                   exclude_conflicts=False) -> dict:
    """A / AUTO / REVIEWED / D on the ROI of `auto`.  Recomputed from scratch whenever the review changes."""
    _, _, V = modules()
    M, t = np.asarray(auto.meta["global"]["M"]), np.asarray(auto.meta["global"]["t"])
    he_mapped = map_points(M, t, he_pts)
    if_pts = np.asarray(if_pts, float)
    fixed, moving = if_pts[auto.if_idx], he_mapped[auto.he_idx]
    gx, gy, _ = DE.make_grid([fixed, moving], spacing=spacing)
    cells = CAn.occupied_cells(moving)
    out = {}
    sets = {}
    if "AUTO" in methods:
        sets["AUTO"] = anchors_from_pairs("auto_b2", auto.auto_pairs(), he_mapped, if_pts, exclude_conflicts)
    if "REVIEWED" in methods and review is not None:
        sets["REVIEWED"] = anchors_from_pairs("reviewed", review.pairs(), he_mapped, if_pts, exclude_conflicts)
    for m in methods:
        if m == "A":
            out[m] = _wrap(m, DE._affine("A", fixed, moving, gx, gy, spacing), gx, gy, spacing, {"anchors_n": 0})
        elif m in sets:
            a = sets[m]
            q = CAn.quality(a, extent_cells=cells)
            info = {"anchors_n": len(a), "anchor_coverage_cells": q.get("anchor_coverage_cells"), "cells_occupied": len(cells),
                    "exclude_conflicts": exclude_conflicts}
            if len(a) >= CAn.MIN_ANCHORS and (q.get("anchor_coverage_cells") or 0.0) < MIN_ANCHOR_COVERAGE:
                r = DE._affine(m, fixed, moving, gx, gy, spacing, reason=f"ANCHOR_COVERAGE_LOW (<{MIN_ANCHOR_COVERAGE:.2f})")
            else:
                r = DE._rbf_from_anchors(m, a, fixed, moving, gx, gy, spacing)
            out[m] = _wrap(m, r, gx, gy, spacing, info)
        elif m == "D":
            try:
                r = V.run_uot(fixed, moving, gx, gy, spacing, "UOT conservative")
            except RuntimeError as exc:
                r = DE._affine("D", fixed, moving, gx, gy, spacing, reason=f"UOT failed: {exc}")
            out[m] = _wrap(m, r, gx, gy, spacing, {"anchors_n": (r.get("diagnostics") or {}).get("anchors_kept")})
    return out


# ----------------------------------------------------------------------------- Phase 5: image (inverse map)
def invert_field(p, fr: FieldResult, tol=INVERSE_TOL_UM, max_iter=INVERSE_MAX_ITER):
    """Solve q + d(q) = p by fixed-point iteration (the same scheme as pointset_registration._inverse_fine_warp), with a residual check.

    Uses forward_displacement (grid field, tapered to 0 outside), i.e. exactly the map applied to the points.
    Returns (q, residual_um, converged mask)."""
    p = np.asarray(p, float).reshape(-1, 2)
    if not fr.applied:
        return p.copy(), np.zeros(len(p)), np.ones(len(p), bool)
    q = p.copy()
    for _ in range(max_iter):
        q = p - fr.forward_displacement(q)
    r = np.linalg.norm(q + fr.forward_displacement(q) - p, axis=1)
    return q, r, r <= tol


def he_source_coords(p_if, M, t, fr: FieldResult | None):
    """IF-frame positions -> HE-frame positions through the inverse of (affine, then field).  Returns (x_he, residual, converged)."""
    if fr is not None:
        q, r, ok = invert_field(p_if, fr)
    else:
        q, r, ok = np.asarray(p_if, float).reshape(-1, 2), np.zeros(len(p_if)), np.ones(len(p_if), bool)
    Mi, ti = inverse_affine(M, t)
    return map_points(Mi, ti, q), r, ok


def target_grid(roi, dst_um):
    x0, y0, x1, y1 = roi
    W, H = max(1, int(math.ceil((x1 - x0) / dst_um))), max(1, int(math.ceil((y1 - y0) / dst_um)))
    jj, ii = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    return np.stack([x0 + jj * dst_um, y0 + ii * dst_um], -1).reshape(-1, 2), (H, W)


def sample_pyramid(read_region, level_shape, src_um_k, corner_um, world_xy, order=1):
    """Sample a pyramid level at world positions, decoding only the region they touch.  read_region(x0, y0, x1, y1) -> array."""
    u = (world_xy[:, 0] - corner_um) / src_um_k - 0.5
    v = (world_xy[:, 1] - corner_um) / src_um_k - 0.5
    h, w = level_shape[:2]
    valid = np.isfinite(u) & np.isfinite(v) & (u >= -0.5) & (u <= w - 0.5) & (v >= -0.5) & (v <= h - 0.5)
    nch = level_shape[2] if len(level_shape) > 2 else 1
    out = np.zeros((len(u), nch), np.float32)
    if not valid.any():
        return out, valid, None
    x0, x1 = int(max(0, math.floor(u[valid].min()) - 2)), int(min(w, math.ceil(u[valid].max()) + 3))
    y0, y1 = int(max(0, math.floor(v[valid].min()) - 2)), int(min(h, math.ceil(v[valid].max()) + 3))
    patch = np.asarray(read_region(x0, y0, x1, y1))
    patch = patch if patch.ndim == 3 else patch[..., None]
    for c in range(nch):
        out[valid, c] = map_coordinates(patch[..., c].astype(np.float32), [v[valid] - y0, u[valid] - x0], order=order, mode="nearest")
    return out, valid, [x0, y0, x1, y1]


def warp_he_roi(he_reader, M, t, roi, dst_um, fr: FieldResult | None = None, order=1):
    """HE image resampled into the IF-frame ROI.  he_reader: dict(read=fn(level,x0,y0,x1,y1), level_shapes, downsamples, pixel_um).

    fr None -> affine only.  Nonrigid is reported as applied only if the field passed its audit AND the inverse converged
    for every target pixel whose source lies in the image."""
    tgt, (H, W) = target_grid(roi, dst_um)
    src, resid, ok = he_source_coords(tgt, M, t, fr)
    lvl = 0
    for k, ds in enumerate(he_reader["downsamples"]):
        if he_reader["pixel_um"] * ds <= dst_um * 1.0001:
            lvl = k
    um_k = he_reader["pixel_um"] * he_reader["downsamples"][lvl]
    corner = -0.5 * he_reader["pixel_um"]
    vals, valid, box = sample_pyramid(lambda a, b, c, d: he_reader["read"](lvl, a, b, c, d), he_reader["level_shapes"][lvl], um_k, corner, src, order)
    img = np.clip(vals.reshape(H, W, -1), 0, 255).astype(np.uint8)
    valid = valid.reshape(H, W)
    nonconv = int(np.sum(~ok.reshape(H, W) & valid))
    status = "AFFINE_ONLY" if fr is None else ("NOT_APPLIED (field rejected: " + str(fr.reason) + ")" if not fr.applied else
                                               ("INVERSE_NOT_CONVERGED" if nonconv else "APPLIED"))
    rec = {"mapping": "inverse: IF-frame target pixel centre -> field inverse (fixed point) -> affine inverse -> HE level pixel",
           "direction_forward": "HE -> IF:  x_IF = A(x_HE) + d(A(x_HE))", "target_frame": "IF (fixed) frame-local µm",
           "roi_if_frame": list(map(float, roi)), "target_um_per_px": dst_um, "target_shape": [H, W], "source_level": lvl,
           "source_um_per_px": um_k, "source_region_px": box, "interpolation": {0: "nearest", 1: "bilinear"}[order],
           "outside": "0 and invalid (no extrapolation)", "inverse_tol_um": INVERSE_TOL_UM, "inverse_max_iter": INVERSE_MAX_ITER,
           "inverse_residual_max_um": float(resid.max()) if len(resid) else 0.0, "inverse_not_converged_px": nonconv,
           "valid_fraction": float(valid.mean()), "status": status, "nonrigid_applied": status == "APPLIED",
           "field_method": None if fr is None else fr.method}
    return img, valid, rec


def nuclei_pixel_positions(he_pts_he_frame, M, t, fr: FieldResult | None, roi, dst_um):
    """Where the same transform puts HE nuclei in the warped ROI image (pixel col,row with centre at +0.5)."""
    mapped = map_points(M, t, he_pts_he_frame)
    p = fr.apply_points(mapped) if fr is not None else mapped
    return np.column_stack([(p[:, 0] - roi[0]) / dst_um - 0.5, (p[:, 1] - roi[1]) / dst_um - 0.5]), p


# ----------------------------------------------------------------------------- Phase 6: outputs
def export_bundle(sample_id, auto: AutoResult, review: Review | None, fields: dict, he_pts, he_ids, if_pts, if_ids, *, images=None,
                  image_records=None, settings=None, root: Path | None = None) -> Path:
    """Write every output of one comparison into a new directory (never overwrites)."""
    M, t = np.asarray(auto.meta["global"]["M"]), np.asarray(auto.meta["global"]["t"])
    ex = sample_dir(sample_id, root) / "exports" / f"exp-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    ex.mkdir(parents=True)
    he_mapped = map_points(M, t, he_pts)
    auto.table.to_csv(ex / "auto_correspondence.csv", index=False)
    pairs_frame(auto.auto_pairs(), he_ids, if_ids, he_mapped, if_pts).to_csv(ex / "auto_pairs.csv", index=False)
    if review is not None:
        pairs_frame(review.pairs(), he_ids, if_ids, he_mapped, if_pts, review.state()).to_csv(ex / "reviewed_pairs.csv", index=False)
        (ex / "review_history.json").write_text(json.dumps(review.to_json(), indent=1, ensure_ascii=False, default=_json_default))
    coords = pd.DataFrame({"he_index": np.arange(len(he_pts)), "he_id": [_scalar(v) for v in he_ids], "he_x": he_pts[:, 0], "he_y": he_pts[:, 1],
                           "affine_x": he_mapped[:, 0], "affine_y": he_mapped[:, 1]})
    sel = np.zeros(len(he_pts), bool); sel[auto.he_idx] = True
    coords["in_roi"] = sel
    safety = {}
    arrays = {}
    for m, fr in fields.items():
        p = fr.apply_points(he_mapped)
        coords[f"{m}_x"], coords[f"{m}_y"] = p[:, 0], p[:, 1]
        coords[f"{m}_inside_field"] = fr.inside(he_mapped)
        safety[m] = fr.summary()
        arrays.update({f"{m}_field_x": fr.field_x, f"{m}_field_y": fr.field_y, f"{m}_attempted_x": fr.attempted_x, f"{m}_attempted_y": fr.attempted_y})
    coords.to_csv(ex / "nuclei_coordinates.csv", index=False)
    any_fr = next(iter(fields.values()), None)
    np.savez_compressed(ex / "displacement_fields.npz", **arrays,
                        bounds=np.asarray(any_fr.bounds if any_fr else [0, 0, 0, 0]), spacing=np.asarray(any_fr.spacing if any_fr else 0.0))
    (ex / "global_transform.json").write_text(json.dumps({"direction": DIRECTION, "M": M.tolist(), "t": t.tolist(),
                                                          "meaning": "x_IF = M @ x_HE + t (frame-local µm)"}, indent=1))
    (ex / "safety.json").write_text(json.dumps(safety, indent=1, ensure_ascii=False, default=_json_default))
    if fields.get("AUTO") is not None and fields.get("REVIEWED") is not None:
        diff = displacement_change(fields["AUTO"], fields["REVIEWED"], he_mapped[auto.he_idx])
        (ex / "review_effect.json").write_text(json.dumps(diff, indent=1, default=_json_default))
    for name, img in (images or {}).items():
        from PIL import Image
        Image.fromarray(img).save(ex / f"{name}.png")
    (ex / "settings.json").write_text(json.dumps({"sample_id": sample_id, "auto_id": auto.auto_id, "auto_meta": auto.meta,
                                                  "review_id": review.review_id if review else None,
                                                  "review_version": review.version if review else None, "settings": settings or {},
                                                  "image_records": image_records or {}, "created_at": now_iso(), "code": git_info(),
                                                  "research_preview_only": True, "workflow_d": "not sent"},
                                                 indent=1, ensure_ascii=False, default=_json_default))
    return ex


def displacement_change(fa: FieldResult, fb: FieldResult, pts) -> dict:
    """How much the review changed the point positions (per nucleus, IF frame)."""
    pa, pb = fa.apply_points(pts), fb.apply_points(pts)
    d = np.linalg.norm(pb - pa, axis=1)
    return {"n": int(len(d)), "median_um": float(np.median(d)) if len(d) else 0.0, "p95_um": float(np.percentile(d, 95)) if len(d) else 0.0,
            "max_um": float(d.max()) if len(d) else 0.0, "moved_gt_1um": int(np.sum(d > 1.0)),
            "auto_applied": fa.applied, "reviewed_applied": fb.applied}


def bundle_zip(path: Path) -> bytes:
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(Path(path).iterdir()):
            z.write(f, f.name)
    return buf.getvalue()
