"""Whole-cell segmentation results for display in Cell Review (research, read-only; 2026-11-18).

Cell outlines are kept apart from the nuclei: the nucleus centre points (``catalog.load_nuclei``) and nuclear contours
(``nuclear_contours``) are never modified.  Each result file is produced by
analysis/whole-cell-segmentation-20261118 (evaluate.py / full_section.py) and lies at

    <root>/<method>/<sample_id>/<section>.geojson           (section = "HE" | "IF"; polygons in that section's frame, um)
    <root>/<method>/<sample_id>/<section>_nucleus_cell.csv  (nucleus <-> cell table)

root = env ``CELLREG_CELLSEG_ROOT`` or the study's ``results`` directory.  Each feature carries ``cell_segment_id``,
``segmentation_method``/``_version``, area, perimeter, centroid, shape measures, the related nuclei (``nucleus_ids``
= original IDs, ``nucleus_indices`` = positions in load_nuclei at export) and ``quality_flags``.

Nuclei are re-joined by their **original ID** (never by position alone).  If the ID at the stored index no longer matches,
the file is reported as ``nucleus_id_mismatch`` and the relation falls back to the ID join (a missing ID -> no relation).

Nothing here claims that an outline is the true (pathological) cell boundary: there is no ground truth, and the
relations are not forced to 1 nucleus = 1 cell (multi-nucleus cells, cells without nucleus and nuclei without cell exist).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .nuclear_contours import _key

REPO = Path(__file__).resolve().parents[2]
STUDY = REPO / "analysis" / "whole-cell-segmentation-20261118"
SECTIONS = ("HE", "IF")
NONE = "none"

# Display registry.  `whole_cell` = the method is meant to detect cell boundaries (Voronoi is an image-agnostic baseline).
METHODS = {
    "none": {"label": "表示しない", "whole_cell": False},
    "S0_nuclei": {"label": "既存の核のみ（細胞輪郭なし・参照）", "whole_cell": False},
    "S1_cpsam_if_mem_full": {"label": "Cellpose-SAM・IF膜マーカー（切片全体）", "whole_cell": True, "sections": ("IF",)},
    "S1_cpsam_if_mem": {"label": "Cellpose-SAM・IF膜マーカー（評価ROIのみ）", "whole_cell": True, "sections": ("IF",)},
    "S1_cpsam_if_rgb": {"label": "Cellpose-SAM・IF表示RGB（評価ROIのみ）", "whole_cell": True, "sections": ("IF",)},
    "S1_cpsam_he_rgb": {"label": "Cellpose-SAM・H&E RGB（評価ROIのみ）", "whole_cell": True, "sections": ("HE",)},
    "S1_cpsam_he_eosin": {"label": "Cellpose-SAM・H&Eエオシン（評価ROIのみ）", "whole_cell": True, "sections": ("HE",)},
    "S2_mesmer": {"label": "Mesmer（IF）", "whole_cell": True, "sections": ("IF",)},
    "S3_watershed": {"label": "核シード Watershed（評価ROIのみ）", "whole_cell": True},
    "S3_voronoi": {"label": "核シード Voronoi（基準線・検出ではない）", "whole_cell": False},
}
# Methods whose results are known to be poor -- shown with a warning (REPORT of the study, §4).
CAVEATS = {
    "S1_cpsam_if_mem_full": "IFの膜マーカー由来の輪郭。031-8_6 は概ね妥当、031-6_2 は部分的（輪郭のない領域が多い）。正解データはありません。",
    "S1_cpsam_if_mem": "IFの膜マーカー由来の輪郭（ROI内）。正解データはありません。",
    "S1_cpsam_if_rgb": "主に核を分割しています（細胞全体の輪郭としては失敗）。",
    "S1_cpsam_he_rgb": "H&Eでは核だけを分割しています（細胞全体の輪郭としては失敗）。",
    "S1_cpsam_he_eosin": "H&Eエオシンからの分割はほとんど検出せず失敗しています。",
    "S3_watershed": "核を種にした境界信号上のWatershed。境界は画像に部分的にしか一致しません。",
    "S3_voronoi": "画像を使わない距離だけの分割（基準線）。細胞境界の検出結果ではありません。",
}
RELIABILITY = {"not_assessed": "未評価", "reliable": "輪郭は妥当", "partly": "一部妥当", "unreliable": "輪郭は信頼できない"}
CELL_FLAGS = {"seg_anomaly": "細胞セグメンテーション異常", "nucleus_cell_mismatch": "核と細胞の対応が不一致", "boundary_unclear": "細胞境界が不明瞭"}
_CACHE: dict = {}


def root() -> Path:
    return Path(os.environ.get("CELLREG_CELLSEG_ROOT") or (STUDY / "results"))


def path_of(sample_id: str, section: str, method: str, base: Path | None = None) -> Path:
    return (base or root()) / method / sample_id / f"{section}.geojson"


def availability(sample_id: str, section: str, method: str, base: Path | None = None) -> tuple[bool, str]:
    """(available, reason) for the selector; the reason is shown when a method cannot be used."""
    if method == NONE:
        return True, ""
    if method == "S0_nuclei":
        return True, "核輪郭は『核輪郭』の表示で切り替えます（細胞全体の輪郭ではありません）"
    spec = METHODS.get(method, {})
    if section not in spec.get("sections", SECTIONS):
        return False, f"{section} には適用できない方式です（{'/'.join(spec.get('sections', ()))} 専用）"
    p = path_of(sample_id, section, method, base)
    if p.is_file():
        return True, CAVEATS.get(method, "")
    if method == "S2_mesmer":
        log = (base or root()).parent / "runs_S2_all.json"
        try:
            err = json.loads(log.read_text()).get("model_error")
        except (OSError, json.JSONDecodeError):
            err = None
        if err and "DEEPCELL_ACCESS_TOKEN" in err:
            return False, "未実行: Mesmerの重みの取得に DEEPCELL_ACCESS_TOKEN が必要で、未設定のため実行できませんでした"
        return False, "未実行: Mesmerの結果がありません" + (f"（{err[:120]}）" if err else "")
    return False, f"このサンプル・切片の結果がありません（{p.relative_to(base or root()) if (base or root()) in p.parents else p}）"


def methods_for(sample_id: str, section: str, base: Path | None = None) -> list[tuple[str, bool, str]]:
    """[(method, available, reason)] for every registered method plus any extra result directory found on disk."""
    names = list(METHODS)
    b = base or root()
    if b.is_dir():
        names += sorted(d.name for d in b.iterdir() if d.is_dir() and d.name not in METHODS)
    return [(m, *availability(sample_id, section, m, base)) for m in names]


def label(method: str) -> str:
    return METHODS.get(method, {}).get("label", method)


@dataclass
class CellSegmentation:
    method: str
    version: str
    section: str
    sample_id: str
    ids: list                         # cell_segment_id (str), file order
    polys: list                       # (n_i, 2) arrays, section frame um (open ring)
    props: list                       # feature properties (dict)
    nuclei_of_cell: list              # [array of current nucleus indices] per cell (ID join)
    cells_of_nucleus: dict            # {nucleus index: [cell positions]}
    source: dict
    report: dict = field(default_factory=dict)
    _bb: np.ndarray | None = None

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def whole_cell(self) -> bool:
        return bool(self.source.get("is_whole_cell_detection"))

    def bboxes(self) -> np.ndarray:
        if self._bb is None:
            self._bb = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in self.polys]).reshape(-1, 4)
        return self._bb

    def query(self, rect, M=None, t=None) -> np.ndarray:
        """Cell positions whose bbox meets `rect` (given in the frame after the optional affine x -> M x + t)."""
        bb = self.bboxes()
        if not len(bb):
            return np.zeros(0, int)
        if M is not None:
            Mi = np.linalg.inv(np.asarray(M, float))
            r = np.asarray(rect, float)
            c = np.array([[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]]) - np.asarray(t, float)
            q = c @ Mi.T
            rect = [q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()]
        return np.flatnonzero((bb[:, 2] >= rect[0]) & (bb[:, 0] <= rect[2]) & (bb[:, 3] >= rect[1]) & (bb[:, 1] <= rect[3]))

    def polygon(self, j, M=None, t=None) -> np.ndarray:
        p = self.polys[int(j)]
        return p if M is None else p @ np.asarray(M, float).T + np.asarray(t, float)

    def pack(self, cells, M=None, t=None, decimals=2) -> dict:
        cells = [int(j) for j in cells]
        return {"idx": cells, "poly": [np.round(self.polygon(j, M, t), decimals).ravel().tolist() for j in cells]}

    def centroid(self, j) -> np.ndarray:
        p = self.props[int(j)]
        return np.array([p.get("centroid_x", np.nan), p.get("centroid_y", np.nan)], float)

    def cells_for(self, nucleus_index) -> list:
        return list(self.cells_of_nucleus.get(int(nucleus_index), []))

    def summary(self) -> dict:
        rel = {}
        for p in self.props:
            rel[p.get("cell_relation", "?")] = rel.get(p.get("cell_relation", "?"), 0) + 1
        return {"method": self.method, "version": self.version, "section": self.section, "n_cells": self.n, "cell_relations": rel,
                "n_nuclei_with_cell": len(self.cells_of_nucleus), "whole_cell_detection": self.whole_cell, **self.report}


def load(sample_id: str, section: str, method: str, nucleus_ids=None, base: Path | None = None) -> CellSegmentation | None:
    """Load one result (None if absent).  ``nucleus_ids`` = the current ``load_nuclei`` IDs of that section (for the join)."""
    if method in (NONE, "S0_nuclei"):
        return None
    p = path_of(sample_id, section, method, base)
    if not p.is_file():
        return None
    st = p.stat()
    idh = hashlib.sha1("|".join(_key(v) for v in nucleus_ids).encode()).hexdigest() if nucleus_ids is not None else None
    ck = (str(p), st.st_mtime_ns, st.st_size, idh)
    if ck in _CACHE:
        return _CACHE[ck]
    raw = p.read_bytes()
    d = json.loads(raw)
    meta = d.get("properties") or {}
    ids, polys, props = [], [], []
    for f in d.get("features", []):
        g = f.get("geometry") or {}
        if g.get("type") != "Polygon" or not g.get("coordinates"):
            continue
        ring = np.asarray(g["coordinates"][0], float)
        if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        if len(ring) < 3:
            continue
        pr = dict(f.get("properties") or {})
        ids.append(str(pr.get("cell_segment_id")))
        polys.append(ring)
        props.append(pr)
    report = {"duplicate_cell_ids": len(ids) - len(set(ids)), "nucleus_id_mismatch": 0, "nucleus_id_unknown": 0}
    lookup = {}
    if nucleus_ids is not None:
        keys = [_key(v) for v in nucleus_ids]
        for i, k in enumerate(keys):
            lookup.setdefault(k.strip("'"), i)
    nuclei_of_cell, cells_of_nucleus = [], {}
    for j, pr in enumerate(props):
        cur = []
        for nid, nidx in zip(pr.get("nucleus_ids") or [], pr.get("nucleus_indices") or [None] * len(pr.get("nucleus_ids") or [])):
            nid = str(nid)
            if nucleus_ids is None:
                i = nidx
            else:
                i = lookup.get(nid)
                if nidx is not None and (nidx >= len(nucleus_ids) or _key(nucleus_ids[nidx]).strip("'") != nid):
                    report["nucleus_id_mismatch"] += 1
                if i is None:
                    report["nucleus_id_unknown"] += 1
                    continue
            if i is None:
                continue
            cur.append(int(i))
            cells_of_nucleus.setdefault(int(i), []).append(j)
        nuclei_of_cell.append(np.asarray(cur, int))
    seg = CellSegmentation(method=meta.get("segmentation_method", method), version=str(meta.get("segmentation_version", "")), section=section,
                           sample_id=sample_id, ids=ids, polys=polys, props=props, nuclei_of_cell=nuclei_of_cell,
                           cells_of_nucleus=cells_of_nucleus,
                           source={"path": str(p), "sha256": hashlib.sha256(raw).hexdigest(), "description": meta.get("description", ""),
                                   "rois": meta.get("rois") or {}, "is_whole_cell_detection": bool(meta.get("is_whole_cell_detection", False)),
                                   "note": meta.get("note", "")},
                           report=report)
    if len(_CACHE) > 16:
        _CACHE.clear()
    _CACHE[ck] = seg
    return seg


def identifier(seg: CellSegmentation | None) -> dict | None:
    """Recorded with each review verdict: which result the reviewer saw."""
    if seg is None:
        return None
    return {"method": seg.method, "version": seg.version, "section": seg.section, "sha256": seg.source["sha256"]}


def validate_annotation(cell: dict | None) -> dict | None:
    """Cell-level review fields stored with a verdict (schema /3).  Raises ValueError on unknown values."""
    if not cell:
        return None
    out = {}
    for sec in ("source", "target"):
        s = cell.get(sec)
        if s:
            out[sec] = {"method": str(s.get("method") or NONE), "version": str(s.get("version") or ""),
                        "sha256": s.get("sha256"), "cell_ids": [str(v) for v in (s.get("cell_ids") or [])]}
    rel = cell.get("reliability", "not_assessed")
    if rel not in RELIABILITY:
        raise ValueError(f"不明な輪郭の信頼度: {rel}")
    out["reliability"] = rel
    flags = sorted(set(cell.get("flags") or []))
    bad = [x for x in flags if x not in CELL_FLAGS]
    if bad:
        raise ValueError(f"不明な細胞フラグ: {bad}")
    out["flags"] = flags
    out["comment"] = str(cell.get("comment") or "")
    return out
