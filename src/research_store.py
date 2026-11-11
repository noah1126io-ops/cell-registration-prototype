"""Persistent research record for the Registration Explorer (SQLite, schema-versioned, append-only).

What is stored: experiment sessions, registration runs (automatic / assisted), transform variants (automatic,
candidate, manual, landmark), structured candidate reviews, annotations (ROI / landmarks / candidate exclusion /
manual initialisation / notes) and an event log.  Large run artifacts stay as JSON files; the DB stores their path
and sha256.

Integrity rules
  * transforms, reviews, annotations and events are append-only (UPDATE / DELETE are blocked by triggers);
    a correction is a new row that names the row it ``supersedes``.
  * runs only change their execution state / result summary through ``update_run``.
  * every write is one transaction under an exclusive file lock (the DB lives on NFS: rollback journal, no WAL).
  * a human review is a *Review Annotation*, never ground truth; nothing here changes an acceptance decision.
"""
from __future__ import annotations

import contextlib
import csv
import fcntl
import getpass
import hashlib
import io
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PATH = REPO / "runs/research-store/research.sqlite"
SCHEMA_VERSION = 1

JUDGEMENTS = {"correct": "正しいと思う", "incorrect": "間違っていると思う", "unsure": "判断できない"}
PROBLEMS = {"region": "対応領域が違う", "rotation": "Rotationが違う", "scale": "Scaleが違う", "translation": "Translationが違う",
            "nuclei": "核配置が合わない", "image": "画像構造が合わない", "display": "表示・座標系に問題がある", "other": "その他"}
ANNOTATION_KINDS = {"roi": "対応がありそうな領域（ROI）", "search_roi": "探索範囲の制限（ROI）", "landmarks": "対応点（ランドマーク）",
                    "exclusion": "誤りの候補（除外）", "manual_init": "手動の初期位置", "note": "メモ"}
TRANSFORM_KINDS = ("automatic", "candidate", "manual", "landmark", "imported")

_DDL = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions(
  id TEXT PRIMARY KEY, name TEXT NOT NULL, sample_id TEXT NOT NULL, note TEXT DEFAULT '',
  created_at TEXT NOT NULL, created_by TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs(
  id TEXT PRIMARY KEY, session_id TEXT REFERENCES sessions(id), sample_id TEXT NOT NULL,
  mode TEXT NOT NULL CHECK(mode IN ('automatic','assisted','imported')), localization TEXT, fixed TEXT NOT NULL, moving TEXT NOT NULL,
  state TEXT NOT NULL, slurm_job_id TEXT, run_dir TEXT, artifact_path TEXT, artifact_sha256 TEXT,
  status TEXT, accepted INTEGER, selected_candidate TEXT, failure_stage TEXT,
  input_fingerprint TEXT, software_json TEXT, params_json TEXT, constraints_json TEXT, annotation_ids_json TEXT,
  created_at TEXT NOT NULL, created_by TEXT NOT NULL, completed_at TEXT);
CREATE TABLE IF NOT EXISTS transforms(
  id TEXT PRIMARY KEY, session_id TEXT REFERENCES sessions(id), sample_id TEXT NOT NULL, run_id TEXT, candidate_id TEXT,
  kind TEXT NOT NULL CHECK(kind IN ('automatic','candidate','manual','landmark','imported')), parent_id TEXT REFERENCES transforms(id),
  fixed TEXT NOT NULL, moving TEXT NOT NULL, M_json TEXT NOT NULL, t_json TEXT NOT NULL, status TEXT NOT NULL,
  label TEXT DEFAULT '', note TEXT DEFAULT '', detail_json TEXT, supersedes TEXT REFERENCES transforms(id),
  created_at TEXT NOT NULL, created_by TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS reviews(
  id TEXT PRIMARY KEY, session_id TEXT REFERENCES sessions(id), sample_id TEXT NOT NULL, run_id TEXT, candidate_id TEXT,
  transform_id TEXT REFERENCES transforms(id), judgement TEXT NOT NULL CHECK(judgement IN ('correct','incorrect','unsure')),
  problems_json TEXT NOT NULL, comment TEXT DEFAULT '', view_json TEXT, supersedes TEXT REFERENCES reviews(id),
  created_at TEXT NOT NULL, created_by TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS annotations(
  id TEXT PRIMARY KEY, session_id TEXT REFERENCES sessions(id), sample_id TEXT NOT NULL,
  kind TEXT NOT NULL CHECK(kind IN ('roi','search_roi','landmarks','exclusion','manual_init','note')),
  frame TEXT NOT NULL, geometry_json TEXT, payload_json TEXT, label TEXT DEFAULT '',
  run_id TEXT, candidate_id TEXT, transform_id TEXT REFERENCES transforms(id), supersedes TEXT REFERENCES annotations(id),
  created_at TEXT NOT NULL, created_by TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, session_id TEXT, sample_id TEXT, actor TEXT NOT NULL,
  action TEXT NOT NULL, payload_json TEXT);
CREATE INDEX IF NOT EXISTS ix_runs_sample ON runs(sample_id, created_at);
CREATE INDEX IF NOT EXISTS ix_tf_sample ON transforms(sample_id, created_at);
CREATE INDEX IF NOT EXISTS ix_rev_sample ON reviews(sample_id, created_at);
CREATE INDEX IF NOT EXISTS ix_ann_sample ON annotations(sample_id, created_at);
"""
_APPEND_ONLY = ("transforms", "reviews", "annotations", "events")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _id(prefix):
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"


def _j(v):
    return json.dumps(v, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Store:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or os.environ.get("CELLREG_RESEARCH_STORE", str(DEFAULT_PATH)))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(".lock")
        with self._write() as con:
            for stmt in (x.strip() for x in _DDL.split(";")):   # executescript() would COMMIT the open transaction
                if stmt:
                    con.execute(stmt)
            for t in _APPEND_ONLY:
                con.execute(f"CREATE TRIGGER IF NOT EXISTS {t}_no_update BEFORE UPDATE ON {t} BEGIN SELECT RAISE(ABORT, '{t} is append-only'); END")
                con.execute(f"CREATE TRIGGER IF NOT EXISTS {t}_no_delete BEFORE DELETE ON {t} BEGIN SELECT RAISE(ABORT, '{t} is append-only'); END")
            row = con.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if row is None:
                con.execute("INSERT INTO meta(key, value) VALUES('schema_version', ?)", (str(SCHEMA_VERSION),))
            elif int(row[0]) > SCHEMA_VERSION:
                raise RuntimeError(f"research store schema {row[0]} is newer than this code ({SCHEMA_VERSION})")

    # ------------------------------------------------------------------ connection helpers
    def _connect(self):
        con = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=DELETE")      # NFS: no WAL
        con.execute("PRAGMA synchronous=FULL")
        con.execute("PRAGMA foreign_keys=ON")
        return con

    @contextlib.contextmanager
    def _write(self):
        with open(self.lock_path, "a+") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            con = self._connect()
            try:
                con.execute("BEGIN IMMEDIATE")
                yield con
                con.execute("COMMIT")
            except BaseException:
                if con.in_transaction:
                    con.execute("ROLLBACK")
                raise
            finally:
                con.close()
                fcntl.flock(lk, fcntl.LOCK_UN)

    def _read(self, sql, args=()):
        con = self._connect()
        try:
            return [dict(r) for r in con.execute(sql, args).fetchall()]
        finally:
            con.close()

    def integrity_check(self) -> str:
        return self._read("PRAGMA integrity_check")[0]["integrity_check"]

    def schema_version(self) -> int:
        return int(self._read("SELECT value FROM meta WHERE key='schema_version'")[0]["value"])

    def _event(self, con, action, payload, session_id=None, sample_id=None):
        con.execute("INSERT INTO events(ts, session_id, sample_id, actor, action, payload_json) VALUES(?,?,?,?,?,?)",
                    (_now(), session_id, sample_id, getpass.getuser(), action, _j(payload)))

    # ------------------------------------------------------------------ sessions
    def create_session(self, name, sample_id, note="") -> str:
        sid = _id("ses")
        with self._write() as con:
            con.execute("INSERT INTO sessions VALUES(?,?,?,?,?,?)", (sid, name, sample_id, note, _now(), getpass.getuser()))
            self._event(con, "session.create", {"name": name}, sid, sample_id)
        return sid

    def sessions(self, sample_id=None):
        if sample_id:
            return self._read("SELECT * FROM sessions WHERE sample_id=? ORDER BY created_at DESC", (sample_id,))
        return self._read("SELECT * FROM sessions ORDER BY created_at DESC")

    def session(self, sid):
        r = self._read("SELECT * FROM sessions WHERE id=?", (sid,))
        return r[0] if r else None

    # ------------------------------------------------------------------ runs
    def register_run(self, *, run_id, session_id, sample_id, mode, localization, fixed, moving, run_dir, params, constraints=None,
                     annotation_ids=(), slurm_job_id=None, state="queued"):
        with self._write() as con:
            con.execute("""INSERT INTO runs(id, session_id, sample_id, mode, localization, fixed, moving, state, slurm_job_id, run_dir,
                           params_json, constraints_json, annotation_ids_json, created_at, created_by) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (run_id, session_id, sample_id, mode, localization, fixed, moving, state, slurm_job_id, str(run_dir),
                         _j(params), _j(constraints or {}), _j(list(annotation_ids)), _now(), getpass.getuser()))
            self._event(con, "run.register", {"run_id": run_id, "mode": mode, "localization": localization}, session_id, sample_id)

    def update_run(self, run_id, *, state=None, slurm_job_id=None, artifact_path=None):
        """Execution state; on completion the artifact summary is copied in (the artifact file itself is not modified)."""
        sets, args = [], []
        if state:
            sets.append("state=?"); args.append(state)
        if slurm_job_id:
            sets.append("slurm_job_id=?"); args.append(str(slurm_job_id))
        if artifact_path:
            art = json.loads(Path(artifact_path).read_text())
            d = art.get("decision", {})
            sets += ["artifact_path=?", "artifact_sha256=?", "status=?", "accepted=?", "selected_candidate=?", "failure_stage=?",
                     "input_fingerprint=?", "software_json=?", "completed_at=?"]
            args += [str(artifact_path), sha256_file(artifact_path), d.get("status"), int(bool(d.get("accepted"))), d.get("selected_candidate"),
                     d.get("failure_stage"), _j(art.get("input")), _j(art.get("software")), _now()]
        if not sets:
            return
        with self._write() as con:
            row = con.execute("SELECT session_id, sample_id, artifact_path FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            if artifact_path and row["artifact_path"] and row["artifact_path"] != str(artifact_path):
                raise ValueError("a run's artifact cannot be replaced")
            con.execute(f"UPDATE runs SET {', '.join(sets)} WHERE id=?", (*args, run_id))
            self._event(con, "run.update", {"run_id": run_id, "state": state, "artifact": str(artifact_path) if artifact_path else None},
                        row["session_id"], row["sample_id"])

    def runs(self, sample_id=None, session_id=None):
        q, a = "SELECT * FROM runs WHERE 1=1", []
        if sample_id:
            q += " AND sample_id=?"; a.append(sample_id)
        if session_id:
            q += " AND session_id=?"; a.append(session_id)
        return self._read(q + " ORDER BY created_at DESC", a)

    def run(self, run_id):
        r = self._read("SELECT * FROM runs WHERE id=?", (run_id,))
        return r[0] if r else None

    # ------------------------------------------------------------------ transforms (append-only)
    def add_transform(self, *, sample_id, kind, fixed, moving, M, t, status, session_id=None, run_id=None, candidate_id=None,
                      parent_id=None, label="", note="", detail=None, supersedes=None) -> str:
        if kind not in TRANSFORM_KINDS:
            raise ValueError(kind)
        if kind in ("manual", "landmark") and status != "MANUAL_UNVERIFIED":
            raise ValueError("manual / landmark transforms are always MANUAL_UNVERIFIED")
        tid = _id("tf")
        with self._write() as con:
            con.execute("""INSERT INTO transforms(id, session_id, sample_id, run_id, candidate_id, kind, parent_id, fixed, moving, M_json, t_json,
                           status, label, note, detail_json, supersedes, created_at, created_by) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (tid, session_id, sample_id, run_id, candidate_id, kind, parent_id, fixed, moving, _j(M), _j(t), status, label, note,
                         _j(detail or {}), supersedes, _now(), getpass.getuser()))
            self._event(con, "transform.add", {"id": tid, "kind": kind, "label": label}, session_id, sample_id)
        return tid

    def transforms(self, sample_id, session_id=None, kind=None):
        q, a = "SELECT * FROM transforms WHERE sample_id=?", [sample_id]
        if session_id:
            q += " AND session_id=?"; a.append(session_id)
        if kind:
            q += " AND kind=?"; a.append(kind)
        return self._read(q + " ORDER BY created_at DESC", a)

    def transform(self, tid):
        r = self._read("SELECT * FROM transforms WHERE id=?", (tid,))
        return r[0] if r else None

    # ------------------------------------------------------------------ reviews (append-only)
    def add_review(self, *, sample_id, judgement, problems=(), comment="", session_id=None, run_id=None, candidate_id=None,
                   transform_id=None, view=None, supersedes=None) -> str:
        if judgement not in JUDGEMENTS:
            raise ValueError(judgement)
        bad = [p for p in problems if p not in PROBLEMS]
        if bad:
            raise ValueError(f"unknown problem types {bad}")
        rid = _id("rev")
        with self._write() as con:
            con.execute("""INSERT INTO reviews(id, session_id, sample_id, run_id, candidate_id, transform_id, judgement, problems_json, comment,
                           view_json, supersedes, created_at, created_by) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (rid, session_id, sample_id, run_id, candidate_id, transform_id, judgement, _j(list(problems)), comment, _j(view or {}),
                         supersedes, _now(), getpass.getuser()))
            self._event(con, "review.add", {"id": rid, "candidate": candidate_id, "judgement": judgement}, session_id, sample_id)
        return rid

    def reviews(self, sample_id, session_id=None, latest_only=False):
        q, a = "SELECT * FROM reviews WHERE sample_id=?", [sample_id]
        if session_id:
            q += " AND session_id=?"; a.append(session_id)
        rows = self._read(q + " ORDER BY created_at DESC", a)
        if latest_only:
            sup = {r["supersedes"] for r in rows if r["supersedes"]}
            rows = [r for r in rows if r["id"] not in sup]
        return rows

    # ------------------------------------------------------------------ annotations (append-only)
    def add_annotation(self, *, sample_id, kind, frame, geometry=None, payload=None, label="", session_id=None, run_id=None,
                       candidate_id=None, transform_id=None, supersedes=None) -> str:
        if kind not in ANNOTATION_KINDS:
            raise ValueError(kind)
        aid = _id("ann")
        with self._write() as con:
            con.execute("""INSERT INTO annotations(id, session_id, sample_id, kind, frame, geometry_json, payload_json, label, run_id, candidate_id,
                           transform_id, supersedes, created_at, created_by) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (aid, session_id, sample_id, kind, frame, _j(geometry), _j(payload or {}), label, run_id, candidate_id, transform_id,
                         supersedes, _now(), getpass.getuser()))
            self._event(con, "annotation.add", {"id": aid, "kind": kind, "label": label}, session_id, sample_id)
        return aid

    def annotations(self, sample_id, session_id=None, kind=None):
        q, a = "SELECT * FROM annotations WHERE sample_id=?", [sample_id]
        if session_id:
            q += " AND session_id=?"; a.append(session_id)
        if kind:
            q += " AND kind=?"; a.append(kind)
        rows = self._read(q + " ORDER BY created_at DESC", a)
        sup = {r["supersedes"] for r in rows if r["supersedes"]}
        return [r for r in rows if r["id"] not in sup]

    def annotation(self, aid):
        r = self._read("SELECT * FROM annotations WHERE id=?", (aid,))
        return r[0] if r else None

    def log(self, action, payload=None, session_id=None, sample_id=None):
        with self._write() as con:
            self._event(con, action, payload or {}, session_id, sample_id)

    def events(self, sample_id=None, session_id=None, limit=200):
        q, a = "SELECT * FROM events WHERE 1=1", []
        if sample_id:
            q += " AND sample_id=?"; a.append(sample_id)
        if session_id:
            q += " AND session_id=?"; a.append(session_id)
        return self._read(q + " ORDER BY id DESC LIMIT ?", (*a, int(limit)))

    # ------------------------------------------------------------------ export
    def export_json(self, sample_id, session_id=None) -> dict:
        def dec(rows):
            out = []
            for r in rows:
                r = dict(r)
                for k in list(r):
                    if k.endswith("_json") and r[k] is not None:
                        try:
                            r[k[:-5]] = json.loads(r.pop(k))
                        except json.JSONDecodeError:
                            pass
                out.append(r)
            return out
        return {"schema": "cellreg-research-store-export/1", "store_schema_version": self.schema_version(), "exported_at": _now(),
                "sample_id": sample_id, "session": self.session(session_id) if session_id else None,
                "runs": dec(self.runs(sample_id, session_id)), "transforms": dec(self.transforms(sample_id, session_id)),
                "reviews": dec(self.reviews(sample_id, session_id)), "annotations": dec(self.annotations(sample_id, session_id)),
                "events": dec(self.events(sample_id, session_id, limit=100000))}

    def export_csv(self, sample_id, session_id=None) -> dict:
        data = self.export_json(sample_id, session_id)
        out = {}
        for table in ("runs", "transforms", "reviews", "annotations", "events"):
            rows = data[table]
            buf = io.StringIO()
            if rows:
                keys = sorted({k for r in rows for k in r})
                w = csv.DictWriter(buf, fieldnames=keys)
                w.writeheader()
                for r in rows:
                    w.writerow({k: (_j(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})
            out[f"{table}.csv"] = buf.getvalue()
        return out


# ---------------------------------------------------------------------- reuse of annotations in a new run
def constraints_from_annotations(store: Store, annotation_ids, fixed_key, moving_key):
    """Build registration_core.Constraints from selected annotations (explicit opt-in; frames are checked)."""
    from .registration_core import Constraints
    c = Constraints(annotation_ids=list(annotation_ids))
    for aid in annotation_ids:
        a = store.annotation(aid)
        if a is None:
            raise KeyError(aid)
        geo = json.loads(a["geometry_json"] or "null")
        pay = json.loads(a["payload_json"] or "{}")
        if a["kind"] in ("search_roi", "roi"):
            if pay.get("frame_key", fixed_key) != fixed_key:
                raise ValueError(f"{aid}: ROI was drawn in the {pay.get('frame_key')} frame, but this run's fixed side is {fixed_key}")
            c.search_roi = geo
        elif a["kind"] == "exclusion":
            M, t = pay["M"], pay["t"]
            if pay.get("fixed") != fixed_key:     # stored in the other direction: invert
                import numpy as np
                Mi = np.linalg.inv(np.asarray(M, float)); M, t = Mi.tolist(), (-Mi @ np.asarray(t, float)).tolist()
            c.exclude_poses.append({"M": M, "t": t, "label": a["label"] or aid})
        elif a["kind"] == "manual_init":
            M, t = pay["M"], pay["t"]
            if pay.get("fixed") != fixed_key:
                import numpy as np
                Mi = np.linalg.inv(np.asarray(M, float)); M, t = Mi.tolist(), (-Mi @ np.asarray(t, float)).tolist()
            c.init = {"M": M, "t": t, "from": f"manual ({aid})"}
        elif a["kind"] == "landmarks":
            pairs = geo or []
            if pay.get("fixed") != fixed_key:
                pairs = [{"fixed": p["moving"], "moving": p["fixed"]} for p in pairs]
            c.landmarks = pairs
        # 'note' annotations carry no geometry for the search
    return c
