"""Manual adjustment of a displayed transform (rotation / scale / translation) by the user.

The adjustment is applied in the display (fixed) frame on top of a base transform (moving world -> fixed world):

    p_fixed = s * R(dθ) @ (M_base @ p + t_base - pivot) + pivot + (dx, dy)

so rotation and scaling happen about `pivot` (by default the centre of the mapped moving FOV) and the
translation is in fixed-frame units.  A manual transform is never a formal result: it is saved as
MANUAL_UNVERIFIED with accepted=False and is never handed to Workflow D.
"""
from __future__ import annotations

import getpass
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MANUAL_DIR = Path(os.environ.get("CELLREG_EXPLORER_MANUAL_DIR", str(REPO / "runs/registration-explorer-manual")))
STATUS = "MANUAL_UNVERIFIED"


@dataclass
class Adjustment:
    rotation_deg: float = 0.0
    scale: float = 1.0
    dx: float = 0.0
    dy: float = 0.0
    pivot: tuple = (0.0, 0.0)

    def is_identity(self) -> bool:
        return self.rotation_deg == 0.0 and self.scale == 1.0 and self.dx == 0.0 and self.dy == 0.0


def adjustment_matrix(adj: Adjustment):
    """(A, b) with p -> A @ p + b, in the display frame."""
    if not (adj.scale > 0 and math.isfinite(adj.scale)):
        raise ValueError("scale must be a positive finite number")
    a = math.radians(adj.rotation_deg)
    A = adj.scale * np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    c = np.asarray(adj.pivot, float)
    return A, c - A @ c + np.array([adj.dx, adj.dy], float)


def compose(M, t, adj: Adjustment):
    """Manual adjustment applied after the base transform (moving -> fixed); returns (M', t')."""
    A, b = adjustment_matrix(adj)
    return A @ np.asarray(M, float), A @ np.asarray(t, float) + b


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")[:60] or "manual"


def build_record(*, sample: str, fixed: str, moving: str, base: dict, adj: Adjustment, M, t, note: str = "", name: str = "",
                 diagnostics: dict | None = None, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    return {"schema": "registration-explorer-manual/1", "id": f"{now.strftime('%Y%m%d-%H%M%S')}-{_safe(name)}", "name": name, "note": note,
            "status": STATUS, "accepted": False, "sample": sample, "fixed": fixed, "moving": moving,
            "forward_transform": {"M": np.asarray(M, float).tolist(), "t": np.asarray(t, float).tolist(),
                                  "direction": f"{moving} world -> {fixed} world"},
            "base": base, "adjustment": {**asdict(adj), "pivot": [float(x) for x in adj.pivot],
                                         "formula": "p_fixed = s*R(rotation) @ (M_base p + t_base - pivot) + pivot + (dx, dy)"},
            "ui_diagnostics": diagnostics or {}, "created_at": now.isoformat(), "created_by": getpass.getuser(),
            "warning": "Manual, unverified transform. PREVIEW ONLY; never handed to Workflow D."}


def save(rec: dict, root: Path | None = None) -> Path:
    """Write one record from build_record() as JSON (atomic).  Existing files are never overwritten."""
    d = Path(root or MANUAL_DIR) / _safe(rec["sample"])
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{rec['id']}.json"
    k = 1
    while path.exists():
        path = d / f"{rec['id']}-{k}.json"; k += 1
    rec = {**rec, "id": path.stem}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=float))
    os.replace(tmp, path)
    return path


def list_saved(sample: str, root: Path | None = None) -> list[dict]:
    d = Path(root or MANUAL_DIR) / _safe(sample)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.json"), reverse=True):
        try:
            rec = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if rec.get("schema") == "registration-explorer-manual/1" and rec.get("sample") == sample:
            out.append({**rec, "_path": str(p)})
    return out
