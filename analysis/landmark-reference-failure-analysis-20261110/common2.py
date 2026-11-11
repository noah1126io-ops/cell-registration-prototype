"""Shared loaders for the 2026-11-10 landmark-reference analysis (read-only access to the research store).

Reuses the 2026-10-27 audit loaders (sides, cached feature maps).  Convention: fixed(HE) = M @ moving(IF) + t, um.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
AUDIT = REPO / "analysis/real-image-localization-failure-audit-20261027"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(AUDIT))

import common as C27  # noqa: E402  (2026-10-27 loaders: side(), features())
from src.registration_core.reference import fit_reference  # noqa: E402

STORE = REPO / "runs/research-store/research.sqlite"
LANDMARK_ANN = "ann-20261027082721-b4b29aac"       # 031-8_6, 8 pairs, saved by the user 2026-10-27 17:27
ACCEPTED_6_2 = "tf-20261022082737-404825d8"        # 031-6_2 ACCEPTED automatic (same-section pair)
AUTO_8_6 = "tf-20261027080008-f02fab49"            # 031-8_6 automatic, occ+cleft (NOT_ACCEPTED)
MANUAL_8_6 = "tf-20261027064332-5467cd8f"          # 031-8_6 user's manual adjustment


def _db():
    return sqlite3.connect(f"file:{STORE}?mode=ro", uri=True)


def transform(tf_id):
    M, t = _db().execute("select M_json, t_json from transforms where id = ?", (tf_id,)).fetchone()
    return np.array(json.loads(M)), np.array(json.loads(t))


def landmarks():
    return json.loads(_db().execute("select payload_json from annotations where id = ?", (LANDMARK_ANN,)).fetchone()[0])["landmarks"]


def landmark_fit():
    L = landmarks()
    return fit_reference([l["fixed"] for l in L], [l["moving"] for l in L]), L


def reference(sample):
    """Reference pose per sample: 8_6 = landmark fit (researcher-specified), 6_2 = accepted automatic transform."""
    if sample == "031-8_6":
        f, _ = landmark_fit()
        return f["M"], f["t"]
    return transform(ACCEPTED_6_2)
