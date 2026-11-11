"""Regression tests of the 2026-11-11 study helpers (derived cases, profiler, planner).  Run:
  pytest -q analysis/adaptive-registration-validation-20261111/test_adaptive_validation.py
Synthetic inputs only (no image I/O)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adaptive_strategy import example_profile, plan, validate_plan  # noqa: E402
from common import carry_reference, probe_dist  # noqa: E402
from data_profiler import relation_evidence, repetitiveness  # noqa: E402


class _Side:
    def __init__(self, A=np.eye(2), b=np.zeros(2)):
        self.A, self.b = np.asarray(A, float), np.asarray(b, float)


def _rot(d, s=1.0):
    a = np.deg2rad(d)
    return s * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


@pytest.mark.parametrize("fA,fb,mA,mb", [
    (np.eye(2), [-100, -40], np.eye(2), [0, 0]),              # HE crop
    (np.eye(2), [0, 0], np.eye(2), [-262, -262]),             # IF crop
    (np.eye(2), [0, 0], np.diag([-1, 1]), [1026, 0]),         # IF mirror
])
def test_carry_reference_maps_the_same_physical_point(fA, fb, mA, mb):
    M, t = _rot(13, 0.99), np.array([500.0, 300.0])
    f, m = _Side(fA, fb), _Side(mA, mb)
    M2, t2 = carry_reference(M, t, f, m)
    p = np.array([[10.0, 20.0], [400, 800]])
    expect = (p @ M.T + t) @ np.asarray(fA).T + fb            # original fixed point, then into the derived fixed frame
    got = (p @ np.asarray(mA).T + mb) @ M2.T + t2              # derived moving point through the carried transform
    assert np.allclose(got, expect)


def test_probe_dist_zero_for_identical():
    pr = np.array([[0.0, 0], [10, 0]])
    assert probe_dist(np.eye(2), [1, 2], np.eye(2), [1, 2], pr) == 0.0


def test_relation_unknown_is_never_same():
    r = relation_evidence(None, "x.tif", "y.tif", None)
    assert r["value"] == "UNKNOWN" and r["confidence"] == "NONE"


def test_relation_conflict_detected_for_6_2_like_records():
    r = relation_evidence("DIFFERENT (user correction)", "031-6_2_HE_031-6_XY02.ome.tif", "031-6_2_seqif.ome.tif", "031-6_2")
    assert r["filename_hint"] == "SAME" and r["conflict"] and r["confidence"] == "CONFLICT"


def test_relation_metadata_and_filename_agree_for_8_6_like_records():
    r = relation_evidence("DIFFERENT", "031-8_6_HE_reference_031-7_XY06.ome.tif", "031-8_6_seqif.ome.tif", "031-8_6")
    assert r["value"] == "DIFFERENT" and r["confidence"] == "CONFIRMED_METADATA"


def test_repetitiveness_periodic_vs_single_structure():
    yy, xx = np.mgrid[0:400, 0:400] * 2.0
    periodic = 1 + 0.5 * np.sin(2 * np.pi * xx / 120.0)
    single = 1 + np.exp(-((xx - 400) ** 2 + (yy - 400) ** 2) / (2 * 15.0 ** 2))
    fov = np.ones_like(periodic)
    assert repetitiveness(periodic, fov, 2.0) > 0.5 > repetitiveness(single, fov, 2.0)


def test_plan_keeps_all_sources_and_never_auto_accepts_tissue_level():
    p = plan(example_profile())
    assert p["sources"] == ["image", "points"]
    assert p["auto_decision"]["tissue_level_auto_accept_allowed"] is False
    assert validate_plan(p) == []


def test_plan_unknown_relation_blocks_cell_level_auto_accept():
    p = plan(example_profile())
    assert p["section_relation"]["gate"] == "both" and not p["auto_decision"]["cell_level_auto_accept_allowed"]


def test_plan_confirmed_same_allows_nuclear_gate_and_icp():
    pr = example_profile(); pr["relation"] = {"value": "SAME", "confidence": "CONFIRMED_METADATA", "metadata": "SAME", "filename_hint": "SAME"}
    p = plan(pr)
    assert p["section_relation"]["gate"] == "same" and p["refinement"]["nuclear_icp_final"] and p["auto_decision"]["cell_level_auto_accept_allowed"]


def test_plan_different_uses_tissue_gate_and_keeps_points():
    pr = example_profile(); pr["relation"] = {"value": "DIFFERENT", "confidence": "CONFIRMED_METADATA", "metadata": "DIFFERENT", "filename_hint": "DIFFERENT"}
    p = plan(pr)
    assert p["section_relation"]["gate"] == "adjacent" and "points" in p["sources"] and not p["refinement"]["nuclear_icp_final"]


def test_plan_weak_structure_keeps_image_source_but_points_primary():
    pr = example_profile(); pr["moving"]["cleft_strength"] = 0.01
    p = plan(pr)
    assert "image" in p["sources"] and p["primary_source"] == "points" and "WEAK_TISSUE_STRUCTURE" in p["risk_flags"]


def test_plan_small_field_requires_review():
    pr = example_profile(); pr["fov_area_ratio"] = 18.0
    assert "SMALL_MOVING_FIELD" in plan(pr)["risk_flags"]


def test_validate_plan_rejects_unsafe():
    p = plan(example_profile()); bad = copy.deepcopy(p)
    bad["auto_decision"]["tissue_level_auto_accept_allowed"] = True
    bad["auto_decision"]["cell_level_auto_accept_allowed"] = True
    assert len(validate_plan(bad)) == 2
