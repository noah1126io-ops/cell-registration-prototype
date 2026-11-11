import json
from pathlib import Path

import numpy as np
import pytest

from src.registration_core import Constraints, SideInput, WorkflowConfig, similarity_from_landmarks
from src.registration_core import workflow as wf
from src.registration_core import workflow_d_bridge as bridge
from src.registration_core.frames import Affine2D
from src.registration_core.search import Candidate, SearchConfig, pose_matrix
from src.workflow_c_result import load_workflow_c_result_artifact


def _rot(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def test_landmark_similarity_recovers_known_transform():
    rng = np.random.default_rng(1)
    M = 1.07 * _rot(23.0); t = np.array([310.0, -42.0])
    mov = rng.uniform(0, 500, (4, 2))
    M2, t2, res = similarity_from_landmarks(mov @ M.T + t, mov)
    assert np.allclose(M2, M, atol=1e-9) and np.allclose(t2, t, atol=1e-6) and res.max() < 1e-6
    with pytest.raises(ValueError):
        similarity_from_landmarks([[0, 0]], [[1, 1]])


def _cand(theta, t, score):
    return Candidate("proper", theta, 1.0, np.asarray(t, float), score, {}, {})


def test_exclusion_drops_matching_pose_and_reapplies_unchanged_rule():
    cfg = SearchConfig()
    fine = [_cand(10, (100, 100), 20.0), _cand(40, (600, 50), 19.5), _cand(80, (900, 900), 9.0)]
    out = {"fine": fine, "verdict": "AMBIGUOUS", "margin": 0.025}
    probe = np.array([[0, 0], [100, 0], [0, 100]], float)
    excl = [{"M": pose_matrix("proper", 10, 1.0), "t": [101.0, 99.0], "label": "wrong region"}]
    new, dropped = wf._apply_exclusions(out, excl, probe, cfg, 1)
    assert len(dropped) == 1 and dropped[0]["excluded_by"] == "wrong region"
    assert new["fine"][0].theta == 40
    # margin (19.5 - 9) / 19.5 = 0.54 >= 0.10 and z 19.5 >= 6 -> UNIQUE by the same prespecified rule
    assert new["verdict"] == "UNIQUE" and new["margin"] == pytest.approx(10.5 / 19.5)
    same, none = wf._apply_exclusions(out, [], probe, cfg, 1)
    assert same is out and none == []


def _c(src, rank, ok, M=np.eye(2), t=(0.0, 0.0), flags=()):
    T = {"M": np.asarray(M).tolist(), "t": list(t)}
    return {"id": f"{src}#{rank}", "source": src, "rank": rank, "geometric_pass": ok, "flags": list(flags), "T_final": T,
            "validation": {"search_pose_fixed_frame": {"geometric_pass": ok}}}


def test_decision_rule_fallback_conflict_assisted():
    probe = np.array([[0, 0], [500, 0], [0, 500], [500, 500]], float)
    S = lambda v: {"verdict": v, "margin": 0.5, "verdict_text": ""}  # noqa: E731
    # image ambiguous, points accepted -> point-only fallback (accepted)
    d = wf.decide({"image": S("AMBIGUOUS"), "points": S("UNIQUE")}, [_c("image", 0, True), _c("points", 0, True)], probe, "automatic", False)
    assert d["status"] == "POINT_ONLY_FALLBACK" and d["accepted"] and d["selected_candidate"] == "points#0"
    assert d["per_source"]["image"]["failure_stage"] == "C1"
    # both accepted, same region -> accepted
    d = wf.decide({"image": S("UNIQUE"), "points": S("UNIQUE")}, [_c("image", 0, True), _c("points", 0, True, t=(3.0, 0.0))], probe, "automatic", False)
    assert d["status"] == "ACCEPTED" and d["accepted"]
    # both accepted, different regions -> conflict, never accepted
    d = wf.decide({"image": S("UNIQUE"), "points": S("UNIQUE")}, [_c("image", 0, True), _c("points", 0, True, t=(800.0, 0.0))], probe, "automatic", False)
    assert d["status"] == "CONFLICT" and not d["accepted"] and d["failure_stage"] == "C3"
    # unique search but geometry fails -> C3
    d = wf.decide({"points": S("UNIQUE")}, [_c("points", 0, False)], probe, "automatic", False)
    assert not d["accepted"] and d["failure_stage"] == "C3"
    # ICP relocated the candidate and geometry fails -> C2
    d = wf.decide({"points": S("UNIQUE")}, [_c("points", 0, False, flags=["RELOCATED_BY_ICP"])], probe, "automatic", False)
    assert d["failure_stage"] == "C2"
    # assisted runs are never accepted automatically, even when the rule is met
    d = wf.decide({"init": S("UNIQUE")}, [_c("init", 0, True)], probe, "assisted", True)
    assert d["status"] == "ASSISTED_REVIEW_REQUIRED" and not d["accepted"]


def test_image_status_distinguishes_missing_and_absent_images():
    pts = np.random.default_rng(0).uniform(0, 100, (50, 2))
    a = SideInput("HE", pts, (100.0, 100.0), "s:HE", 1.0)
    b = SideInput("IF", pts, (100.0, 100.0), "s:IF", 1.0)
    st, ff, mf = wf.image_status(a, b)
    assert "NO_IMAGE" in st["flags"] and ff is None and mf is None
    a.image_path = "/nonexistent/x.ome.tif"
    st, _, _ = wf.image_status(a, b)
    assert "IMAGE_READ_FAILED" in st["flags"] and any("not found" in r for r in st["reasons"])


def _synthetic_points(seed=3, n=900, size=800.0):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, size, (n, 2))
    keep = [pts[0]]
    for p in pts[1:]:
        if np.min(np.linalg.norm(np.asarray(keep) - p, axis=1)) > 6:
            keep.append(p)
    return np.asarray(keep)


def test_run_workflow_point_only_recovers_known_partial_overlap_transform():
    """End to end on a synthetic pair with a KNOWN similarity: moving = 400x400 crop, rotated 30 deg."""
    fixed_pts = _synthetic_points()
    T_true = Affine2D(_rot(30.0), np.array([420.0, 120.0]), "m", "f")              # moving -> fixed
    inv = T_true.inverse()
    mov_all = inv.apply(fixed_pts)
    mov = mov_all[(mov_all[:, 0] >= 0) & (mov_all[:, 0] <= 400) & (mov_all[:, 1] >= 0) & (mov_all[:, 1] <= 400)]
    mov = mov + np.random.default_rng(5).normal(0, 0.5, mov.shape)
    f = SideInput("F", fixed_pts, (800.0, 800.0), "syn:F", 1.0)
    m = SideInput("M", mov, (400.0, 400.0), "syn:M", 1.0)
    cfg = WorkflowConfig(localization="points", search={"rot_step": 10.0, "res_coarse": 8.0, "res_fine": 4.0}, n_candidates=3)
    art = wf.run_workflow("syn", f, m, cfg, log=lambda s: None)
    json.dumps(art)                                                   # artifact is JSON-serialisable
    assert art["schema"] == wf.SCHEMA and art["mode"] == "automatic"
    assert art["decision"]["accepted"] and art["decision"]["status"] == "ACCEPTED"
    T = Affine2D.from_dict(art["transform"]["forward"])
    probe = np.array([[0, 0], [400, 0], [0, 400], [400, 400]], float)
    assert np.median(np.linalg.norm(T.apply(probe) - T_true.apply(probe), axis=1)) < 2.0
    assert art["input"]["fixed"]["points_sha256"] and art["software"]["core"]


def test_workflow_d_bridge_refuses_unaccepted_and_converts_accepted():
    art = {"schema": wf.SCHEMA, "mode": "automatic", "run_id": "r", "software": {}, "config": {"localization": "auto"},
           "decision": {"accepted": False, "status": "NOT_ACCEPTED", "selected_candidate": None, "rule": "x"},
           "input": {"fixed": {"key": "IF"}, "moving": {"key": "HE"}}, "transform": None, "candidates": []}
    assert bridge.check_handoff(art)
    with pytest.raises(bridge.NotAcceptedError):
        bridge.to_workflow_c_artifact(art, None, None)
    he_um = 0.5
    he_pts_world = np.random.default_rng(2).uniform(0, 200, (60, 2))
    M = 1.0 * _rot(10.0); t = np.array([50.0, 20.0])
    if_pts = he_pts_world @ M.T + t
    he = SideInput("HE", he_pts_world, (200.0, 200.0), "s:HE", he_um, shape_px=(400, 400))
    iff = SideInput("IF", if_pts, (300.0, 300.0), "s:IF", 1.0)
    T = Affine2D(M, t, "s:HE", "s:IF")
    art.update({"decision": {"accepted": True, "status": "ACCEPTED", "selected_candidate": "points#0", "rule": "x"},
                "transform": {"forward": T.to_dict()}, "candidates": [{"id": "points#0", "validation": {}}],
                "input": {"fixed": {"key": "IF", "points_sha256": wf.fingerprint(iff)["points_sha256"]},
                          "moving": {"key": "HE", "points_sha256": wf.fingerprint(he)["points_sha256"]}}})
    assert bridge.check_handoff(art) == []
    z = load_workflow_c_result_artifact(bridge.to_workflow_c_artifact(art, he, iff))
    A, tt = z.arrays["affine_matrix"], z.arrays["affine_translation"]
    he_px = he_pts_world / he_um
    assert np.allclose(he_px @ A.T + tt, if_pts, atol=1e-9) and not z.applied and z.fine_method == "off"
    art["mode"] = "assisted"
    assert any("assisted" in w for w in bridge.check_handoff(art))
    art["mode"] = "automatic"
    with pytest.raises(ValueError):                                   # different nuclei than the registration used
        bridge.to_workflow_c_artifact(art, SideInput("HE", he_pts_world + 1, (200.0, 200.0), "s:HE", he_um, shape_px=(400, 400)), iff)


def test_constraints_used_flag():
    assert not Constraints().used()
    assert Constraints(search_roi=[[0, 0], [1, 0], [1, 1]]).used()


ART_62 = Path(__file__).resolve().parents[1] / "analysis/workflow-c-unification-and-registration-review-20261022/cache/artifact_points_031-6_2_HEfix.json"


@pytest.mark.skipif(not ART_62.is_file(), reason="031-6_2 research artifact not present")
def test_workflow_c_research_affine_matches_research_registration():
    """Production-shaped global affine (HE px -> IF um) from the accepted 031-6_2 run reproduces it; wrong points refused."""
    from scipy.spatial import cKDTree
    from src import workflow_c_research as wcr
    from src.registration_explorer.catalog import load_manifest, side_input
    art = json.loads(ART_62.read_text())
    S = load_manifest()["031-6_2"]
    he, iff = side_input(S, "HE", use_image=False), side_input(S, "IF", use_image=False)
    he_px = he.points / he.pixel_um
    res = wcr.affine_from_research(art, he_px, iff.points, 11243, 11411)
    T = Affine2D.from_dict(art["transform"]["forward"]).inverse()            # stored IF->HE (HE fixed); Workflow C needs HE->IF
    assert np.allclose(res.transformed_points, T.apply(he.points), atol=1e-6)
    d = cKDTree(he_px @ res.affine_matrix.T + res.translation).query(iff.points)[0]
    assert res.success and not res.flip_x and not res.flip_y and np.median(d) < 5.0
    with pytest.raises(ValueError):
        wcr.affine_from_research(art, he_px[::-1] * 0.5, iff.points, 11243, 11411)


@pytest.mark.skipif(not ART_62.is_file(), reason="031-6_2 research artifact not present")
def test_scene_places_moving_image_and_nuclei_consistently():
    """Display check with the accepted transform: the moving image pixel of each nucleus lands on the drawn nucleus."""
    from src.registration_explorer import scene, catalog
    S = catalog.load_manifest()["031-6_2"]
    art = json.loads(ART_62.read_text())
    T = Affine2D.from_dict(art["transform"]["forward"])                     # IF -> HE
    av = catalog.availability(S)
    sc, _, _ = scene.build_scene(S, av, "HE", T.M, T.t, state_label="t", banners=[], opts=scene.ImageOptions(target_um=3.2), show_vectors=False)
    mov = next(p for p in sc["points"] if p["id"] == "moving_nuclei")
    xy = np.asarray(mov["xy"]).reshape(-1, 2)
    pts = catalog.load_nuclei(S.modality("IF"))[0]
    assert np.abs(xy - T.apply(pts)).max() < 0.006
    L = next(L for L in sc["images"] if L["role"] == "moving")
    m = L["m"]
    A = np.array([[m[0], m[2]], [m[1], m[3]]]); b = np.array([m[4], m[5]])
    um0 = S.modality("IF").pixel_um
    um_k = np.linalg.norm(np.linalg.inv(T.M) @ A[:, 0])                     # level-k pixel size in IF um
    uv = (pts[:50] / um0 + 0.5) * um0 / um_k                                 # nucleus centre -> level-k pixel coords (corner convention)
    assert np.abs(uv @ A.T + b - T.apply(pts[:50])).max() < 1e-6 + 1e-9 * um_k


# ----------------------------------------------------------------------------- 2026-10-27: reference landmarks / features
def test_reference_fit_rejects_outlier_and_reports_heldout():
    from src.registration_core import reference as ref
    rng = np.random.default_rng(3)
    M = 1.1 * _rot(15.5); t = np.array([700.0, 630.0])
    mov = rng.uniform(0, 1000, (9, 2))
    fix = mov @ M.T + t + rng.normal(0, 2.0, (9, 2))
    fix[4] += [180.0, -90.0]                                   # one wrong click
    r = ref.fit_reference(fix, mov)
    assert r["inliers"][4] is False and r["n_inliers"] == 8
    assert abs(r["params"]["scale"] - 1.1) < 0.01 and abs(r["params"]["rotation_deg"] - 15.5) < 0.5
    assert r["loo_median_um"] < 10 and r["holdout"]["n_test"] >= 2 and r["holdout"]["median_um"] < 10
    c = ref.compare(M, t, r["M"], r["t"], (1026, 1026), [{"fixed": f.tolist(), "moving": m.tolist()} for f, m in zip(fix[:3], mov[:3])])
    assert c["probe_median_um"] < 10 and abs(c["d_rotation_deg"]) < 0.5 and c["landmark_error_median_um"] < 10


def test_reference_compare_flags_180_degree_flip():
    from src.registration_core import reference as ref
    M = _rot(15.0); t = np.array([100.0, 100.0])
    c = ref.compare(_rot(195.0), t, M, t, (1000, 1000))
    assert abs(abs(c["d_rotation_deg"]) - 180) < 1e-6 and c["probe_median_um"] > 500


def test_occupancy_ignores_nuclei_and_maps_background_to_zero():
    from src.registration_core.features import occupancy
    rng = np.random.default_rng(0)
    x = np.full((200, 200), 1.0, np.float32)                   # slide background
    x[40:160, 40:160] = 10.0                                    # tissue
    x[90:110, 40:160] = 1.0                                     # a cleft
    ij = rng.integers(45, 155, (60, 2)); x[ij[:, 0], ij[:, 1]] = 200.0     # bright nuclei
    o = occupancy(x, 2.0, 6.0)
    assert o[10, 10] < 0.05 and o[100, 100] < 0.3 and np.median(o[50:85, 50:150]) > 0.9


def test_section_relation_from_file_names():
    def side(key, path):
        return SideInput(key, np.zeros((0, 2)), (100.0, 100.0), "f", image_path=path)
    adj = wf.infer_section_relation("031-8_6", side("HE", "/d/031-8_6_HE_reference_031-7_XY06.ome.tif"), side("IF", "/d/031-8_6_seqif.ome.tif"))
    assert adj["value"] == "adjacent" and adj["sections"] == {"HE": "031-7", "IF": "031-8"}
    same = wf.infer_section_relation("031-6_2", side("HE", "/d/031-6_2_HE_031-6_XY02.ome.tif"), side("IF", "/d/031-6_2_seqif.ome.tif"))
    assert same["value"] == "same"
    assert wf.infer_section_relation("x", side("HE", None), side("IF", None))["value"] == "unknown"
    assert wf.infer_section_relation("031-8_6", side("HE", None), side("IF", None), "same")["source"] == "config"


def test_failure_stage_is_that_of_the_primary_image_search():
    # 031-8_6 (2026-10-27): image search UNIQUE but nuclear check failed, point search AMBIGUOUS -> the run stopped at C3,
    # not C1 (the old minimum over sources reported C1)
    probe = np.array([[0, 0], [500, 0], [0, 500], [500, 500]], float)
    S = lambda v: {"verdict": v, "margin": 0.5, "verdict_text": ""}  # noqa: E731
    d = wf.decide({"image": S("UNIQUE"), "points": S("AMBIGUOUS")}, [_c("image", 0, False), _c("points", 0, False)], probe, "automatic", False)
    assert d["status"] == "NOT_ACCEPTED" and d["failure_stage"] == "C3"
    assert d["per_source"]["points"]["failure_stage"] == "C1"


def _cq(src, rank, q_pass):
    c = _c(src, rank, False)
    if q_pass is not None:
        c["validation"]["quadrant_consistency"] = {"pass": q_pass, "n_agree": 4 if q_pass else 1}
    return c


def test_adjacent_section_decision_is_tissue_level_and_never_accepted():
    probe = np.array([[0, 0], [500, 0], [0, 500], [500, 500]], float)
    S = lambda v: {"verdict": v, "margin": 0.5, "verdict_text": ""}  # noqa: E731
    d = wf.decide({"image": S("UNIQUE"), "points": S("AMBIGUOUS")}, [_cq("image", 0, True), _cq("points", 0, None)], probe, "automatic", False, "adjacent")
    assert d["status"] == "TISSUE_LEVEL" and d["accepted_tissue_level"] and not d["accepted"] and d["failure_stage"] is None
    assert d["selected_candidate"] == "image#0"
    d = wf.decide({"image": S("UNIQUE")}, [_cq("image", 0, False)], probe, "automatic", False, "adjacent")
    assert d["status"] == "NOT_ACCEPTED" and d["failure_stage"] == "C3" and not d["accepted_tissue_level"]
    d = wf.decide({"image": S("AMBIGUOUS")}, [_cq("image", 0, True)], probe, "automatic", False, "adjacent")
    assert d["status"] == "NOT_ACCEPTED" and d["failure_stage"] == "C1"
    # a point search alone can never give a tissue-level result on adjacent sections
    d = wf.decide({"points": S("UNIQUE")}, [_c("points", 0, True)], probe, "automatic", False, "adjacent")
    assert d["status"] == "NOT_ACCEPTED" and not d["accepted"]
    d = wf.decide({"image": S("UNIQUE")}, [_cq("image", 0, True)], probe, "assisted", True, "adjacent")
    assert d["status"] == "ASSISTED_REVIEW_REQUIRED" and not d["accepted"]
    # TISSUE_LEVEL is not a Workflow D status
    assert "TISSUE_LEVEL" not in bridge.ACCEPTED_STATUSES


def test_quadrant_consistency_reproduces_true_pose_and_rejects_wrong_one():
    from scipy.ndimage import gaussian_filter, map_coordinates
    from src.registration_core.features import FeatureMaps
    from src.registration_core.pipeline import Side
    rng = np.random.default_rng(5)
    res = 2.0
    fixed_ext, mov_ext = np.array([700.0, 700.0]), np.array([300.0, 300.0])
    fshape = (int(fixed_ext[1] / res), int(fixed_ext[0] / res))
    maps_f = {k: gaussian_filter(rng.normal(size=fshape), 6.0 / res).astype(np.float32) + 1.0 for k in ("occ", "cleft")}
    M = 1.0 * _rot(10.0)
    t = np.array([220.0, 180.0])
    yy, xx = np.mgrid[0:int(mov_ext[1] / res), 0:int(mov_ext[0] / res)]
    w = np.c_[xx.ravel(), yy.ravel()] * res @ M.T + t
    maps_m = {k: map_coordinates(v, [w[:, 1] / res, w[:, 0] / res], order=1).reshape(xx.shape).astype(np.float32) for k, v in maps_f.items()}
    ff = FeatureMaps("f", "HE", res, maps_f, np.ones(fshape, np.float32))
    mf = FeatureMaps("m", "IF", res, maps_m, np.ones(xx.shape, np.float32))
    fixed, moving = Side("HE", np.zeros((0, 2)), tuple(fixed_ext), ff), Side("IF", np.zeros((0, 2)), tuple(mov_ext), mf)
    scfg = SearchConfig(combine="zsum")
    ok = wf.quadrant_consistency(M, t, fixed, moving, "occ+cleft", scfg)
    assert ok["pass"] and ok["n_agree"] == 4
    assert max(q["shift_um"] for q in ok["quadrants"]) < 10
    bad = wf.quadrant_consistency(_rot(40.0), t + np.array([150.0, -60.0]), fixed, moving, "occ+cleft", scfg)
    assert not bad["pass"]
