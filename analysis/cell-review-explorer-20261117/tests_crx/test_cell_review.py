"""Tests for the cell-correspondence review -> deformation -> H&E warp research path (2026-11-17)."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO

STUDY = REPO / "analysis/cell-review-explorer-20261117"
MANIFEST = json.loads((STUDY / "baseline_manifest.json").read_text())


# ----------------------------------------------------------------------------- existing behaviour kept
def test_existing_files_unchanged_except_allowed():
    """Production Workflow C/D, the engine, the reused research code and the existing viewer are byte-identical."""
    from src.registration_research import legacy
    changed = [rel for rel, h in MANIFEST["files_sha256_at_head"].items() if legacy.sha256_file(REPO / rel) != h]
    assert set(changed) <= set(MANIFEST["allowed_to_change"]), changed
    for rel, h in MANIFEST["reused_research_sha256_from_1113"].items():
        assert legacy.sha256_file(REPO / rel) == h, rel


def test_tab_off_by_default(monkeypatch):
    from src.registration_explorer import cell_review_ui as U
    monkeypatch.delenv(U.ENV, raising=False)
    assert not U.enabled()
    monkeypatch.setenv(U.ENV, "1")
    assert U.enabled()


def test_page_hook_keeps_existing_tabs():
    src = (REPO / "src/registration_explorer/page.py").read_text()
    assert "tabs = st.tabs(tab_labels + ([rcm_ui.TAB_LABEL] if show_rcm else []))" in src      # the 11-13 assertion still holds
    assert "if show_crv:" in src and "crv_ui.render(S, avail, rec, fixed_key)" in src


def test_run_pipeline_frames_unchanged_by_runs_key(syn):
    """Returning the B0/B1/B3 runs (additive) does not change any frame."""
    from src.registration_research import cell_correspondence as CC
    from src.registration_research import cell_review as CR
    mapped = CR.map_points(syn["M"], syn["t"], syn["he_pts"])[:150]
    a = CC.run_pipeline(syn["if_pts"], mapped)
    b = CC.run_pipeline(syn["if_pts"], mapped)
    assert "runs" in a
    for k in CC.METHODS:
        assert a["frames"][k].drop(columns=[c for c in a["frames"][k].columns if "runtime" in c]).equals(
            b["frames"][k].drop(columns=[c for c in b["frames"][k].columns if "runtime" in c]))


# ----------------------------------------------------------------------------- Phase 2: automatic correspondence
def test_ids_preserved(syn, syn_auto):
    T, C = syn_auto.table, syn_auto.candidates
    assert (T["he_id"].to_numpy() == syn["he_ids"][T["he_index"].to_numpy(int)]).all()
    assert (T["top1_if_id"].to_numpy() == syn["if_ids"][T["top1_if_index"].to_numpy(int)]).all()
    assert (C["if_id"].to_numpy() == syn["if_ids"][C["if_index"].to_numpy(int)]).all()
    np.testing.assert_allclose(T[["he_x", "he_y"]].to_numpy(), syn["he_pts"][T["he_index"].to_numpy(int)])
    assert set(T["status"]) <= {"matched", "ambiguous", "unmatched"}
    assert C.groupby("he_index")["rank"].max().max() == 5          # candidates shown also for non-matched nuclei
    assert set(C["he_index"]) == set(T["he_index"])


def test_auto_quality_on_synthetic(syn, syn_auto):
    """Evaluation only (truth is never an input): the frozen B2 matched pairs are mostly correct here."""
    T = syn_auto.table
    m = T["status"] == "matched"
    correct = syn["true"][T.loc[m, "he_index"].to_numpy(int)] == T.loc[m, "top1_if_index"].to_numpy(int)
    assert m.sum() >= 100 and correct.mean() >= 0.9


def test_truth_not_an_input():
    import inspect
    from src.registration_research import cell_review as CR
    assert "true" not in inspect.signature(CR.compute_auto).parameters


def test_auto_written_once_and_never_overwritten(tmp_path, syn_auto):
    from src.registration_research import cell_review as CR
    d = CR.save_auto(copy.deepcopy(syn_auto), "SYN", root=tmp_path)
    with pytest.raises(FileExistsError):
        CR.save_auto(copy.deepcopy(syn_auto), "SYN", root=tmp_path)
    before = {p.name: p.read_bytes() for p in d.iterdir()}
    a = CR.load_auto(d)
    rv = CR.Review.start(a)
    h = next(iter(a.auto_pairs()))
    rv.apply("exclude", h, user="t", reason="x")
    rv.save("SYN", root=tmp_path); rv.save("SYN", root=tmp_path)
    assert {p.name: p.read_bytes() for p in d.iterdir()} == before          # review never touches the auto result
    assert len(list((tmp_path / "SYN" / "reviews" / rv.review_id).glob("v*.json"))) == 2   # versions are kept
    assert CR.load_auto(d).auto_pairs() == a.auto_pairs()
    (d / "auto_correspondence.csv").write_text("tampered")
    with pytest.raises(CR.ReviewError):
        CR.load_auto(d)


# ----------------------------------------------------------------------------- Phase 3: review
def test_review_operations_undo_redo_history(syn, syn_auto):
    from src.registration_research import cell_review as CR
    rv = CR.Review.start(syn_auto)
    auto = dict(syn_auto.auto_pairs())
    T = syn_auto.table
    h_sup, h_exc, h_re = list(auto)[:3]
    amb = int(T.loc[T.status != "matched", "he_index"].iloc[0])
    f_new = int(syn_auto.candidates.query("he_index == @h_re and rank == 2")["if_index"].iloc[0])
    f_add = int(syn_auto.candidates.query("he_index == @amb and rank == 1")["if_index"].iloc[0])
    rv.apply("support", h_sup, user="alice", reason="looks right")
    rv.apply("exclude", h_exc, user="alice", reason="wrong")
    rv.apply("reassign", h_re, f_new, user="alice", reason="rank 2")
    rv.apply("add", amb, f_add, user="alice", reason="new pair")
    rv.apply("unset", h_sup, user="alice", reason="not sure")
    s = rv.state()
    assert s[h_exc]["if_index"] is None and s[h_exc]["decision"] == "excluded"
    assert s[h_re] == {"if_index": f_new, "decision": "reassigned"}
    assert s[amb] == {"if_index": f_add, "decision": "added"}
    assert s[h_sup] == {"if_index": None, "decision": "undecided"}
    rv.undo("alice")
    assert rv.state()[h_sup] == {"if_index": auto[h_sup], "decision": "supported"}
    rv.redo("alice")
    assert rv.state()[h_sup]["decision"] == "undecided"
    rv.undo("alice"); rv.undo("alice")
    rv.apply("exclude", amb, user="bob")          # new edit after undo discards the redo tail (kept in the history)
    with pytest.raises(CR.ReviewError):
        rv.redo("bob")
    assert rv.history[-1]["discarded_redo"]
    h0 = rv.history[0]
    assert h0["user"] == "alice" and h0["reason"] == "looks right" and h0["at"] and h0["auto"]["if_index"] == auto[h_sup]
    assert syn_auto.auto_pairs() == auto                                     # the automatic result is unchanged
    rv2 = CR.Review.from_json(json.loads(json.dumps(rv.to_json(), default=CR._json_default)))
    assert rv2.pairs() == rv.pairs() and rv2.cursor == rv.cursor             # reproducible from the saved file


def test_invalid_ids_and_conflicts(syn, syn_auto):
    from src.registration_research import cell_review as CR
    rv = CR.Review.start(syn_auto)
    with pytest.raises(CR.ReviewError):
        rv.apply("add", len(syn["he_pts"]) + 5, 0)
    with pytest.raises(CR.ReviewError):
        rv.apply("add", 0, -1)
    with pytest.raises(CR.ReviewError):
        rv.apply("add", 0, len(syn["if_pts"]))
    with pytest.raises(CR.ReviewError):
        rv.apply("teleport", 0, 0)
    T = syn_auto.table
    h1, h2 = [int(h) for h in T.loc[T.status == "matched", "he_index"].iloc[:2]]
    f1 = syn_auto.auto_pairs()[h1]
    base = len(rv.conflicts())
    rv.apply("reassign", h2, f1, user="t")
    conf = rv.conflicts()
    assert len(conf) == base + (0 if f1 in CR.Review.start(syn_auto).conflicts() else 1)
    assert set(conf[f1]) >= {h1, h2}
    a = CR.anchors_from_pairs("x", rv.pairs(), CR.map_points(syn["M"], syn["t"], syn["he_pts"]), syn["if_pts"], exclude_conflicts=True)
    assert h1 not in set(a.moving_index.tolist()) and h2 not in set(a.moving_index.tolist())


# ----------------------------------------------------------------------------- Phase 4: deformation
@pytest.fixture(scope="module")
def base_fields(syn, syn_auto):
    from src.registration_research import cell_review as CR
    return CR.compute_fields(syn_auto, CR.Review.start(syn_auto), syn["he_pts"], syn["if_pts"])


def test_fields_applied_and_improve_on_synthetic(syn, syn_auto, base_fields):
    from src.registration_research import cell_review as CR
    f = base_fields
    assert f["A"].applied and f["AUTO"].applied and f["REVIEWED"].applied
    # an unedited review gives exactly the automatic field
    np.testing.assert_array_equal(f["AUTO"].field_x, f["REVIEWED"].field_x)
    # independent check (truth only for evaluation): shared nuclei end closer to their true IF partner than with affine only
    mapped = CR.map_points(syn["M"], syn["t"], syn["he_pts"])
    hs = syn_auto.he_idx[syn["true"][syn_auto.he_idx] >= 0]
    anchors = set(syn_auto.auto_pairs())
    hs = np.asarray([h for h in hs if h not in anchors])                  # nuclei that were NOT anchors
    tgt = syn["if_pts"][syn["true"][hs]]
    e_aff = np.median(np.linalg.norm(mapped[hs] - tgt, axis=1))
    e_def = np.median(np.linalg.norm(f["AUTO"].apply_points(mapped[hs]) - tgt, axis=1))
    assert len(hs) >= 20 and e_def < e_aff


def test_edits_change_the_field(syn, syn_auto, base_fields):
    from src.registration_research import cell_review as CR
    rv = CR.Review.start(syn_auto)
    T = syn_auto.table
    m = T.loc[T.status == "matched", "he_index"].to_numpy(int)
    for h in m[:10]:
        rv.apply("exclude", int(h), user="t")
    h = int(m[10]); f2 = int(syn_auto.candidates.query("he_index == @h and rank == 2")["if_index"].iloc[0])
    rv.apply("reassign", h, f2, user="t")
    amb = T.loc[T.status != "matched"].iloc[0]
    rv.apply("add", int(amb.he_index), int(amb.top1_if_index), user="t")
    f = CR.compute_fields(syn_auto, rv, syn["he_pts"], syn["if_pts"], methods=("REVIEWED",))["REVIEWED"]
    assert f.anchors["anchors_n"] == len(syn_auto.auto_pairs()) - 10 + 1
    assert not np.array_equal(f.attempted_x, base_fields["AUTO"].attempted_x)
    mapped = CR.map_points(syn["M"], syn["t"], syn["he_pts"])
    d = CR.displacement_change(base_fields["AUTO"], f, mapped[syn_auto.he_idx])
    assert d["max_um"] > 0


def test_too_few_or_badly_spread_anchors_fall_back_to_affine(syn, syn_auto):
    from src.registration_research import cell_review as CR
    rv = CR.Review.start(syn_auto)
    for h in list(syn_auto.auto_pairs())[15:]:
        rv.apply("exclude", h, user="t")
    f = CR.compute_fields(syn_auto, rv, syn["he_pts"], syn["if_pts"], methods=("REVIEWED",))["REVIEWED"]
    assert not f.applied and f.reason == "ANCHORS_INSUFFICIENT"
    # >= 20 anchors, all inside ONE occupied 100 um cell -> coverage gate (1 / 9 cells < 0.20)
    rv = CR.Review.start(syn_auto)
    mapped = CR.map_points(syn["M"], syn["t"], syn["he_pts"])
    pairs = syn_auto.auto_pairs()
    cell = {h: tuple(np.floor(mapped[h] / 100.0).astype(int)) for h in pairs}
    best = max(set(cell.values()), key=lambda c: sum(v == c for v in cell.values()))
    corner = [h for h in pairs if cell[h] == best]
    assert len(corner) >= 20
    for h in set(pairs) - set(corner):
        rv.apply("exclude", h, user="t")
    f = CR.compute_fields(syn_auto, rv, syn["he_pts"], syn["if_pts"], methods=("REVIEWED",))["REVIEWED"]
    assert not f.applied and f.reason.startswith("ANCHOR_COVERAGE_LOW")
    assert np.all(f.field_x == 0) and np.all(f.apply_points(mapped) == mapped)


def test_dangerous_edits_are_not_applied(syn, syn_auto):
    """Re-pairing half of the tissue to nuclei ~25 um away is rejected by the unchanged safety audit."""
    from src.registration_research import cell_review as CR
    from scipy.spatial import cKDTree
    rv = CR.Review.start(syn_auto)
    mapped = CR.map_points(syn["M"], syn["t"], syn["he_pts"])
    tree = cKDTree(syn["if_pts"])
    left = [h for h in syn_auto.auto_pairs() if mapped[h, 0] < np.median(mapped[syn_auto.he_idx, 0])]
    for h in left:
        _, f = tree.query(mapped[h] + np.array([25.0, 0.0]))
        rv.apply("reassign", h, int(f), user="t", reason="adversarial")
    f = CR.compute_fields(syn_auto, rv, syn["he_pts"], syn["if_pts"], methods=("REVIEWED",))["REVIEWED"]
    assert not f.applied and f.reason
    img, valid, rec = CR.warp_he_roi(_reader(np.zeros((50, 50, 3), np.uint8), 8.0), syn["M"], syn["t"], syn_auto.meta["roi_if_frame"], 4.0, f)
    assert rec["status"].startswith("NOT_APPLIED") and not rec["nonrigid_applied"]


# ----------------------------------------------------------------------------- Phase 5: image
def _reader(img, pixel_um, levels=(1, 2)):
    pyr = [img]
    for ds in levels[1:]:
        h, w = img.shape[0] // ds, img.shape[1] // ds
        pyr.append(img[:h * ds, :w * ds].reshape(h, ds, w, ds, -1).mean((1, 3)).astype(img.dtype))
    calls = []

    def read(lvl, x0, y0, x1, y1):
        calls.append((lvl, x0, y0, x1, y1))
        return pyr[lvl][y0:y1, x0:x1]
    return {"read": read, "level_shapes": tuple(p.shape for p in pyr), "downsamples": tuple(float(d) for d in levels),
            "pixel_um": pixel_um, "calls": calls}


def _blob_image(pts_he, pixel_um, shape, sigma_um=1.2):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w), np.float32)
    for x, y in pts_he:
        u, v = x / pixel_um, y / pixel_um           # nuclei: world = pixel index * pixel_um (pixel centres at integers)
        if -5 <= u < w + 5 and -5 <= v < h + 5:
            u0, v0 = int(round(u)), int(round(v))
            sl = (slice(max(0, v0 - 8), v0 + 9), slice(max(0, u0 - 8), u0 + 9))
            img[sl] += np.exp(-((xx[sl] - u) ** 2 + (yy[sl] - v) ** 2) * pixel_um ** 2 / (2 * sigma_um ** 2))
    return np.repeat(np.clip(img * 220, 0, 255).astype(np.uint8)[..., None], 3, -1)


def _smooth_field(roi, amp=3.0, spacing=4.0):
    from src.registration_research import cell_review as CR
    xs, ys = np.arange(roi[0] - 20, roi[2] + 24, spacing), np.arange(roi[1] - 20, roi[3] + 24, spacing)
    gx, gy = np.meshgrid(xs, ys)
    fx, fy = amp * np.sin(gy / 60.0), amp * np.cos(gx / 50.0)
    return CR.FieldResult("TEST", True, None, fx, fy, fx, fy, (float(xs[0]), float(ys[0]), float(xs[-1]), float(ys[-1])), spacing, {}, {})


def test_image_follows_the_same_transform_as_the_nuclei(syn, syn_auto):
    """Blobs drawn at HE nuclei, warped by inverse map, end where the forward map puts the nuclei (affine and nonrigid)."""
    from src.registration_research import cell_review as CR
    pix = 0.5
    he = syn["he_pts"] - syn["he_pts"].min(0) + 15.0        # keep everything inside the image
    M, t = syn["M"], syn["t"] + (syn["he_pts"].min(0) - 15.0) @ syn["M"].T
    shape = tuple(int(v) for v in (np.ceil(he.max(0)[::-1] / pix) + 30))
    rd = _reader(_blob_image(he, pix, shape), pix)
    roi = list(syn_auto.meta["roi_if_frame"])
    dst = 0.5
    for fr in (None, _smooth_field(roi), CR.compute_fields(syn_auto, None, syn["he_pts"], syn["if_pts"], methods=("AUTO",))["AUTO"]):
        img, valid, rec = CR.warp_he_roi(rd, M, t, roi, dst, fr)
        if fr is not None:
            assert rec["status"] == "APPLIED", rec
        cr, p = CR.nuclei_pixel_positions(he, M, t, fr, roi, dst)
        g = img[..., 0].astype(float)
        H, W = g.shape
        ok = (cr[:, 0] > 8) & (cr[:, 0] < W - 9) & (cr[:, 1] > 8) & (cr[:, 1] < H - 9)
        # isolated nuclei only (another nucleus within 6 um would shift the intensity centroid)
        from scipy.spatial import cKDTree
        dnn = cKDTree(p).query(p, 2)[0][:, 1]
        idx = np.flatnonzero(ok & (dnn > 6.0))
        errs = []
        for i in idx[:150]:
            c0, r0 = int(round(cr[i, 0])), int(round(cr[i, 1]))
            win = g[r0 - 4:r0 + 5, c0 - 4:c0 + 5]
            yy, xx = np.mgrid[r0 - 4:r0 + 5, c0 - 4:c0 + 5]
            wsum = win.sum()
            errs.append(np.hypot((win * xx).sum() / wsum - cr[i, 0], (win * yy).sum() / wsum - cr[i, 1]))
        assert len(errs) >= 30
        assert np.median(errs) < 0.25 and np.percentile(errs, 95) < 0.6, (fr and fr.method, np.median(errs), np.percentile(errs, 95))
    assert all(c[0] == 0 for c in rd["calls"])                 # 0.5 um output -> level 0 region reads only
    assert all((c[3] - c[1]) * (c[4] - c[2]) < shape[0] * shape[1] for c in rd["calls"])   # never the whole level


def test_inverse_not_converged_is_not_reported_as_applied(syn_auto):
    from src.registration_research import cell_review as CR
    roi = list(syn_auto.meta["roi_if_frame"])
    fr = _smooth_field(roi)
    c = np.array([(roi[0] + roi[2]) / 2, (roi[1] + roi[3]) / 2])
    gx, gy = np.meshgrid(np.arange(fr.field_x.shape[1]) * fr.spacing + fr.bounds[0], np.arange(fr.field_x.shape[0]) * fr.spacing + fr.bounds[1])
    fr.field_x, fr.field_y = -1.6 * (gx - c[0]), -1.6 * (gy - c[1])     # folding field (J < 0): no fixed point convergence
    img, valid, rec = CR.warp_he_roi(_reader(np.full((400, 400, 3), 128, np.uint8), 1.0), np.eye(2), np.zeros(2), roi, 4.0, fr)
    assert rec["status"] == "INVERSE_NOT_CONVERGED" and not rec["nonrigid_applied"] and rec["inverse_not_converged_px"] > 0


# ----------------------------------------------------------------------------- Phase 6: outputs
def test_export_bundle(tmp_path, syn, syn_auto, base_fields):
    from src.registration_research import cell_review as CR
    rv = CR.Review.start(syn_auto)
    rv.apply("exclude", next(iter(syn_auto.auto_pairs())), user="t", reason="r")
    ex = CR.export_bundle("SYN", syn_auto, rv, base_fields, syn["he_pts"], syn["he_ids"], syn["if_pts"], syn["if_ids"],
                          images={"he_affine": np.zeros((4, 4, 3), np.uint8)}, root=tmp_path)
    names = {p.name for p in ex.iterdir()}
    assert {"auto_correspondence.csv", "auto_pairs.csv", "reviewed_pairs.csv", "review_history.json", "nuclei_coordinates.csv",
            "displacement_fields.npz", "global_transform.json", "safety.json", "settings.json", "review_effect.json", "he_affine.png"} <= names
    import pandas as pd
    rp = pd.read_csv(ex / "reviewed_pairs.csv")
    assert len(rp) == len(syn_auto.auto_pairs()) - 1
    s = json.loads((ex / "settings.json").read_text())
    assert s["research_preview_only"] and s["workflow_d"] == "not sent"


# ----------------------------------------------------------------------------- UI (Streamlit AppTest)
def _app(sample, hide, he_only, root):
    import os
    import sys
    sys.path.insert(0, "/mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu")
    os.environ["CELLREG_CELL_REVIEW_ROOT"] = root
    import streamlit as st
    from src.registration_explorer import catalog, results
    from src.registration_explorer import cell_review_ui as U
    S = catalog.load_manifest()[sample]
    avail = catalog.availability(S, hide_images=hide)
    if he_only:
        avail["IF"]["image"], avail["IF"]["image_reason"] = False, "test: one-sided"
    recs = [r for r in results.load_records(S) if r.M is not None]
    rec = next((r for r in recs if r.method == "WF"), recs[0])
    st.session_state.setdefault("regx-crv-roisrc", "manual")
    U.render(S, avail, rec, "IF")


@pytest.mark.slow
@pytest.mark.parametrize("sample,hide,he_only", [("031-6_2", False, False), ("031-6_2", False, True), ("031-6_2", True, False),
                                                  ("serial_section_pair_20260904", False, False)])
def test_ui_flow(tmp_path, sample, hide, he_only):
    """Both images / one image / points only: compute -> pick a nucleus -> exclude -> undo/redo -> fields -> export."""
    from streamlit.testing.v1 import AppTest
    from src.registration_explorer import catalog, results
    from src.registration_research import cell_review as CR
    S = catalog.load_manifest()[sample]
    recs = [r for r in results.load_records(S) if r.M is not None]
    rec = next((r for r in recs if r.method == "WF"), recs[0])
    M, t, _ = results.oriented(rec.M, rec.t, rec.fixed, "IF")
    he, _ = catalog.load_nuclei(S.modality("HE"))
    iff, _ = catalog.load_nuclei(S.modality("IF"))
    roi = CR.default_roi(CR.map_points(M, t, he), iff, S.modality("IF").extent())
    c = [(roi[0] + roi[2]) / 2, (roi[1] + roi[3]) / 2]
    small = [c[0] - 160, c[1] - 160, c[0] + 160, c[1] + 160]
    at = AppTest.from_function(_app, kwargs={"sample": sample, "hide": hide, "he_only": he_only, "root": str(tmp_path)}, default_timeout=600)
    at.session_state["regx-crv-roisrc"] = "manual"
    at.run()
    assert not at.exception, at.exception
    if not CR.list_autos(sample, tmp_path):
        for i, v in enumerate(small):
            next(w for w in at.number_input if w.key.startswith(f"regx-crv-roi{i}-manual")).set_value(v)
        at.run()
        at.button(key="regx-crv-run").click().run()
    assert not at.exception, at.exception
    autos = CR.list_autos(sample, tmp_path)
    assert len(autos) == 1
    auto = CR.load_auto(autos[0])
    h = next(iter(auto.auto_pairs()), int(auto.table.he_index.iloc[0]))
    at.session_state[f"regx-crv-selhe-{auto.auto_id}"] = int(h)
    at.run()
    assert not at.exception, at.exception
    has_pair = h in auto.auto_pairs()
    at.button(key="regx-crv-exc" if has_pair else "regx-crv-uns").click().run()
    assert not at.exception, at.exception
    rv = at.session_state[f"regx-crv-review-{auto.auto_id}"]
    assert rv.cursor == 1
    at.button(key="regx-crv-undo").click().run()
    assert at.session_state[f"regx-crv-review-{auto.auto_id}"].cursor == 0
    at.button(key="regx-crv-redo").click().run()
    at.button(key="regx-crv-fgo").click().run()
    assert not at.exception, at.exception
    assert any("Affineのみ" in str(tb.value) for tb in at.table)
    if not hide and S.modality("HE").image_on_disk():
        at.selectbox(key="regx-crv-dst").set_value(2.0).run()
        at.button(key="regx-crv-imggo").click().run()
        assert not at.exception, at.exception
        imgs, recs, *_ = at.session_state[f"regx-crv-imgs-{auto.auto_id}"]
        assert "he_affine" in imgs and ("if_roi" in imgs) == (not he_only)
    at.button(key="regx-crv-export").click().run()
    assert not at.exception, at.exception
    ex = Path(at.session_state[f"regx-crv-export-{auto.auto_id}"])
    assert (ex / "nuclei_coordinates.csv").is_file() and (ex / "review_history.json").is_file()
    assert CR.load_auto(autos[0]).auto_pairs() == auto.auto_pairs()          # unchanged by the review


# ----------------------------------------------------------------------------- viewer JavaScript (Node, no browser)
def test_viewer_js_syntax_and_nucleus_pick(tmp_path):
    """The picker JS runs, and a click selects the nearest pickable nucleus of the requested role (global index)."""
    import shutil
    import subprocess
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    from src.registration_explorer import cell_review_viewer as crv
    # (the gateway's node is v12, too old for the unchanged template's `??`; only the new picker code is executed here)
    assert crv._BOOT in crv.html(600)
    he = crv.pick_points_layer("he", "HE", np.array([[10.0, 10.0], [50.0, 50.0]]), [7, 42], "he", "#f00")
    iff = crv.pick_points_layer("if", "IF", np.array([[12.0, 10.0], [49.0, 52.0]]), [3, 99], "if", "#0f0")
    harness = r"""
const sent = [];
global.window = {parent: {postMessage: (m) => sent.push(m)}, addEventListener: () => {}, __regx: {st: {view: {k: 2.0, ox: 0, oy: 0}}}};
global.sessionStorage = {getItem: () => null, setItem: () => {}};
__BOOT__
const S = {points: __PTS__};
const A = makeAPI(S, 't');
A.mode = 'pick_he'; A.up([49.0, 49.0]);            // near HE #42
A.mode = 'pick_if'; A.up([11.0, 10.5]);            // near IF #3
A.mode = 'pick_he'; A.up([200.0, 200.0]);          // nothing within 12 px -> no event
A.mode = 'view'; A.up([10.0, 10.0]);               // view mode never picks
const ev = sent.filter(m => m.type === 'streamlit:setComponentValue').map(m => m.value);
console.log(JSON.stringify(ev.map(e => [e.kind, e.role, e.index])));
"""
    js = harness.replace("__BOOT__", crv._BOOT).replace("__PTS__", json.dumps([he, iff]))
    (tmp_path / "h.js").write_text(js)
    out = subprocess.run([node, str(tmp_path / "h.js")], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip()) == [["nucleus", "he", 42], ["nucleus", "if", 3]]


def _full_page():
    import sys
    sys.path.insert(0, "/mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu")
    from src.registration_explorer.page import show_registration_explorer
    show_registration_explorer()


@pytest.mark.slow
@pytest.mark.parametrize("on", [False, True])
def test_full_explorer_page_with_and_without_tab(monkeypatch, tmp_path, on):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("CELLREG_CELL_REVIEW_ROOT", str(tmp_path))
    if on:
        monkeypatch.setenv("CELLREG_CELL_REVIEW", "1")
    else:
        monkeypatch.delenv("CELLREG_CELL_REVIEW", raising=False)
    at = AppTest.from_function(_full_page, default_timeout=300).run()
    assert not at.exception, at.exception
    labels = [t.label for t in at.tabs]
    assert labels[:8] == ["対応領域の探索", "Viewer（重ね合わせ・手動調整）", "評価・注釈", "方式比較", "Warp Preview", "新規実行", "実験記録", "Artifact"]
    assert ("細胞対応レビュー（研究）" in labels) == on
