"""Tests for nuclear contours, side-by-side comparison and the human review states (2026-11-18)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import REPO

STUDY = REPO / "analysis/cell-review-annotation-20261118"
MANIFEST = json.loads((STUDY / "baseline_manifest.json").read_text())


# ----------------------------------------------------------------------------- existing behaviour kept
def test_protected_files_unchanged():
    """Workflow C/D, engine, Graph/UOT/B2, Density/Joint Flow, viewer/scene/imaging/catalog/page and the sample manifest are byte-identical."""
    from src.registration_research import legacy
    changed = [rel for rel, h in MANIFEST["files_sha256_at_head"].items() if legacy.sha256_file(REPO / rel) != h]
    assert changed == [], changed


# ----------------------------------------------------------------------------- contours: synthetic files
def _circle(c, r, n=24):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p = np.column_stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)])
    return np.vstack([p, p[:1]]).tolist()


def _mod(tmp_path, fmt, pts_px, ids, polys, id_prop, *, pixel_um=0.5, extra_feats=()):
    from src.registration_explorer import catalog
    feats = [{"type": "Feature", "geometry": None if poly is None else {"type": "Polygon", "coordinates": [poly]},
              "properties": {id_prop: i, "centroid_x": p[0], "centroid_y": p[1]}}
             for p, i, poly in zip(pts_px, ids, polys) if (poly is not None or fmt != "csv_px")] + list(extra_feats)
    if fmt == "csv_px":
        csv = tmp_path / "nuc.csv"
        pd.DataFrame({"point_id": np.arange(len(pts_px)), "cx": pts_px[:, 0], "cy": pts_px[:, 1]}).to_csv(csv, index=False)
        (tmp_path / "nuc.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
        nuc = {"format": "csv_px", "path": str(csv), "x": "cx", "y": "cy", "units": "px"}
    else:
        g = tmp_path / "seg.geojson"
        g.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
        nuc = {"format": "geojson_px_props", "path": str(g), "x": "centroid_x", "y": "centroid_y", "units": "px"}
    return catalog.ModalityInfo("HE", "HE", "brightfield_rgb", pixel_um, None, None, nuc)


@pytest.mark.parametrize("fmt", ["csv_px", "geojson_px_props"])
def test_contours_join_by_id_and_detect_problems(tmp_path, fmt):
    from src.registration_explorer import catalog
    from src.registration_research import nuclear_contours as NC
    rng = np.random.default_rng(0)
    pts_px = rng.uniform(50, 950, (40, 2))
    n = len(pts_px)
    ids = list(range(n)) if fmt == "csv_px" else [float(10 + 3 * i) for i in range(n)]       # segment_id 10.0, 13.0, ...
    polys = [_circle(p, 8.0) for p in pts_px]
    polys[3] = None                                             # missing contour
    polys[5] = _circle(pts_px[5] + 40.0, 8.0)                   # contour of another place -> centre outside
    polys[7] = _circle(pts_px[7] + np.array([6.5, 0.0]), 8.0)    # inside but far from the area centroid (6.5 / 8 < 1 -> ok)
    polys[8] = _circle(pts_px[8] + np.array([7.8, 0.0]), 12.0, 48)
    # CSV only: a second polygon with the ID of nucleus 9 (duplicate) and a polygon with no nucleus (orphan).
    # (In the GeoJSON format every feature is itself a nucleus, so these cases are covered by the duplicate-ID rule below.)
    extra = [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [_circle(pts_px[9], 8.0)]}, "properties": {"point_id": ids[9]}},
             {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [_circle([5, 5], 3.0)]}, "properties": {"point_id": 99999}}] \
        if fmt == "csv_px" else []
    if fmt != "csv_px":
        ids[9] = ids[8]                                         # two nuclei share one segment_id -> both ambiguous
    mod = _mod(tmp_path, fmt, pts_px, ids, polys, "point_id" if fmt == "csv_px" else "segment_id", extra_feats=extra)
    pts, nid = catalog.load_nuclei(mod)
    c = NC.load_contours(mod, pts, nid)
    assert c is not None and c.n == n
    assert c.status[3] == "missing" and not c.has(3) and c.polygon(3) is None
    assert c.status[5] == "centroid_outside"
    assert c.status[9] == "duplicate_id" and c.polygon(9) is None        # ambiguous IDs are never used
    assert c.status[7] == "ok"
    if fmt == "csv_px":
        assert c.status[8] == "ok" and c.report["n_orphan_contours"] == 1
    else:
        assert c.status[8] == "duplicate_id"
    assert c.report["n_duplicate_ids_in_file"] == 1 and not c.report["frame_mismatch"]
    for i in np.flatnonzero(c.status == "ok"):                  # world units like the points (px * 0.5) and the same nucleus
        assert NC._inside(c.polygon(i), pts[i])
        assert np.allclose(np.asarray(polys[i])[:-1] * 0.5, c.polygon(i))
    # spatial query + pack (only what is needed for the view)
    q = c.query([0, 0, 150, 150])
    assert set(q) == {i for i in range(n) if c.polygon(i) is not None and c.bboxes()[i, 0] <= 150 and c.bboxes()[i, 1] <= 150}
    pk = c.pack(range(n))
    assert 3 not in pk["idx"] and 9 not in pk["idx"] and len(pk["idx"]) == n - (2 if fmt == "csv_px" else 3)


def test_contours_frame_mismatch_detected(tmp_path):
    """Contours of another pyramid level (coordinates x 0.5) are reported as a frame mismatch."""
    from src.registration_explorer import catalog
    from src.registration_research import nuclear_contours as NC
    pts_px = np.random.default_rng(1).uniform(200, 900, (30, 2))
    mod = _mod(tmp_path, "geojson_px_props", pts_px, [float(i) for i in range(30)], [_circle(p * 0.5, 6.0) for p in pts_px], "segment_id")
    pts, nid = catalog.load_nuclei(mod)
    c = NC.load_contours(mod, pts, nid)
    assert c.report["frame_mismatch"] and c.report["status_counts"]["centroid_outside"] == 30


def test_points_only_sample_has_no_contours():
    from src.registration_explorer import catalog
    from src.registration_research import nuclear_contours as NC
    S = catalog.load_manifest()["serial_section_pair_20260904"]
    for k in ("HE", "IF"):
        p, i = catalog.load_nuclei(S.modality(k))
        assert NC.load_contours(S.modality(k), p, i) is None


@pytest.mark.parametrize("sample", ["031-6_2", "031-8_6"])
def test_real_contours_match_the_centre_points(sample):
    """Every centre point of the real samples lies inside the contour with its own ID (H&E StarDist and IF provider)."""
    from src.registration_explorer import catalog
    from src.registration_research import nuclear_contours as NC
    S = catalog.load_manifest()[sample]
    for k, prop in (("HE", "point_id"), ("IF", "segment_id")):
        m = S.modality(k)
        p, i = catalog.load_nuclei(m)
        c = NC.load_contours(m, p, i)
        assert c.source["id_property"] == prop
        assert c.report["status_counts"]["ok"] == len(p), c.report
        assert not c.report["frame_mismatch"] and c.report["n_orphan_contours"] == 0
        rng = np.random.default_rng(0)
        for j in rng.choice(len(p), 50, replace=False):
            assert NC._inside(c.polygon(j), p[j])
        # mapped into another frame with an affine: the mapped contour still contains the mapped point
        M, t = np.array([[0.9, -0.2], [0.2, 0.9]]), np.array([10.0, -5.0])
        pk = c.pack([0, 1, 2], M, t)
        for j, poly in zip(pk["idx"], pk["poly"]):
            assert NC._inside(np.asarray(poly).reshape(-1, 2), M @ p[j] + t)


# ----------------------------------------------------------------------------- human review states
def _first(auto, status):
    t = auto.table
    return int(t.loc[t["status"] == status, "he_index"].iloc[0])


def test_review_states_are_separate_from_the_algorithm(syn, syn_auto, tmp_path):
    from src.registration_research import cell_review as CR
    auto = syn_auto
    rv = CR.Review.start(auto)
    pairs = auto.auto_pairs()
    h_ok, h_nid, h_abs, h_unc = list(pairs)[:4]
    h_um = _first(auto, "unmatched") if (auto.table["status"] == "unmatched").any() else _first(auto, "ambiguous")
    rv.apply("support", h_ok, user="r1")
    rv.apply("not_identified", h_nid, user="r1")
    with pytest.raises(CR.ReviewError, match="根拠"):
        rv.apply("absent", h_abs, user="r1")                    # Absent needs a stated reason
    rv.apply("absent", h_abs, user="r1", reason="IF側の同じ位置に核が見えない", tags=["edge_of_observation"])
    rv.apply("uncertain", h_unc, user="r1", tags=["image_quality", "split_merge"])
    with pytest.raises(CR.ReviewError, match="タグ"):
        rv.apply("uncertain", h_unc, tags=["no-such-tag"])
    v = rv.verdicts()
    assert v[h_ok]["review_status"] == "confirmed_match" and v[h_ok]["if_index"] == pairs[h_ok]
    assert v[h_nid]["review_status"] == "not_identified" and v[h_nid]["if_index"] is None and v[h_nid]["held_if_index"] == pairs[h_nid]
    assert v[h_abs]["review_status"] == "absent" and v[h_abs]["scope"]["absent_in"] == "IF" and v[h_abs]["scope"]["source_section"] == "HE"
    assert v[h_unc]["review_status"] == "uncertain" and v[h_unc]["tags"] == ["image_quality", "split_merge"]
    assert rv.review_status(h_um) == "unreviewed"               # an algorithm 'unmatched' is never turned into Absent
    # decisions without a partner are not anchors (same as "exclude"); the confirmed one stays
    pr = rv.pairs()
    assert h_ok in pr and h_nid not in pr and h_abs not in pr and h_unc not in pr
    # re-edit: change the verdict, then back to unreviewed
    rv.apply("uncertain", h_nid, user="r2", reason="再検討")
    assert rv.verdicts()[h_nid]["review_status"] == "uncertain" and rv.verdicts()[h_nid]["user"] == "r2"
    rv.apply("reset", h_nid)
    assert h_nid not in rv.verdicts() and rv.state()[h_nid] == {"if_index": pairs[h_nid], "decision": "auto"}
    with pytest.raises(CR.ReviewError):
        rv.apply("reset", h_nid)
    rv.undo(); assert rv.verdicts()[h_nid]["review_status"] == "uncertain"
    rv.redo(); assert h_nid not in rv.verdicts()
    # reviewed Uncertain differs from unreviewed
    assert rv.review_status(h_unc) == "uncertain" and rv.review_status(h_nid) == "unreviewed"
    # the existing state() format of the 11-17 ops is unchanged
    assert rv.state()[h_ok] == {"if_index": pairs[h_ok], "decision": "supported"}
    # persistence: schema /2 round trip with verdicts, tags, scope and context
    rv.context["segmentation"] = {"HE": {"sha256": "a" * 64}, "IF": {"sha256": "b" * 64}}
    p = rv.save("SYN", tmp_path)
    rv2 = CR.load_review(p)
    assert json.loads(p.read_text())["schema"] == "cellreg-cell-review-edits/2"
    assert rv2.verdicts() == rv.verdicts() and rv2.state() == rv.state() and rv2.context == rv.context
    # the auto result object is untouched
    assert auto.auto_pairs() == pairs


def test_schema_v1_reviews_still_load(syn_auto, tmp_path):
    from src.registration_research import cell_review as CR
    rv = CR.Review.start(syn_auto)
    h = next(iter(syn_auto.auto_pairs()))
    rv.apply("exclude", h, user="old")
    d = rv.to_json()
    d["schema"] = "cellreg-cell-review-edits/1"
    for k in ("verdicts", "context"):
        d.pop(k)
    for op in d["ops"]:                                         # 11-17 records had no tags / scope
        op.pop("tags", None)
    old = CR.Review.from_json(json.loads(json.dumps(d, default=CR._json_default)))
    assert old.state()[h] == {"if_index": None, "decision": "excluded"}
    assert old.verdicts()[h]["review_status"] == "released" and old.verdicts()[h]["tags"] == []
    assert old.context == {}


def test_conflicts_are_detected_not_resolved(syn_auto):
    from src.registration_research import cell_review as CR
    rv = CR.Review.start(syn_auto)
    (h1, f1), (h2, f2) = list(syn_auto.auto_pairs().items())[:2]
    rv.apply("reassign", h2, f1, user="r")                      # two H&E nuclei now point to the same IF nucleus
    before = CR.Review.start(syn_auto).conflicts()                # (the synthetic auto result already has its own conflicts)
    assert rv.conflicts()[f1] == sorted([h1, h2]) and f1 not in before
    assert rv.pairs()[h1] == f1 and rv.pairs()[h2] == f1        # nothing is silently dropped
    ann = CR.review_annotations("SYN", syn_auto, rv, np.arange(syn_auto.meta["inputs"]["n_he"]), np.arange(syn_auto.meta["inputs"]["n_if"]))
    assert bool(ann.loc[ann.source_index == h2, "if_conflict"].iloc[0])


def test_annotation_table_metrics_and_agreement(syn, syn_auto, tmp_path):
    from src.registration_research import cell_review as CR
    auto = syn_auto
    he_ids, if_ids = syn["he_ids"], syn["if_ids"]
    mapped = CR.map_points(syn["M"], syn["t"], syn["he_pts"])
    pairs = list(auto.auto_pairs().items())
    a = CR.Review.start(auto)
    (h1, f1), (h2, f2), (h3, _), (h4, _) = pairs[:4]
    a.apply("support", h1, user="a")
    other = int(auto.candidates[(auto.candidates.he_index == h2) & (auto.candidates.if_index != f2)]["if_index"].iloc[0])
    a.apply("reassign", h2, other, user="a")
    a.apply("absent", h3, user="a", reason="見えない")
    a.apply("uncertain", h4, user="a")
    ann = CR.review_annotations("SYN", auto, a, he_ids, if_ids, mapped, syn["if_pts"])
    assert list(ann.columns) == CR.ANNOTATION_COLUMNS and len(ann) == len(auto.table)
    r1 = ann[ann.source_index == h1].iloc[0]
    assert r1.review_status == "confirmed_match" and r1.target_cell_id == if_ids[f1] and r1.target_algorithm_rank == 1 and r1.agrees_with_algorithm_top1
    r2 = ann[ann.source_index == h2].iloc[0]
    assert r2.target_cell_id == if_ids[other] and r2.target_algorithm_rank > 1 and not r2.agrees_with_algorithm_top1
    assert r2.source_cell_id == he_ids[h2]                       # original IDs, not indices
    r3 = ann[ann.source_index == h3].iloc[0]
    assert r3.review_status == "absent" and r3.absent_in == "IF" and pd.isna(r3.target_index) and r3.comment == "見えない"
    un = ann[~ann.source_index.isin([h1, h2, h3, h4])]
    assert set(un.review_status) == {"unreviewed"} and un.target_index.isna().all()      # unreviewed: no human target
    assert (ann.algorithm_status.to_numpy() == auto.table.status.to_numpy()).all()       # algorithm state carried unchanged
    m = CR.review_metrics(ann)
    assert m["n_reviewed"] == 4 and m["review_status_counts"]["absent"] == 1 and m["review_status_counts"]["uncertain"] == 1
    assert m["precision_denominator"] == 3 and m["precision_of_algorithm_matched"] == pytest.approx(1 / 3)   # uncertain excluded
    assert m["algorithm_recall_of_confirmed"] == pytest.approx(1 / 2)
    assert m["absent_rate"] == pytest.approx(1 / 4)
    b = CR.Review.start(auto)
    b.apply("support", h1, user="b"); b.apply("support", h2, user="b"); b.apply("absent", h3, user="b", reason="x")
    ag = CR.review_agreement(ann, CR.review_annotations("SYN", auto, b, he_ids, if_ids))
    assert ag["n_both_reviewed"] == 3 and ag["status_agreement"] == pytest.approx(1.0) and ag["status_target_agreement"] == pytest.approx(2 / 3)


def test_export_contains_annotations(syn, syn_auto, tmp_path):
    from src.registration_research import cell_review as CR
    rv = CR.Review.start(syn_auto)
    h = next(iter(syn_auto.auto_pairs()))
    rv.apply("not_identified", h, user="r")
    ex = CR.export_bundle("SYN", syn_auto, rv, {}, syn["he_pts"], syn["he_ids"], syn["if_pts"], syn["if_ids"], root=tmp_path)
    ann = pd.read_csv(ex / "review_annotations.csv")
    assert ann.loc[ann.source_index == h, "review_status"].iloc[0] == "not_identified"
    assert json.loads((ex / "review_metrics.json").read_text())["n_reviewed"] == 1
    assert ann["source_contour_status"].eq("no_segmentation").all()


# ----------------------------------------------------------------------------- viewer JavaScript (Node, no browser)
def test_picker_contours_level_of_detail_and_culling(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    from src.registration_explorer import cell_review_viewer as crv
    sq = lambda x, y: [x - 2, y - 2, x + 2, y - 2, x + 2, y + 2, x - 2, y + 2]   # noqa: E731
    g = crv.contour_group("c", "c", {"idx": [0, 1, 2], "poly": [sq(10, 10), sq(50, 50), sq(500, 500)]}, "#fff", ids=[1, 2, 3],
                          at=[[10, 10], [50, 50], [500, 500]])
    harness = r"""
global.window = {parent: {postMessage: () => {}}, addEventListener: () => {}, __regx: {st: {view: {k: 1, ox: 0, oy: 0}}}};
global.sessionStorage = {getItem: () => null, setItem: () => {}};
__BOOT__
const S = {points: [], contours: [__G__], selected: {he: [10, 10], he_poly: [0, 0, 1, 0, 1, 1]}};
const A = makeAPI(S, 't');
let labels = 0, strokes = 0;
const ctx = {canvas: {width: 200, height: 200}, setTransform(){}, beginPath(){}, arc(){}, stroke(){ strokes++; }, lineTo(){}, moveTo(){}, closePath(){},
  save(){}, restore(){}, fillText(){ labels++; }, setLineDash(){}};
const out = {};
A.overlay(ctx, {k: 0.5, ox: 0, oy: 0}); out.far = A.drawn;                 // zoomed out: no outlines (LOD)
A.overlay(ctx, {k: 2, ox: 0, oy: 0}); out.near = A.drawn;                  // view 0..100 um: 2 of 3 outlines (culling)
A.opt.ids = true; labels = 0; A.overlay(ctx, {k: 5, ox: 0, oy: 0}); out.labels = labels;   // view 0..40: 1 label
A.opt.contours = false; A.overlay(ctx, {k: 2, ox: 0, oy: 0}); out.off = A.drawn;
console.log(JSON.stringify(out));
"""
    js = harness.replace("__BOOT__", crv._BOOT).replace("__G__", json.dumps(g))
    (tmp_path / "h.js").write_text(js)
    r = subprocess.run([node, str(tmp_path / "h.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip()) == {"far": 0, "near": 2, "labels": 1, "off": 0}


def test_compare_viewer_html(tmp_path):
    from src.registration_explorer import cell_compare_viewer as cmpv
    left = cmpv.pane("L", [0, 0, 10, 10], [cmpv.item("sel", [5, 5], [[4, 4], [6, 4], [6, 6]], "H&E 1"), cmpv.item("faint", [1, 1])])
    right = cmpv.pane("R", [0, 0, 10, 10], [cmpv.item("active", [5, 5], None, "#1 IF 2")])
    M = np.array([[0.0, -1.0], [1.0, 0.0]]) * 2.0
    sy = cmpv.sync_spec(M, [3.0, 4.0])
    assert np.allclose(np.asarray(sy["Ainv"]) @ M, np.eye(2)) and sy["scale"] == pytest.approx(2.0)
    h = cmpv.html(left, right, sy)
    assert "H&E 1" in h and "__DATA__" not in h and "__H__" not in h
    with pytest.raises(AssertionError):
        cmpv.item("bogus", [0, 0])
    node = shutil.which("node")
    if node:
        import re
        (tmp_path / "c.js").write_text(re.search(r"<script>(.*)</script>", h, re.S).group(1))
        r = subprocess.run([node, "--check", str(tmp_path / "c.js")], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


# ----------------------------------------------------------------------------- UI (Streamlit AppTest) on real data
def _app(sample, root):
    import os
    import sys
    sys.path.insert(0, "/mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu")
    os.environ["CELLREG_CELL_REVIEW_ROOT"] = root
    import streamlit as st
    from src.registration_explorer import catalog, results
    from src.registration_explorer import cell_review_ui as U
    from src.registration_explorer import cell_review_viewer as crv
    orig = getattr(crv.interactive, "_orig", crv.interactive)

    def cap(sc, *a, **k):
        st.session_state["_picker_scene"] = {"contours": sc.get("contours"), "selected": sc.get("selected")}
        return orig(sc, *a, **k)
    cap._orig = orig
    crv.interactive = cap
    S = catalog.load_manifest()[sample]
    avail = catalog.availability(S)
    recs = [r for r in results.load_records(S) if r.M is not None]
    rec = next((r for r in recs if r.method == "WF"), recs[0])
    st.session_state.setdefault("regx-crv-roisrc", "manual")
    try:
        U.render(S, avail, rec, "IF")
    finally:
        crv.interactive = orig


def _start(tmp_path, sample, half=110.0):
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
    at = AppTest.from_function(_app, kwargs={"sample": sample, "root": str(tmp_path)}, default_timeout=600)
    at.session_state["regx-crv-roisrc"] = "manual"
    at.run()
    assert not at.exception, at.exception
    for i, v in enumerate([c[0] - half, c[1] - half, c[0] + half, c[1] + half]):
        next(w for w in at.number_input if w.key.startswith(f"regx-crv-roi{i}-manual")).set_value(v)
    at.run()
    at.button(key="regx-crv-run").click().run()
    assert not at.exception, at.exception
    autos = CR.list_autos(sample, tmp_path)
    assert len(autos) == 1
    return at, CR.load_auto(autos[0]), autos[0], S


def _select(at, auto, h):
    at.session_state[f"regx-crv-selhe-{auto.auto_id}"] = int(h)
    at.run()
    assert not at.exception, at.exception


@pytest.mark.slow
def test_ui_contours_compare_and_review_states(tmp_path):
    """Real 031-6_2: contours in both viewers, the left/right comparison, all review states, save -> reload, export."""
    from src.registration_explorer import catalog
    from src.registration_research import cell_review as CR
    from src.registration_research import nuclear_contours as NC
    at, auto, auto_dir, S = _start(tmp_path, "031-6_2")
    sha0 = json.loads((auto_dir / "auto_meta.json").read_text())["files_sha256"]
    he, he_ids = catalog.load_nuclei(S.modality("HE"))
    iff, if_ids = catalog.load_nuclei(S.modality("IF"))
    pairs = auto.auto_pairs()
    assert len(pairs) >= 3
    h1, h2, h3 = list(pairs)[:3]
    _select(at, auto, h1)
    # main viewer: outlines of H&E (mapped) and IF, the selected nucleus highlighted with its own outline
    ps = at.session_state["_picker_scene"]
    assert ps["contours"] and any(g["id"] == "c_if" and g["poly"] for g in ps["contours"])
    Ma, ta = np.asarray(auto.meta["global"]["M"]), np.asarray(auto.meta["global"]["t"])
    assert NC._inside(np.asarray(ps["selected"]["he_poly"]).reshape(-1, 2), Ma @ he[h1] + ta)
    assert len(ps["selected"]["cand_polys"]) >= 1
    for g in ps["contours"]:
        if g["id"].startswith("c_he_") and g["poly"]:
            j, poly = g["idx"][0], g["poly"][0]
            assert NC._inside(np.asarray(poly).reshape(-1, 2), Ma @ he[j] + ta)
    # left/right comparison: undeformed crops in each section frame; outlines contain their own centres
    cmp = at.session_state["regx-crv-cmp-last"]
    assert cmp["he"] == h1 and cmp["active"] == pairs[h1]                 # default = current partner
    left, right = cmp["left"], cmp["right"]
    assert left["image"] is not None and right["image"] is not None
    sel = next(i for i in left["items"] if i["s"] == "sel")
    assert np.allclose(sel["c"], he[h1], atol=1e-3) and NC._inside(np.asarray(sel["poly"]).reshape(-1, 2), he[h1])
    act = next(i for i in right["items"] if i["s"] == "active")
    assert np.allclose(act["c"], iff[pairs[h1]], atol=1e-3) and NC._inside(np.asarray(act["poly"]).reshape(-1, 2), iff[pairs[h1]])
    m = left["image"]["m"]                                                   # crop placement: image covers the window around the nucleus
    assert m[4] <= he[h1][0] <= m[4] + m[0] * left["image"]["w"] and m[5] <= he[h1][1] <= m[5] + m[3] * left["image"]["h"]
    assert any(i["s"] == "faint" and "poly" in i for i in right["items"])   # neighbours shown
    # switch the compared candidate, then confirm it
    cands = auto.candidates[auto.candidates.he_index == h1]
    alt = int(cands.loc[cands["rank"] == 2, "if_index"].iloc[0]) if (cands["rank"] == 2).any() else None
    rk = next(w for w in at.radio if w.key.startswith(f"regx-crv-cmpcand-{h1}-"))
    if alt is not None:
        rk.set_value(alt).run()
        assert at.session_state["regx-crv-cmp-last"]["active"] == alt
        at.button(key="regx-crv-cmpok").click().run()
        rv = at.session_state[f"regx-crv-review-{auto.auto_id}"]
        assert rv.verdicts()[h1]["review_status"] == "confirmed_match" and rv.pairs()[h1] == alt
    else:
        at.button(key="regx-crv-cmpok").click().run()
    # Absent without a comment is refused, with a comment recorded (direction and scope explicit)
    _select(at, auto, h2)
    at.button(key="regx-crv-abs").click().run()
    assert any("根拠" in str(e.value) for e in at.error)
    rv = at.session_state[f"regx-crv-review-{auto.auto_id}"]
    assert rv.review_status(h2) == "unreviewed"
    at.text_input(key="regx-crv-reason").set_value("IFの対応位置に核が見えない").run()
    at.button(key="regx-crv-abs").click().run()
    v = at.session_state[f"regx-crv-review-{auto.auto_id}"].verdicts()[h2]
    assert v["review_status"] == "absent" and v["scope"]["absent_in"] == "IF" and v["scope"]["window_um"] == 60.0 and v["scope"]["if_image_used"]
    # Not identified, Uncertain with tags, then back to unreviewed
    at.text_input(key="regx-crv-reason").set_value("").run()
    _select(at, auto, h3)
    at.multiselect(key=f"regx-crv-tags-{h3}").set_value(["image_quality"]).run()
    at.button(key="regx-crv-unc").click().run()
    rv = at.session_state[f"regx-crv-review-{auto.auto_id}"]
    assert rv.verdicts()[h3]["review_status"] == "uncertain" and rv.verdicts()[h3]["tags"] == ["image_quality"]
    at.button(key="regx-crv-nid").click().run()
    assert at.session_state[f"regx-crv-review-{auto.auto_id}"].review_status(h3) == "not_identified"
    at.button(key="regx-crv-reset").click().run()
    assert at.session_state[f"regx-crv-review-{auto.auto_id}"].review_status(h3) == "unreviewed"
    at.button(key="regx-crv-undo").click().run()
    rv = at.session_state[f"regx-crv-review-{auto.auto_id}"]
    assert rv.review_status(h3) == "not_identified"
    # save -> reload in a fresh review session
    assert rv.context["segmentation"]["HE"]["sha256"] and rv.context["segmentation"]["IF"]["id_property"] == "segment_id"
    want = rv.verdicts()
    at.button(key="regx-crv-save").click().run()
    saved = CR.list_reviews("031-6_2", auto.auto_id, tmp_path)
    assert len(saved) == 1
    at.session_state[f"regx-crv-review-{auto.auto_id}"] = CR.Review.start(auto)
    at.run()
    at.selectbox(key=f"regx-crv-revsel-{auto.auto_id}").set_value(str(saved[0])).run()
    at.button(key="regx-crv-revopen").click().run()
    assert at.session_state[f"regx-crv-review-{auto.auto_id}"].verdicts() == want
    # the review table can be downloaded without computing any deformation
    assert any("レビュー結果の表" in w.proto.label for w in at.get("download_button"))
    # full export bundle (after the fields, as in the 11-17 flow): one row per nucleus, statuses kept apart from the algorithm
    at.button(key="regx-crv-fgo").click().run()
    at.button(key="regx-crv-export").click().run()
    assert not at.exception, at.exception
    ex = Path(at.session_state[f"regx-crv-export-{auto.auto_id}"])
    ann = pd.read_csv(ex / "review_annotations.csv")
    st_ = dict(zip(ann.source_index, ann.review_status))
    assert st_[h2] == "absent" and st_[h3] == "not_identified" and st_[h1] == "confirmed_match"
    assert (ann.source_contour_status == "ok").all() and ann.source_segmentation_sha256.notna().all()
    assert set(ann.loc[~ann.source_index.isin([h1, h2, h3]), "review_status"]) == {"unreviewed"}
    # the automatic estimate is untouched
    assert CR.load_auto(auto_dir).auto_pairs() == pairs
    assert json.loads((auto_dir / "auto_meta.json").read_text())["files_sha256"] == sha0


@pytest.mark.slow
def test_ui_points_only_sample_reviews_without_contours(tmp_path):
    """No segmentation polygons: the review works with centre points only."""
    at, auto, _, S = _start(tmp_path, "serial_section_pair_20260904", half=160.0)
    h = int(auto.table["he_index"].iloc[0])
    _select(at, auto, h)
    ps = at.session_state["_picker_scene"]
    assert all(not g["poly"] for g in ps["contours"]) and ps["selected"].get("he_poly") is None
    cmp = at.session_state["regx-crv-cmp-last"]
    assert cmp["left"]["image"] is None and all("poly" not in i for i in cmp["left"]["items"])
    at.button(key="regx-crv-nid").click().run()
    assert not at.exception, at.exception
    assert at.session_state[f"regx-crv-review-{auto.auto_id}"].review_status(h) == "not_identified"
