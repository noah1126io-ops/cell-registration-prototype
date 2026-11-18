"""Whole-cell segmentation + Cell Review integration (2026-11-18).  Numbers in brackets = the 12 required test items."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from conftest import REPO

STUDY = REPO / "analysis/whole-cell-segmentation-20261118"
MANIFEST = json.loads((STUDY / "baseline_manifest.json").read_text())
sys.path.insert(0, str(STUDY))


def _square(x, y, r):
    return [[x - r, y - r], [x + r, y - r], [x + r, y + r], [x - r, y + r], [x - r, y - r]]


def _write_result(base, sample, section, method, cells, version="v-test"):
    """cells: list of (cell_segment_id, ring, nucleus_ids, nucleus_indices, relation)."""
    d = base / method / sample
    d.mkdir(parents=True, exist_ok=True)
    feats = [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
              "properties": {"cell_segment_id": cid, "segmentation_method": method, "segmentation_version": version, "area_um2": 100.0,
                             "centroid_x": float(np.mean([p[0] for p in ring[:-1]])), "centroid_y": float(np.mean([p[1] for p in ring[:-1]])),
                             "nucleus_ids": nids, "nucleus_indices": nidx, "cell_relation": rel, "quality_flags": [], "nucleus_area_um2": 20.0}}
             for cid, ring, nids, nidx, rel in cells]
    (d / f"{section}.geojson").write_text(json.dumps({"type": "FeatureCollection", "properties": {
        "segmentation_method": method, "segmentation_version": version, "is_whole_cell_detection": True}, "features": feats}))
    return d / f"{section}.geojson"


# ----------------------------------------------------------------------------- [1][11] existing code unchanged
def test_protected_files_unchanged():
    """[1][11] nuclear contours module, Workflow C/D, engine, Graph/UOT/B2, viewer/scene/imaging/catalog and manifest: byte-identical."""
    from src.registration_research import legacy
    changed = [rel for rel, h in MANIFEST["files_sha256_at_head"].items() if legacy.sha256_file(REPO / rel) != h]
    assert changed == [], changed
    assert "src/registration_research/nuclear_contours.py" in MANIFEST["files_sha256_at_head"]


def _head_module(rel, name):
    src = subprocess.run(["git", "-C", str(REPO), "show", f"{MANIFEST['head']}:{rel}"], capture_output=True, text=True, check=True).stdout
    tmp = Path(STUDY / "cache" / "head_modules")
    tmp.mkdir(parents=True, exist_ok=True)
    p = tmp / f"{name}.py"
    p.write_text(src)
    spec = importlib.util.spec_from_file_location(f"src.registration_research.{name}", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_auto_fields_and_warp_identical_to_head(syn):
    """[11] Graph/UOT/B2 auto estimate, RBF/field computation and the image warp give identical numbers before/after this work."""
    from src.registration_research import cell_review as CR
    OLD = _head_module("src/registration_research/cell_review.py", "_cell_review_head")
    a_new = CR.compute_auto(syn["he_pts"], syn["he_ids"], syn["if_pts"], syn["if_ids"], syn["M"], syn["t"], roi=syn["roi"], meta={"sample_id": "SYN"})
    a_old = OLD.compute_auto(syn["he_pts"], syn["he_ids"], syn["if_pts"], syn["if_ids"], syn["M"], syn["t"], roi=syn["roi"], meta={"sample_id": "SYN"})
    pd.testing.assert_frame_equal(a_new.table, a_old.table)
    pd.testing.assert_frame_equal(a_new.candidates, a_old.candidates)
    assert a_new.auto_pairs() == a_old.auto_pairs()
    f_new = CR.compute_fields(a_new, None, syn["he_pts"], syn["if_pts"])
    f_old = OLD.compute_fields(a_old, None, syn["he_pts"], syn["if_pts"])
    assert set(f_new) == set(f_old)
    for k in f_new:
        assert f_new[k].applied == f_old[k].applied
        assert np.array_equal(f_new[k].field_x, f_old[k].field_x) and np.array_equal(f_new[k].field_y, f_old[k].field_y)
    img = (np.arange(400 * 400 * 3) % 251).astype(np.uint8).reshape(400, 400, 3)
    rd = {"read": lambda lvl, x0, y0, x1, y1: img[y0:y1, x0:x1], "level_shapes": [img.shape[:2]], "downsamples": [1.0], "pixel_um": 1.0}
    fr = next(iter(f_new.values()))
    w_new = CR.warp_he_roi(rd, syn["M"], syn["t"], syn["roi"], 2.0, fr)
    w_old = OLD.warp_he_roi(rd, syn["M"], syn["t"], syn["roi"], 2.0, f_old[fr.method])
    assert np.array_equal(np.asarray(w_new[0]), np.asarray(w_old[0]))


# ----------------------------------------------------------------------------- [2] loading + coordinates, [3] ID tracking
def test_cell_contours_load_coordinates_and_query(tmp_path):
    """[2] polygons are read in the section frame (um), bbox query and affine packing are exact."""
    from src.registration_research import cell_segmentation as CS
    ids = np.array([10, 11, 12])
    _write_result(tmp_path, "S", "IF", "S1_cpsam_if_mem_full", [("full:1", _square(10, 10, 5), ["10"], [0], "one_nucleus"),
                                                                 ("full:2", _square(40, 10, 5), [], [], "cell_without_nucleus")])
    g = CS.load("S", "IF", "S1_cpsam_if_mem_full", ids, base=tmp_path)
    assert g.n == 2 and g.version == "v-test" and g.method == "S1_cpsam_if_mem_full"
    assert np.allclose(g.polygon(0), np.array(_square(10, 10, 5))[:-1])
    assert list(g.query([0, 0, 20, 20])) == [0]
    M, t = np.array([[0.0, -1.0], [1.0, 0.0]]), np.array([100.0, 0.0])
    pk = g.pack([1], M, t)
    assert np.allclose(np.asarray(pk["poly"][0]).reshape(-1, 2), np.array(_square(40, 10, 5))[:-1] @ M.T + t)
    mapped_box = [85, 30, 95, 50]                                       # cell 2 after the affine
    assert list(g.query(mapped_box, M, t)) == [1]
    assert g.cells_for(0) == [0] and g.cells_for(1) == [] and g.cells_for(2) == []
    assert CS.identifier(g)["sha256"] and CS.identifier(None) is None


def test_nucleus_ids_tracked_not_renumbered(tmp_path):
    """[3] nuclei are re-joined by original ID; a reordered nucleus list is detected and still joined correctly."""
    from src.registration_research import cell_segmentation as CS
    _write_result(tmp_path, "S", "IF", "M1", [("r:1", _square(0, 0, 3), ["a7", "b9"], [0, 1], "multi_nucleus")])
    g = CS.load("S", "IF", "M1", np.array(["a7", "b9", "c1"]), base=tmp_path)
    assert g.report["nucleus_id_mismatch"] == 0 and list(g.nuclei_of_cell[0]) == [0, 1]
    g2 = CS.load("S", "IF", "M1", np.array(["c1", "b9", "a7"]), base=tmp_path)       # same IDs, other order (a7 moved)
    assert g2.report["nucleus_id_mismatch"] == 1 and sorted(g2.nuclei_of_cell[0]) == [1, 2]
    g3 = CS.load("S", "IF", "M1", np.array(["zz"]), base=tmp_path)                    # ID gone -> no relation (never guessed)
    assert g3.report["nucleus_id_unknown"] == 2 and len(g3.nuclei_of_cell[0]) == 0


def test_real_full_section_cells_relate_to_existing_nucleus_ids():
    """[2][3] real S1_cpsam_if_mem_full: the cell of an 'assigned' nucleus contains the nucleus centre (same frame); IDs join exactly."""
    from src.registration_explorer import catalog
    from src.registration_research import cell_segmentation as CS
    from src.registration_research import nuclear_contours as NC
    man = catalog.load_manifest()
    for s in ("031-6_2", "031-8_6"):
        pts, ids = catalog.load_nuclei(man[s].modality("IF"))
        g = CS.load(s, "IF", "S1_cpsam_if_mem_full", ids)
        assert g is not None and g.report["nucleus_id_mismatch"] == 0 and g.report["nucleus_id_unknown"] == 0
        tab = pd.read_csv(CS.root() / "S1_cpsam_if_mem_full" / s / "IF_nucleus_cell.csv")
        asg = tab[tab.relation == "assigned"].head(300)
        inside = [NC._inside(g.polygon(g.ids.index(c)), pts[int(i)]) for i, c in zip(asg.nucleus_index, asg.cell_segment_id)]
        assert np.mean(inside) > 0.95, np.mean(inside)
        assert {str(v) for v in tab.nucleus_id} <= {NC._key(v).strip("'") for v in ids}


# ----------------------------------------------------------------------------- [4] relations are not forced to 1:1
def test_association_cases():
    """[4] multi-nucleus cell, cell without nucleus, nucleus without cell, nucleus outside its cell, nucleus split between cells."""
    import evaluate as E
    H = W = 60
    lab = np.zeros((H, W), np.int32)
    lab[5:25, 5:25] = 1                       # cell 1: two nuclei (A, B)
    lab[5:25, 35:55] = 2                      # cell 2: no nucleus
    lab[35:55, 5:25] = 3                      # cells 3 | 4 split nucleus D
    lab[35:55, 25:45] = 4
    nuc = np.zeros((H, W), np.int32)
    centres = {0: (10, 10), 1: (18, 18), 2: (57, 30), 3: (25, 45), 4: (30, 30)}   # (x, y) px: A, B in cell 1; C, E in background;
    for i, (x, y) in centres.items():                                               # D straddles cells 3 | 4
        nuc[y - 2:y + 3, x - 2:x + 3] = i + 1
    c = SimpleNamespace(nuc_idx=np.arange(5), nuc_lab=nuc, nuc_pts=np.array([centres[i] for i in range(5)], float) + 0.5,
                        nuc_ids=np.array(["A", "B", "C", "D", "E"]), origin=np.zeros(2), um=1.0, w=W, h=H)
    cells, rows = E.associate(c, lab, "T")
    rel = {r["nucleus_id"]: r["relation"] for r in rows}
    assert rel["A"] == "assigned" and rel["B"] == "assigned"
    assert rel["D"] == "nucleus_competing_cells"
    assert rel["C"] == "nucleus_without_cell" and rel["E"] == "nucleus_without_cell"
    cr = {x["label"]: x["relation"] for x in cells}
    assert cr[1] == "multi_nucleus" and cr[2] == "cell_without_nucleus"
    # outside: centre in background, but the nucleus overlaps a cell
    nuc2 = np.zeros((H, W), np.int32)
    nuc2[24:29, 8:13] = 1                     # rows 24..28: overlaps cell 1 (rows <25) only at its edge, centre (10, 26) outside
    c2 = SimpleNamespace(nuc_idx=np.arange(1), nuc_lab=nuc2, nuc_pts=np.array([[10.5, 26.5]]), nuc_ids=np.array(["F"]), origin=np.zeros(2),
                         um=1.0, w=W, h=H)
    _, rows2 = E.associate(c2, lab, "T")
    assert rows2[0]["relation"] == "nucleus_outside_cell" and rows2[0]["nucleus_inside_cell"] is False


# ----------------------------------------------------------------------------- [5] points-only sample, availability reasons
def test_points_only_sample_and_unavailable_methods(tmp_path):
    """[5] a sample without any cell result: every method is unavailable with a reason and load() returns None (review unaffected)."""
    from src.registration_research import cell_segmentation as CS
    ms = CS.methods_for("NO_SUCH_SAMPLE", "IF")
    assert {m for m, a, _ in ms if a} == {"none", "S0_nuclei"}
    assert all(r for m, a, r in ms if not a)
    assert CS.load("NO_SUCH_SAMPLE", "IF", "S1_cpsam_if_mem_full", np.array([1])) is None
    a, r = CS.availability("031-8_6", "HE", "S1_cpsam_if_mem_full")
    assert not a and "IF" in r
    a, r = CS.availability("031-8_6", "IF", "S2_mesmer")
    assert not a and "DEEPCELL_ACCESS_TOKEN" in r


# ----------------------------------------------------------------------------- [8][9][10] review record
def test_review_cell_record_saved_and_old_formats_load(syn_auto, tmp_path):
    """[8] method/version/cell IDs/judgement saved with the verdict (schema /3); [9] /1 and /2 files still load."""
    from src.registration_research import cell_review as CR
    rv = CR.Review.start(syn_auto)
    h = next(iter(syn_auto.auto_pairs()))
    cell = {"source": None, "target": {"method": "S1_cpsam_if_mem_full", "version": "wcs-20261118.1", "sha256": "x", "cell_ids": ["full:5"]},
            "reliability": "partly", "flags": ["boundary_unclear"], "comment": "膜が薄い"}
    rv.apply("support", h, user="t", cell=cell)
    v = rv.verdicts()[h]
    assert v["cell"]["target"]["method"] == "S1_cpsam_if_mem_full" and v["cell"]["target"]["cell_ids"] == ["full:5"]
    assert v["review_status"] == "confirmed_match" and rv.pairs()[h] == syn_auto.auto_pairs()[h]   # the cell record does not change the pair
    with pytest.raises(CR.ReviewError):
        rv.apply("uncertain", h, user="t", cell={"reliability": "bogus"})
    d = rv.to_json()
    assert d["schema"] == "cellreg-cell-review-edits/3"
    rv2 = CR.Review.from_json(json.loads(json.dumps(d, default=CR._json_default)))
    assert rv2.verdicts()[h]["cell"] == v["cell"]
    ann = CR.review_annotations("SYN", syn_auto, rv2, np.arange(syn_auto.meta["inputs"]["n_he"]), np.arange(syn_auto.meta["inputs"]["n_if"]))
    row = ann[ann.source_index == h].iloc[0]
    assert row.cell_method_target == "S1_cpsam_if_mem_full" and row.cell_ids_target == "full:5" and bool(row.cell_boundary_unclear)
    assert row.cell_contour_reliability == "partly" and row.cell_comment == "膜が薄い"
    # without cell records the file stays /2, and /1 and /2 load; cell columns empty
    rv3 = CR.Review.start(syn_auto)
    rv3.apply("support", h, user="t")
    d3 = rv3.to_json()
    assert d3["schema"] == "cellreg-cell-review-edits/2"
    for sch in ("cellreg-cell-review-edits/1", "cellreg-cell-review-edits/2"):
        d3["schema"] = sch
        r = CR.Review.from_json(json.loads(json.dumps(d3, default=CR._json_default)))
        assert r.verdicts()[h]["cell"] is None
        a = CR.review_annotations("SYN", syn_auto, r, np.arange(syn_auto.meta["inputs"]["n_he"]), np.arange(syn_auto.meta["inputs"]["n_if"]))
        assert a[a.source_index == h].iloc[0].cell_method_target is None


# ----------------------------------------------------------------------------- [6][7] viewers (JS)
def test_picker_cell_groups_toggle_independently(tmp_path):
    """[6][7] nucleus and cell outlines are separate groups with separate toggles; cells are dashed and drawn at lower zoom."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available (compute node)")
    from src.registration_explorer import cell_review_viewer as crv
    sq = lambda x, y, r: [x - r, y - r, x + r, y - r, x + r, y + r, x - r, y + r]   # noqa: E731
    gn = crv.contour_group("n", "n", {"idx": [0], "poly": [sq(10, 10, 2)]}, "#fff", ids=[1], at=[[10, 10]])
    gc = crv.contour_group("c", "c", {"idx": [0], "poly": [sq(10, 10, 8)]}, "#f80", ids=["full:1"], at=[[10, 10]], kind="cell", dash=[8, 4])
    assert gc["kind"] == "cell" and gc["dash"] == [8, 4] and gn["kind"] == "nucleus"
    harness = r"""
global.window = {parent: {postMessage: () => {}}, addEventListener: () => {}, __regx: {st: {view: {k: 1, ox: 0, oy: 0}}}};
global.sessionStorage = {getItem: () => null, setItem: () => {}};
__BOOT__
const S = {points: [], contours: [__GN__, __GC__], selected: {if_cell_polys: [[0, 0, 1, 0, 1, 1]]}};
const A = makeAPI(S, 't');
let dashes = [];
const ctx = {canvas: {width: 200, height: 200}, setTransform(){}, beginPath(){}, arc(){}, stroke(){}, lineTo(){}, moveTo(){}, closePath(){},
  save(){}, restore(){}, fillText(){}, setLineDash(d){ dashes.push(d.length); }};
const out = {};
A.overlay(ctx, {k: 0.8, ox: 0, oy: 0}); out.lowzoom = A.drawn;           // only the cell outline (cell_min_k 0.6 < 0.8 < 1.2)
A.overlay(ctx, {k: 2, ox: 0, oy: 0}); out.both = A.drawn;
A.opt.contours = false; A.overlay(ctx, {k: 2, ox: 0, oy: 0}); out.cells_only = A.drawn;
A.opt.contours = true; A.opt.cells = false; A.overlay(ctx, {k: 2, ox: 0, oy: 0}); out.nuclei_only = A.drawn;
out.dashed = dashes.some(n => n > 0);
console.log(JSON.stringify(out));
"""
    js = harness.replace("__BOOT__", crv._BOOT).replace("__GN__", json.dumps(gn)).replace("__GC__", json.dumps(gc))
    (tmp_path / "h.js").write_text(js)
    r = subprocess.run([node, str(tmp_path / "h.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip()) == {"lowzoom": 1, "both": 2, "cells_only": 1, "nuclei_only": 1, "dashed": True}


def test_compare_viewer_cell_items(tmp_path):
    """[6] the left/right viewer carries cell items (own toggles, dashed styles) only when cells are shown."""
    from src.registration_explorer import cell_compare_viewer as cmpv
    left = cmpv.pane("L", [0, 0, 10, 10], [cmpv.item("sel", [5, 5], [[4, 4], [6, 4], [6, 6]], "H&E 1")])
    right = cmpv.pane("R", [0, 0, 10, 10], [cmpv.item("cell_sel", [5, 5], [[1, 1], [9, 1], [9, 9]], "7", kind="cell"),
                                            cmpv.item("active", [5, 5], None, "#1 IF 2")])
    h = cmpv.html(left, right)
    esc = lambda x: json.dumps(x)[1:-1]                                       # noqa: E731  (data is JSON-escaped)
    assert "細胞輪郭（破線）" in h and '"k": "cell"' in h and esc("核を含む細胞") in h
    assert cmpv.STYLES["cell_sel"][2] and cmpv.STYLES["cell"][2]                  # dashed
    h0 = cmpv.html(left, cmpv.pane("R", [0, 0, 10, 10], [cmpv.item("active", [5, 5])]))
    assert esc("核を含む細胞") not in h0
    with pytest.raises(AssertionError):
        cmpv.item("sel", [0, 0], kind="cell")
    node = shutil.which("node")
    if node:
        import re
        (tmp_path / "c.js").write_text(re.search(r"<script>(.*)</script>", h, re.S).group(1))
        r = subprocess.run([node, "--check", str(tmp_path / "c.js")], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


# ----------------------------------------------------------------------------- [6][7][8][12] UI on real data (AppTest)
def _cra():
    spec = importlib.util.spec_from_file_location("cra_tests", REPO / "analysis/cell-review-annotation-20261118/tests_cra/test_annotation.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.slow
def test_ui_cell_outlines_compare_switch_and_save(tmp_path):
    """Real 031-8_6: default off; IF cells selectable; picker groups + selected-cell highlight; compare panes; method switch;
    unavailable method shows its reason; the verdict saves method/version/cell IDs.  (Rendering in a real browser: not verified here.)"""
    from src.registration_explorer import catalog
    from src.registration_research import cell_review as CR
    from src.registration_research import cell_segmentation as CS
    from src.registration_research import nuclear_contours as NC
    cra = _cra()
    at, auto, auto_dir, S = cra._start(tmp_path, "031-8_6")
    iff, if_ids = catalog.load_nuclei(S.modality("IF"))
    g = CS.load("031-8_6", "IF", "S1_cpsam_if_mem_full", if_ids)
    # 031-8_6 (adjacent sections) may have no automatic pair in the ROI: use the rank-1 candidate, compared by default
    c1 = auto.candidates[auto.candidates["rank"] == 1]
    h, f1 = next((int(h), int(f)) for h, f in zip(c1.he_index, c1.if_index) if len(g.cells_for(f)) == 1)
    cra._select(at, auto, h)
    ps = at.session_state["_picker_scene"]
    assert not any(x.get("kind") == "cell" for x in ps["contours"])                 # default: no cell outlines
    assert at.session_state["regx-crv-cellm-IF"] == "none"
    at.selectbox(key="regx-crv-cellm-IF").set_value("S1_cpsam_if_mem_full").run()
    assert not at.exception, at.exception
    ps = at.session_state["_picker_scene"]
    cg = [x for x in ps["contours"] if x.get("kind") == "cell"]
    assert len(cg) == 1 and cg[0]["dash"] and cg[0]["poly"]
    nuc_groups = [x for x in ps["contours"] if x.get("kind") == "nucleus"]
    assert nuc_groups and all("dash" not in x for x in nuc_groups)                    # nuclei unchanged (solid)
    cmp = at.session_state["regx-crv-cmp-last"]
    f = cmp["active"]
    assert f == f1
    cs = [i for i in cmp["right"]["items"] if i.get("k") == "cell" and i["s"] == "cell_sel"]
    assert len(cs) == 1 and NC._inside(np.asarray(cs[0]["poly"]).reshape(-1, 2), iff[f])   # right pane, IF frame, contains the IF nucleus
    assert not any(i.get("k") == "cell" for i in cmp["left"]["items"])               # H&E: none selected
    # switch method -> other groups
    at.selectbox(key="regx-crv-cellm-IF").set_value("S3_voronoi").run()
    cg2 = [x for x in at.session_state["_picker_scene"]["contours"] if x.get("kind") == "cell"]
    assert len(cg2) == 1 and cg2[0]["id"].endswith("S3_voronoi")
    at.selectbox(key="regx-crv-cellm-IF").set_value("S2_mesmer").run()
    assert not at.exception
    assert any("DEEPCELL_ACCESS_TOKEN" in w.value for w in at.warning)
    assert not any(x.get("kind") == "cell" for x in at.session_state["_picker_scene"]["contours"])
    at.selectbox(key="regx-crv-cellm-IF").set_value("S1_cpsam_if_mem_full").run()
    # record a verdict with the cell judgement and save
    at.selectbox(key=f"regx-crv-cellrel-{h}").set_value("reliable").run()
    at.button(key="regx-crv-cmpok").click().run()
    assert not at.exception, at.exception
    at.button(key="regx-crv-save").click().run()
    revs = CR.list_reviews("031-8_6", auto.auto_id, tmp_path)
    d = json.loads(Path(revs[-1]).read_text())
    assert d["schema"] == "cellreg-cell-review-edits/3"
    cell = d["verdicts"][str(h)]["cell"]
    assert cell["target"]["method"] == "S1_cpsam_if_mem_full" and cell["target"]["version"] == g.version
    assert cell["target"]["cell_ids"] == [g.ids[j] for j in g.cells_for(f1)] and cell["reliability"] == "reliable"
    assert d["verdicts"][str(h)]["if_index"] == f1
    assert json.loads((auto_dir / "auto_meta.json").read_text())["files_sha256"] == CR.load_auto(auto_dir).meta["files_sha256"]


@pytest.mark.slow
def test_ui_review_works_without_cell_outlines(tmp_path):
    """[5][12] 031-6_2 with no cell method selected: the existing nucleus review (contours, compare, verdict) is unchanged; /2 file."""
    from src.registration_research import cell_review as CR
    cra = _cra()
    at, auto, auto_dir, S = cra._start(tmp_path, "031-6_2")
    h = next(iter(auto.auto_pairs()))
    cra._select(at, auto, h)
    assert not any(x.get("kind") == "cell" for x in at.session_state["_picker_scene"]["contours"])
    assert not any(i.get("k") == "cell" for p in ("left", "right") for i in at.session_state["regx-crv-cmp-last"][p]["items"])
    at.button(key="regx-crv-cmpok").click().run()
    at.button(key="regx-crv-save").click().run()
    assert not at.exception, at.exception
    d = json.loads(Path(CR.list_reviews("031-6_2", auto.auto_id, tmp_path)[-1]).read_text())
    assert d["schema"] == "cellreg-cell-review-edits/2" and d["verdicts"][str(h)]["cell"] is None
