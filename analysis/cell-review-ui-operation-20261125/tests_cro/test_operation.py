"""Cell Review operation (2026-11-25): unified click / drag / focus, quick judgement panel, keyboard shortcuts.

JS parts run in Node with a minimal fake window (no browser); the Streamlit flow runs in AppTest on real 031-6_2.
Real-browser checks (pointer, wheel, devicePixelRatio, focus, tabs) are in browser/e2e.py (headless Chromium).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((HERE / "baseline_manifest.json").read_text())


# ----------------------------------------------------------------------------- unchanged code / formats
def test_protected_files_unchanged():
    """Graph/UOT/B2, correspondence, RBF / UOT conservative, segmentation, warp, Workflow C/D, viewer.py, cell_review.py: byte-identical."""
    bad = [f for f, h in MANIFEST["protected"].items() if hashlib.sha256((REPO / f).read_bytes()).hexdigest() != h]
    assert not bad, bad
    for f in ("src/registration_research/cell_review.py", "src/registration_explorer/viewer.py", "src/registration_research/cell_segmentation.py",
              "src/registration_research/nuclear_contours.py"):
        assert f in MANIFEST["protected"]
    assert len(MANIFEST["protected"]) >= 60


# ----------------------------------------------------------------------------- viewer JavaScript (Node)
HARNESS = r"""
const sent = [];
const store = {};
global.window = {devicePixelRatio: __DPR__, parent: {postMessage: (m) => sent.push(m)}, addEventListener: () => {},
                 __regx: {st: {view: {k: 1.0, ox: 0, oy: 0}, mode: 'overlay', alpha: 0.5}}};
global.sessionStorage = {getItem: (k) => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = String(v); }};
global.document = {getElementById: () => ({width: 1200, height: 600})};
__BOOT__
const ev = () => sent.filter(m => m.type === 'streamlit:setComponentValue').map(m => m.value);
const last = () => { const e = ev(); return e.length ? e[e.length - 1] : null; };
const out = {};
__BODY__
console.log(JSON.stringify(out));
"""


def _run_js(tmp_path, body, dpr=1.0):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available (compute node)")
    from src.registration_explorer import cell_review_viewer as crv
    js = HARNESS.replace("__BOOT__", crv._BOOT).replace("__BODY__", body).replace("__DPR__", str(dpr))
    (tmp_path / "h.js").write_text(js)
    r = subprocess.run([node, str(tmp_path / "h.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def _scene():
    from src.registration_explorer import cell_review_viewer as crv
    he = crv.pick_points_layer("he", "HE", np.array([[100.0, 100.0], [140.0, 100.0], [300.0, 300.0]]), [7, 42, 9], "he", "#f00")
    iff = crv.pick_points_layer("if", "IF", np.array([[104.0, 100.0], [100.0, 112.0], [500.0, 500.0]]), [3, 4, 99], "if", "#0f0")
    return {"points": [he, iff], "polys": [], "images": [], "contours": [], "view_key": "vk",
            "cand_table": {"7": [[3, 104.0, 100.0, 1], [4, 100.0, 112.0, 2]], "42": [[4, 100.0, 112.0, 1]]},
            "partner": {"7": 4}, "he_ids": {"7": "HE-7", "42": "HE-42", "9": "HE-9"}}


def _body(scene, code):
    return f"const S = {json.dumps(scene)}; const A = makeAPI(S, 't'); A.redraw = () => {{}};\n" + code


def test_click_selects_nearest_he_without_mode_and_focuses(tmp_path):
    """[1][2] review mode: a click picks the nearest H&E nucleus (global index), centres it and zooms to ~60 µm the first
    time; the next click keeps the user's zoom.  Tolerance is in CSS px (x dpr), on a dpr-2 display too."""
    for dpr in (1.0, 2.0):
        code = """
out.mode = A.mode;
A.down([141, 101], {button: 0, clientX: 50, clientY: 50}); A.up([141, 101], {clientX: 51, clientY: 50});    // 1 CSS px jitter = click
out.e1 = last(); out.sel1 = A.selHe();
const v = window.__regx.st.view; out.k1 = v.k; out.c1 = [(1200/2 - v.ox)/v.k, (600/2 - v.oy)/v.k];
v.k = 25; A.centreOn([0, 0], 25);                                     // the user zooms in
A.down([100.2, 100.1], {button: 0, clientX: 10, clientY: 10}); A.up([100.2, 100.1], {clientX: 10, clientY: 10});
out.e2 = last(); out.k2 = window.__regx.st.view.k; out.active2 = A.sel.active;
out.saved = JSON.parse(sessionStorage.getItem('regx-view-vk'));
// tolerance: 12 CSS px at k = 25 device px/µm -> 12*dpr/25 µm
const tol = 12 * window.devicePixelRatio / 25; const n = ev().length;
A.down([300 + tol * 0.9, 300], {button: 0, clientX: 0, clientY: 0}); A.up([300 + tol * 0.9, 300], {clientX: 0, clientY: 0}); out.in_tol = ev().length - n;
A.down([300 + tol * 1.2, 300], {button: 0, clientX: 0, clientY: 0}); A.up([300 + tol * 1.2, 300], {clientX: 0, clientY: 0}); out.out_tol = ev().length - n - 1;
"""
        o = _run_js(tmp_path, _body(_scene(), code), dpr)
        assert o["mode"] == "review"
        assert [o["e1"]["kind"], o["e1"]["role"], o["e1"]["index"]] == ["nucleus", "he", 42] and o["sel1"] == 42
        assert abs(o["k1"] - 600 / 60.0) < 1e-9 and np.allclose(o["c1"], [140, 100])             # ~60 µm across the shorter side
        assert [o["e2"]["kind"], o["e2"]["index"]] == ["nucleus", 7] and o["k2"] == 25            # user's zoom kept
        assert o["active2"] == 4                                                                   # default = current partner (rank 2)
        assert abs(o["saved"]["cx"] - 100) < 1e-6 and o["saved"]["k"] == 25                        # restored by the template on re-render
        assert o["in_tol"] == 1 and o["out_tol"] == 0, (dpr, o)


def test_drag_pans_and_never_selects(tmp_path):
    """[3] a press moved by more than 5 CSS px pans (offset = travel x dpr) and its release selects nothing."""
    code = """
const v0 = Object.assign({}, window.__regx.st.view);
A.down([100, 100], {button: 0, clientX: 200, clientY: 200});
out.m1 = A.move([0, 0], {clientX: 203, clientY: 202});                // within 5 px: not yet a drag
out.m2 = A.move([0, 0], {clientX: 260, clientY: 230});
A.up([100, 100], {clientX: 260, clientY: 230});
const v = window.__regx.st.view; out.d = [v.ox - v0.ox, v.oy - v0.oy]; out.events = ev().length; out.sel = A.selHe();
A.down([100, 100], {button: 2, clientX: 0, clientY: 0}); A.up([100, 100], {button: 2, clientX: 0, clientY: 0}); out.right = ev().length;
"""
    for dpr in (1.0, 2.0):
        o = _run_js(tmp_path, _body(_scene(), code), dpr)
        assert o["m1"] is False and o["m2"] is True
        assert np.allclose(o["d"], [60 * dpr, 30 * dpr]) and o["events"] == 0 and o["sel"] is None
        assert o["right"] == 0                                                                     # right button: nothing


def test_candidate_ring_and_shift_click(tmp_path):
    """[7] with H&E 7 selected, a click on its candidate ring switches the compared candidate; shift+click picks any IF nucleus."""
    code = """
A.setSel({he_index: 7, he: [100, 100], candidates: [[104, 100], [100, 112]], cand_index: [3, 4], cand_rank: [1, 2], active: 4});
window.__regx.st.view = {k: 20, ox: 0, oy: 0};
A.down([100.3, 111.6], {button: 0, clientX: 0, clientY: 0}); A.up([100.3, 111.6], {clientX: 0, clientY: 0}); out.c = last(); out.a1 = A.sel.active;
A.down([104.2, 100.1], {button: 0, clientX: 0, clientY: 0}); A.up([104.2, 100.1], {clientX: 0, clientY: 0}); out.c2 = last(); out.a2 = A.sel.active;
A.down([100.2, 100.1], {button: 0, clientX: 0, clientY: 0}); A.up([100.2, 100.1], {clientX: 0, clientY: 0}); out.h = last();   // the H&E nucleus itself
A.down([500.2, 500], {button: 0, shiftKey: true, clientX: 0, clientY: 0}); A.up([500.2, 500], {shiftKey: true, clientX: 0, clientY: 0}); out.s = last();
"""
    o = _run_js(tmp_path, _body(_scene(), code))
    assert (o["c"]["kind"], o["c"]["he"], o["c"]["index"], o["a1"]) == ("candidate", 7, 4, 4)
    assert (o["c2"]["kind"], o["c2"]["index"], o["a2"]) == ("candidate", 3, 3)
    assert (o["h"]["kind"], o["h"]["role"], o["h"]["index"]) == ("nucleus", "he", 7)
    assert (o["s"]["kind"], o["s"]["role"], o["s"]["index"]) == ("nucleus", "if", 99)


def test_keyboard_shortcuts_and_typing(tmp_path):
    """[5][8][9] 1-4 / Enter / Ctrl+Z / Ctrl+Y / Esc carry the selected nucleus and candidate; typing in text inputs is ignored
    (checkboxes / radios are not text); digits without a selection send nothing."""
    code = """
const k = (key, o) => A.key(Object.assign({key: key, target: {tagName: 'CANVAS'}, preventDefault(){}, stopPropagation(){}}, o || {}));
out.none = [k('1'), ev().length];                                     // no selection: handled (message) but nothing sent
A.setSel({he_index: 7, he: [100, 100], active: 3});
const seq = [];
for (const key of ['1', '2', '3', '4']) { k(key); seq.push([last().action, last().he, last().active]); }
k('Enter'); seq.push([last().action, last().he]); k('z', {ctrlKey: true}); seq.push([last().action]); k('y', {ctrlKey: true}); seq.push([last().action]);
k('z', {ctrlKey: true, shiftKey: true}); seq.push([last().action]);
out.seq = seq; const n = ev().length;
out.typed = [k('1', {target: {tagName: 'INPUT', type: 'text'}}), k('Enter', {target: {tagName: 'TEXTAREA'}}), k('2', {target: {tagName: 'INPUT', type: 'number'}}),
             k('3', {target: {tagName: 'DIV', isContentEditable: true}}), k('1', {altKey: true}), k('a')];
out.after_typed = ev().length - n;
out.checkbox = k('2', {target: {tagName: 'INPUT', type: 'checkbox'}});
k('Escape'); out.esc = [last().kind, A.selHe()];
"""
    o = _run_js(tmp_path, _body(_scene(), code))
    assert o["none"] == [True, 0]
    assert o["seq"] == [["confirm", 7, 3], ["not_identified", 7, 3], ["absent", 7, 3], ["uncertain", 7, 3], ["next", 7], ["undo"], ["redo"], ["redo"]]
    assert o["typed"] == [False] * 6 and o["after_typed"] == 0
    assert o["checkbox"] is True
    assert o["esc"] == ["deselect", None]


def test_selection_from_python_focuses_only_on_change_and_toggles_persist(tmp_path):
    """[2][6][11] a new nucleus from Python (Enter / typed ID / advance) is centred; the same one is not re-centred (the user's
    pan is kept); layer toggles and outline options survive a re-render (new makeAPI on the same key)."""
    code = """
window.__regx.st.view = {k: 30, ox: 0, oy: 0};
A.setSel({he_index: 9, he: [300, 300]}); const v = window.__regx.st.view; out.c = [(600 - v.ox)/v.k, (300 - v.oy)/v.k, v.k];
v.ox += 123; A.setSel({he_index: 9, he: [300, 300], active: 99}); out.ox_kept = window.__regx.st.view.ox === v.ox;
S.points[1].visible = false; A.opt.cells = false; sessionStorage.setItem('crv-topt', JSON.stringify(A.opt));
const ctx = {canvas: {width: 1200, height: 600}, setTransform(){}, beginPath(){}, arc(){}, stroke(){}, lineTo(){}, moveTo(){}, closePath(){},
  save(){}, restore(){}, fillText(){}, setLineDash(){}};
A.overlay(ctx, window.__regx.st.view);
const S2 = JSON.parse(JSON.stringify(S)); S2.points[1].visible = true; const B = makeAPI(S2, 't');
out.vis = S2.points.map(L => L.visible); out.cells = B.opt.cells; out.focus_he = B.focusHe;
"""
    o = _run_js(tmp_path, _body(_scene(), code))
    assert np.allclose(o["c"], [300, 300, 30])                                                   # zoomed in already: k kept
    assert o["ox_kept"] is True
    assert o["vis"] == [True, False] and o["cells"] is False and o["focus_he"] == 9


def test_interactive_sends_selection_apart_from_scene(monkeypatch):
    """[6] the scene hash ignores the selection (selection-only changes redraw in place); view_key is in the scene."""
    from src.registration_explorer import cell_review_viewer as crv
    calls = []
    monkeypatch.setattr(crv, "_component", lambda **kw: calls.append(kw))
    sc = {"points": [], "selected": {"he_index": 1}}
    crv.interactive(sc, 600, key="k", view_key="vk")
    crv.interactive({**sc, "selected": {"he_index": 2}}, 600, key="k", view_key="vk")
    assert calls[0]["scene_id"] == calls[1]["scene_id"] and calls[0]["sel_id"] != calls[1]["sel_id"]
    assert "selected" not in calls[0]["scene"] and calls[0]["scene"]["view_key"] == "vk" and calls[1]["selected"] == {"he_index": 2}
    crv.interactive({**sc, "points": [{"id": "x"}]}, 600, key="k", view_key="vk")
    assert calls[2]["scene_id"] != calls[0]["scene_id"]


def test_compare_viewer_view_state():
    """[6] the comparison carries the anchors and the identity of its view state (nucleus, candidate, window)."""
    from src.registration_explorer import cell_compare_viewer as cmpv
    left = cmpv.pane("L", [0, 0, 60, 60], [cmpv.item("sel", [30, 30])], anchor=[30, 30])
    right = cmpv.pane("R", [10, 10, 70, 70], [cmpv.item("active", [40, 40])], anchor=[40, 40])
    h = cmpv.html(left, right, view_key="crv-cmp-a", he=5, active=9, window=60.0)
    d = json.loads(h.split("const D = ", 1)[1].split(";\n", 1)[0])
    assert d["view_key"] == "crv-cmp-a" and d["he"] == 5 and d["active"] == 9 and d["window"] == 60.0
    assert d["panes"][1]["anchor"] == [40.0, 40.0] and "sessionStorage" in h and "dispatchEvent(new KeyboardEvent" in h
    h0 = cmpv.html(cmpv.pane("L", [0, 0, 1, 1], []), cmpv.pane("R", [0, 0, 1, 1], []))
    assert '"view_key": null' in h0                                                              # old call signature still works


# ----------------------------------------------------------------------------- Streamlit flow (AppTest, real data)
def _cra():
    spec = importlib.util.spec_from_file_location("cra_tests", REPO / "analysis/cell-review-annotation-20261118/tests_cra/test_annotation.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _ev(at, auto, i, **kw):
    at.session_state[f"regx-crv-viewer-{auto.auto_id}"] = {"nonce": f"n{i}", **kw}
    at.run()
    assert not at.exception, at.exception


@pytest.mark.slow
def test_ui_click_key_flow_records_through_review_apply(tmp_path):
    """[1][4][5][6][7][8][10][12] viewer events drive selection, candidate, verdict keys, undo / redo, next; nothing is
    recorded by a click; absent needs a comment; a stale key is not recorded; the saved file keeps the existing format."""
    from src.registration_research import cell_review as CR
    cra = _cra()
    at, auto, auto_dir, S = cra._start(tmp_path, "031-6_2")
    sha0 = json.loads((auto_dir / "auto_meta.json").read_text())["files_sha256"]
    aid = auto.auto_id
    rk = f"regx-crv-review-{aid}"
    c = auto.candidates
    multi = [int(h) for h in c.groupby("he_index").size().loc[lambda x: x >= 2].index]
    h1, h2 = multi[0], multi[1]
    # click (event) -> selected, comparison on it, nothing recorded
    _ev(at, auto, 1, kind="nucleus", role="he", index=h1)
    assert at.session_state[f"regx-crv-selhe-{aid}"] == h1
    assert at.session_state["regx-crv-cmp-last"]["he"] == h1 and at.session_state[rk].cursor == 0
    assert at.button(key="regx-crv-cmpok") and at.button(key="regx-crv-nid") and at.button(key="regx-crv-abs") and at.button(key="regx-crv-unc")
    ps = at.session_state["_picker_scene"]
    assert ps["selected"]["he_index"] == h1 and ps["selected"]["cand_index"] == c[c.he_index == h1].sort_values("rank")["if_index"].tolist()
    # the same nonce again: ignored
    _ev(at, auto, 1, kind="nucleus", role="he", index=h2)
    assert at.session_state[f"regx-crv-selhe-{aid}"] == h1
    # candidate ring -> compared candidate (radio + comparison right pane centred on it)
    f2 = int(c[(c.he_index == h1) & (c["rank"] == 2)]["if_index"].iloc[0])
    _ev(at, auto, 2, kind="candidate", he=h1, index=f2)
    cmp = at.session_state["regx-crv-cmp-last"]
    from src.registration_explorer import catalog
    iff, _ = catalog.load_nuclei(S.modality("IF"))
    assert cmp["active"] == f2 and np.allclose(cmp["right"]["anchor"], iff[f2])
    assert next(w for w in at.radio if w.key.startswith(f"regx-crv-cmpcand-{h1}-")).value == f2
    # key 1 -> confirm the compared candidate for h1 via Review.apply
    _ev(at, auto, 3, kind="key", action="confirm", he=h1, active=f2)
    rv = at.session_state[rk]
    op = rv.ops[-1]
    assert (op["he_index"], op["if_index"]) == (h1, f2) and op["op"] in ("reassign", "add") and rv.verdicts()[h1]["review_status"] == "confirmed_match"
    # a key for a nucleus the panel was not showing (stale): not recorded, that nucleus is selected instead
    n = len(rv.ops)
    _ev(at, auto, 4, kind="key", action="uncertain", he=h2, active=None)
    assert len(at.session_state[rk].ops) == n and at.session_state[f"regx-crv-selhe-{aid}"] == h2
    assert any("記録していません" in w.value for w in at.warning)
    # absent without a comment: refused, nothing recorded; with a comment: recorded
    _ev(at, auto, 5, kind="key", action="absent", he=h2, active=None)
    assert len(at.session_state[rk].ops) == n and any("コメントが必要" in w.value for w in at.warning)
    at.text_input(key="regx-crv-reason").set_value("IF切片の同じ位置に核が見えない").run()
    _ev(at, auto, 6, kind="key", action="absent", he=h2, active=None)
    rv = at.session_state[rk]
    assert rv.ops[-1]["op"] == "absent" and rv.ops[-1]["he_index"] == h2 and rv.ops[-1]["reason"]
    # undo / redo keys (history kept), the undone nucleus is shown
    _ev(at, auto, 7, kind="key", action="undo", he=h2)
    rv = at.session_state[rk]
    assert rv.cursor == len(rv.ops) - 1 and h2 not in rv.verdicts()
    _ev(at, auto, 8, kind="key", action="redo", he=h2)
    rv = at.session_state[rk]
    assert rv.cursor == len(rv.ops) and rv.verdicts()[h2]["review_status"] == "absent"
    assert [h["action"] for h in rv.history][-2:] == ["undo", "redo"]
    # Enter -> next unreviewed (no record); changing nucleus clears the comment
    _ev(at, auto, 9, kind="key", action="next", he=h2)
    h3 = at.session_state[f"regx-crv-selhe-{aid}"]
    assert h3 not in (h1, h2) and h3 not in at.session_state[rk].verdicts() and len(at.session_state[rk].ops) == n + 1
    assert at.text_input(key="regx-crv-reason").value == ""
    # auto-advance only when a verdict was recorded
    at.checkbox(key="regx-crv-advance").check().run()
    _ev(at, auto, 10, kind="key", action="absent", he=h3, active=None)          # refused (no comment): stays
    assert at.session_state[f"regx-crv-selhe-{aid}"] == h3
    _ev(at, auto, 11, kind="key", action="not_identified", he=h3, active=None)
    assert at.session_state[f"regx-crv-selhe-{aid}"] not in (h3, None) and at.session_state[rk].verdicts()[h3]["review_status"] == "not_identified"
    # Esc -> deselect
    _ev(at, auto, 12, kind="deselect")
    assert f"regx-crv-selhe-{aid}" not in at.session_state
    # buttons still work (same keys as before) and the saved file keeps the existing format
    _ev(at, auto, 13, kind="nucleus", role="he", index=h1)
    at.button(key="regx-crv-undo").click().run()
    assert not at.exception, at.exception
    at.button(key="regx-crv-save").click().run()
    d = json.loads(Path(CR.list_reviews("031-6_2", aid, tmp_path)[-1]).read_text())
    assert d["schema"] == "cellreg-cell-review-edits/2"
    assert set(d) == set(CR.Review.start(auto).to_json()) and CR.Review.from_json(d).verdicts().keys() == at.session_state[rk].verdicts().keys()
    assert json.loads((auto_dir / "auto_meta.json").read_text())["files_sha256"] == sha0      # automatic estimate untouched
