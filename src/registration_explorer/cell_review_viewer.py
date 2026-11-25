"""Two-way viewer for the cell-correspondence review tab (research mode).

Reuses the canvas renderer of ``viewer.py`` (its HTML template and layer helpers, unchanged) with its own pointer API.
The template sends every pointer event to the API whenever ``API.mode != 'view'``; this API runs in mode ``review``
(2026-11-25) and does click / drag / focus itself, so no mode switch is needed:

  left click            nearest H&E nucleus (or a candidate ring of the selected one) -> select + focus (centre, zoom to
                        ~60 µm the first time; the user's zoom is kept afterwards)
  shift + left click    nearest IF nucleus of any kind (a partner outside the candidates)
  left drag             pan (a move of more than DRAG_CSS_PX CSS pixels; the release never selects)
  wheel                 zoom about the cursor (template)
  keys                  1-4 verdicts, Enter next unreviewed, Ctrl+Z / Ctrl+Y undo / redo, Esc deselect (ignored while typing)

Events sent to Python (``{nonce, kind, ...}``):
  {kind: "nucleus", role: "he", index}        H&E nucleus selected (global HE index)
  {kind: "nucleus", role: "if", index}        any IF nucleus picked with shift (global IF index)
  {kind: "candidate", he, index}              candidate ring of the selected H&E nucleus clicked
  {kind: "deselect"}
  {kind: "key", action, he, active}           action in confirm | not_identified | absent | uncertain | next | undo | redo
Screen coordinates: event clientX/Y are CSS px; the canvas and ``st.view`` (k, ox, oy) are device px
(CSS px x devicePixelRatio); positions are world µm (IF frame).  Tolerances are given in CSS px and converted.

The selection (``selected``) is sent separately from the scene: changing it redraws in place (no reboot, no image reload).
The legacy modes ``pick_he`` / ``pick_if`` / ``view`` remain available through ``A.mode`` (tests of 2026-11-17).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import viewer

CLICK_CSS_PX = 12        # pick tolerance (CSS px)
DRAG_CSS_PX = 5          # pointer travel above which a press is a drag (CSS px)
FOCUS_UM = 60.0          # first focus: about this many µm across the shorter canvas side
KEY_ACTIONS = {"1": "confirm", "2": "not_identified", "3": "absent", "4": "uncertain"}

_BOOT = r"""
function send(type, data){ window.parent.postMessage(Object.assign({isStreamlitMessage: true, type: type}, data || {}), '*'); }
function emit(data){ send('streamlit:setComponentValue', {value: Object.assign({nonce: Date.now() + '-' + Math.random().toString(16).slice(2)}, data), dataType: 'json'}); }
const CRV = {CLICK: __CLICK__, DRAG: __DRAG__, FOCUS_UM: __FOCUS__, KEYS: __KEYS__};
function typingTarget(t){ if (!t) return false; if (t.isContentEditable || /^(TEXTAREA|SELECT)$/.test(t.tagName || '')) return true;
  if (t.tagName !== 'INPUT') return false;                             // text-like inputs only (checkboxes / radios keep the shortcuts)
  return !/^(checkbox|radio|button|submit|reset|range|color|file|image)$/i.test(t.type || 'text'); }
function makeAPI(S, key){
  const store = (k, v) => { try { if (v === undefined) return JSON.parse(sessionStorage.getItem('crv-' + key + k) || 'null'); sessionStorage.setItem('crv-' + key + k, JSON.stringify(v)); } catch (e) { return null; } };
  const A = {mode: 'review', sel: S.selected || {}, drag: null, focusHe: store('focus_he'), redraw: () => {}};
  const opt = Object.assign({contours: true, ids: false, centres: true, alpha: 0.45, cells: true, cell_ids: false}, store('opt') || {});
  A.opt = opt;
  const groups = S.contours || [];
  const hasCells = groups.some(G => G.kind === 'cell');
  for (const G of groups) G.bb = G.poly.map(P => { let a = 1e30, b = 1e30, c = -1e30, d = -1e30;
    for (let i = 0; i < P.length; i += 2) { a = Math.min(a, P[i]); c = Math.max(c, P[i]); b = Math.min(b, P[i+1]); d = Math.max(d, P[i+1]); } return [a, b, c, d]; });
  // layer visibility chosen in the side panel survives re-renders (scene changes after a verdict)
  const vis = store('vis') || {};
  for (const L of [].concat(S.images || [], S.points || [], S.polys || [])) if (L.id && vis[L.id] !== undefined) L.visible = vis[L.id];
  const applyCentres = () => { for (const L of (S.points || [])) { if (L._userVis === undefined) L._userVis = L.visible; L.visible = opt.centres ? L._userVis : false; } };
  const R = () => window.__regx || {st: {}};
  const dpr = () => window.devicePixelRatio || 1;
  const canvasWH = () => { const c = typeof document !== 'undefined' && document.getElementById ? document.getElementById('cv') : null;
    return c ? [c.width, c.height] : [800, 600]; };
  // same storage as the template's saveView(), so a re-render restores this view
  A.saveView = () => { const st = R().st, v = st.view; if (!S.view_key || !v) return; const [W, H] = canvasWH();
    try { sessionStorage.setItem('regx-view-' + S.view_key, JSON.stringify({k: v.k, cx: (W/2 - v.ox)/v.k, cy: (H/2 - v.oy)/v.k, mode: st.mode, alpha: st.alpha})); } catch (e) {} };
  A.centreOn = (q, k) => { const st = R().st; if (!st.view || !q) return; const [W, H] = canvasWH(); k = k || st.view.k;
    st.view = {k: k, ox: W/2 - k*q[0], oy: H/2 - k*q[1]}; st.touched = true; A.saveView(); };
  A.focusK = (um) => { const [W, H] = canvasWH(); return Math.min(W, H) / (um || CRV.FOCUS_UM); };
  A.focus = (q) => { const st = R().st; if (!st.view || !q) return; const kf = A.focusK();
    A.centreOn(q, st.view.k < 0.5 * kf ? kf : st.view.k); };           // zoomed out (overview) -> ~60 µm; otherwise keep the user's zoom
  A.selHe = () => (A.sel && A.sel.he_index !== undefined && A.sel.he_index !== null) ? A.sel.he_index : null;
  // new selection from Python (typed ID, Enter, auto-advance, undo): focus when the nucleus changed
  A.setSel = (sel) => { A.sel = sel || {}; const h = A.selHe();
    if (h !== null && h !== A.focusHe) { A.focus(A.sel.he); }
    A.focusHe = h; store('focus_he', h); };
  const cands = (h) => (S.cand_table && S.cand_table[String(h)]) || [];
  const defaultActive = (h) => { const c = cands(h), p = S.partner ? S.partner[String(h)] : undefined;
    if (p !== undefined && c.some(r => r[0] === p)) return p; return c.length ? c[0][0] : (p === undefined ? null : p); };
  const status = (txt) => { if (A.msg) A.msg.textContent = txt; };
  A.panel = (side, el, redraw) => {
    A.redraw = redraw;
    side.appendChild(el('h4', {}, 'マウス操作'));
    side.appendChild(el('div', {style: 'color:var(--mut)'}, '左クリック: H&E核を選択して拡大<br>候補の白い輪をクリック: 比較する候補を変更<br>'
      + 'Shift+クリック: 任意のIF核を対応先の候補に<br>ドラッグ: 移動 / ホイール: ズーム<br>キー: 1 対応確認 / 2 未特定 / 3 非観測 / 4 判断不能<br>'
      + 'Enter 次の未レビュー核 / Ctrl+Z 取り消し / Ctrl+Y やり直し / Esc 選択解除'));
    A.msg = el('div', {style: 'margin-top:4px;color:#ffeb3b;min-height:1.4em'}); side.appendChild(A.msg);
    const row = el('div', {style: 'margin-top:4px;display:flex;gap:4px;flex-wrap:wrap'});
    for (const um of (S.focus_windows || [30, 60, 120])) { const b = el('button', {}, um + ' µm');
      b.onclick = () => { const q = A.sel.he; if (q) { A.centreOn(q, A.focusK(um)); redraw(); } else status('核を選択すると、その周辺を拡大します'); }; row.appendChild(b); }
    side.appendChild(el('div', {style: 'color:var(--mut);margin-top:6px'}, '選択核の周辺を表示'));
    side.appendChild(row);
    side.appendChild(el('h4', {}, 'セグメンテーション表示'));
    const chk = (k, label) => { const l = el('label'); const c = el('input', {type: 'checkbox', checked: !!opt[k]});
      c.onchange = () => { opt[k] = c.checked; store('opt', opt); if (k === 'centres') applyCentres(); redraw(); };
      l.appendChild(c); l.appendChild(document.createTextNode(label)); side.appendChild(l); };
    chk('contours', '核輪郭（実線・拡大時に表示）'); chk('ids', '核ID（さらに拡大時）'); chk('centres', '核中心点');
    if (hasCells) { chk('cells', '細胞全体の輪郭（破線）'); chk('cell_ids', '細胞ID（さらに拡大時）'); }
    side.appendChild(el('div', {style: 'color:var(--mut)'}, '輪郭の濃さ'));
    const ca = el('input', {type: 'range', min: 0.1, max: 1, step: 0.05, value: opt.alpha});
    ca.oninput = () => { opt.alpha = +ca.value; store('opt', opt); redraw(); }; side.appendChild(ca);
    side.appendChild(el('div', {style: 'color:var(--mut)'}, groups.length ? '選択中の核と候補の輪郭は常に強調表示します' : '輪郭データなし（中心点でレビュー）'));
    applyCentres();
  };
  // ---- hit testing (world µm; tolerance CLICK CSS px -> device px -> µm)
  const nearest = (role, w, tol2) => { let best = -1, bd = tol2, xy = null;
    for (const L of (S.points || [])) { if (L.pick !== role || !(L.visible || (!opt.centres && L._userVis))) continue;
      for (let i = 0, n = L.xy.length / 2; i < n; i++) { const dx = L.xy[2*i] - w[0], dy = L.xy[2*i+1] - w[1], d = dx*dx + dy*dy;
        if (d < bd) { bd = d; best = L.idx[i]; xy = [L.xy[2*i], L.xy[2*i+1]]; } } }
    return best >= 0 ? {index: best, d2: bd, xy: xy} : null; };
  A.pickAt = (w, e) => {
    const v = R().st.view; const tol = CRV.CLICK * dpr() / v.k, tol2 = tol * tol;
    if (A.mode === 'view') return null;
    if (A.mode === 'pick_he' || A.mode === 'pick_if') { const r = nearest(A.mode === 'pick_he' ? 'he' : 'if', w, tol2);
      return r ? {kind: 'nucleus', role: A.mode === 'pick_he' ? 'he' : 'if', index: r.index, xy: r.xy} : null; }
    if (e && e.shiftKey) { const r = nearest('if', w, tol2); return r ? {kind: 'nucleus', role: 'if', index: r.index, xy: r.xy} : null; }
    const h = nearest('he', w, tol2); let c = null; const hs = A.selHe();
    if (hs !== null) for (const r of cands(hs)) { const dx = r[1] - w[0], dy = r[2] - w[1], d = dx*dx + dy*dy; if (d < (c ? c.d2 : tol2)) c = {index: r[0], d2: d, xy: [r[1], r[2]]}; }
    if (c && (!h || c.d2 <= h.d2)) return {kind: 'candidate', he: hs, index: c.index, xy: c.xy};
    return h ? {kind: 'nucleus', role: 'he', index: h.index, xy: h.xy} : null; };
  A.click = (w, e) => {
    const p = A.pickAt(w, e); if (!p) { if (A.mode === 'review') status(''); return; }
    if (A.mode !== 'review') { emit({kind: p.kind, role: p.role, index: p.index}); return; }
    if (p.kind === 'nucleus' && p.role === 'he') {                    // optimistic: highlight + focus now, Python fills in the rest
      if (p.index !== A.selHe()) A.sel = {he_index: p.index, he: p.xy, candidates: cands(p.index).map(r => [r[1], r[2]]),
        cand_index: cands(p.index).map(r => r[0]), cand_rank: cands(p.index).map(r => r[3]), active: defaultActive(p.index)};
      A.focus(p.xy); A.focusHe = p.index; store('focus_he', p.index); status('H&E核 ' + (S.he_ids ? S.he_ids[String(p.index)] : p.index) + ' を選択');
      emit({kind: 'nucleus', role: 'he', index: p.index}); return; }
    if (p.kind === 'candidate') { A.sel = Object.assign({}, A.sel, {active: p.index}); status('比較する候補を変更');
      emit({kind: 'candidate', he: p.he, index: p.index}); return; }
    A.sel = Object.assign({}, A.sel, {if: p.xy, if_index: p.index, active: p.index}); status('IF核を対応先の候補に追加');
    emit({kind: 'nucleus', role: 'if', index: p.index}); };
  // ---- pointer: press / move / release (template routes them here because mode != 'view')
  A.down = (w, e) => { if (e && e.button !== undefined && e.button !== 0) return; const v = R().st.view; if (!v) return;
    A.drag = {x: e ? e.clientX : 0, y: e ? e.clientY : 0, ox: v.ox, oy: v.oy, moved: false}; };
  A.move = (w, e) => { const d = A.drag; if (!d || !e) return false; const dx = e.clientX - d.x, dy = e.clientY - d.y;
    if (!d.moved && Math.hypot(dx, dy) <= CRV.DRAG) return false;
    d.moved = true; const st = R().st; st.touched = true; st.view.ox = d.ox + dx * dpr(); st.view.oy = d.oy + dy * dpr();
    if (e.target && e.target.style) e.target.style.cursor = 'grabbing'; return true; };
  A.up = (w, e) => { const d = A.drag; A.drag = null; if (e && e.target && e.target.style) e.target.style.cursor = 'grab';
    if (d && d.moved) { A.saveView(); return; }                       // end of a drag: never a selection
    if (!d && e) return;                                               // release without our press (other button)
    A.click(w, e); };
  // ---- keyboard (in the viewer and in the page around it); typing in inputs is never a shortcut
  A.key = (e) => {
    if (typingTarget(e.target) || e.altKey || e.metaKey) return false;
    const k = (e.key || '').toLowerCase(); let act = null;
    if (e.ctrlKey) { if (k === 'z' && !e.shiftKey) act = 'undo'; else if (k === 'y' || (k === 'z' && e.shiftKey)) act = 'redo'; }
    else if (!e.shiftKey && CRV.KEYS[e.key]) act = CRV.KEYS[e.key];
    else if (e.key === 'Enter' && !e.shiftKey) act = 'next';
    else if (e.key === 'Escape') act = 'deselect';
    if (!act) return false;
    if (e.preventDefault) e.preventDefault(); if (e.stopPropagation) e.stopPropagation();
    if (act === 'deselect') { A.sel = {}; A.focusHe = null; store('focus_he', null); status('選択を解除'); A.redraw(); emit({kind: 'deselect'}); return true; }
    const h = A.selHe();
    if (CRV.KEYS[e.key] && h === null) { status('先にH&E核を選択してください'); return true; }
    emit({kind: 'key', action: act, he: h, active: (A.sel && A.sel.active !== undefined) ? A.sel.active : null}); return true; };
  A.overlay = (ctx, v) => {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    const ring = (q, c, r, w) => { if (!q) return; ctx.beginPath(); ctx.arc(q[0]*v.k + v.ox, q[1]*v.k + v.oy, r, 0, 6.2832); ctx.lineWidth = w || 2.5; ctx.strokeStyle = c; ctx.stroke(); };
    const sp = A.sel || {};
    const W = ctx.canvas.width, H = ctx.canvas.height;
    const vx0 = -v.ox / v.k, vy0 = -v.oy / v.k, vx1 = (W - v.ox) / v.k, vy1 = (H - v.oy) / v.k;
    const path = (P) => { ctx.beginPath(); for (let i = 0; i < P.length; i += 2) { const x = P[i]*v.k + v.ox, y = P[i+1]*v.k + v.oy; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.closePath(); };
    A.drawn = 0;
    {                                                                 // level of detail: outlines only when zoomed in
      ctx.save(); ctx.globalAlpha = opt.alpha; const lw = Math.min(1.6, 0.5 + 0.08 * v.k);
      for (const G of groups) { if (G.visible === false) continue; const cell = G.kind === 'cell';
        if (cell ? (!opt.cells || v.k < (S.cell_min_k || 0.6)) : (!opt.contours || v.k < (S.contour_min_k || 1.2))) continue;
        ctx.strokeStyle = G.color; ctx.setLineDash(G.dash || []); ctx.lineWidth = lw * (G.wmul || 1);
        for (let j = 0; j < G.poly.length; j++) { const b = G.bb[j]; if (b[2] < vx0 || b[0] > vx1 || b[3] < vy0 || b[1] > vy1) continue; path(G.poly[j]); ctx.stroke(); A.drawn++; } }
      ctx.restore();
    }
    if ((opt.ids || opt.cell_ids) && v.k >= (S.label_min_k || 4)) {
      ctx.save(); const fs = Math.max(9, Math.min(14, v.k * 1.1)); ctx.font = fs + 'px sans-serif'; ctx.textBaseline = 'bottom'; let nl = 0;
      for (const G of groups) { if (!G.ids || G.visible === false || !(G.kind === 'cell' ? (opt.cell_ids && opt.cells) : opt.ids)) continue; ctx.fillStyle = G.label_color || G.color;
        for (let j = 0; j < G.ids.length && nl < 800; j++) { const q = G.at[j]; if (q[0] < vx0 || q[0] > vx1 || q[1] < vy0 || q[1] > vy1) continue;
          ctx.fillText(String(G.ids[j]), q[0]*v.k + v.ox + 3, q[1]*v.k + v.oy - 3); nl++; } }
      ctx.restore();
    }
    const hl = (P, c, w, dash) => { if (!P || !P.length) return; ctx.save(); ctx.setLineDash(dash || []); ctx.lineWidth = w; ctx.strokeStyle = c; path(P); ctx.stroke(); ctx.restore(); };
    if (opt.cells) { for (const P of (sp.cand_cell_polys || [])) hl(P, '#ffcc80', 1.4, [7, 4]);
      for (const P of (sp.he_cell_polys || [])) hl(P, '#ff9800', 2.4, [8, 4]); for (const P of (sp.if_cell_polys || [])) hl(P, '#ff6d00', 2.4, [8, 4]); }
    for (const P of (sp.cand_polys || [])) hl(P, '#ffffff', 1.6, [4, 3]);
    hl(sp.active_poly, '#ff40ff', 2.6);
    hl(sp.he_poly, '#ffeb3b', 2.6); hl(sp.if_poly, '#00e5ff', 2.6);
    const ci = sp.cand_index || [], cr = sp.cand_rank || [];
    (sp.candidates || []).forEach((q, i) => { const isAct = ci[i] !== undefined && ci[i] === sp.active, isCur = ci[i] !== undefined && ci[i] === sp.partner;
      ring(q, isAct ? '#ff40ff' : (isCur ? '#00e676' : '#ffffff'), isAct ? 8 : 6, isAct ? 3 : 2);
      if (cr[i] !== undefined && v.k >= 1.5 && ctx.fillText) { ctx.save(); ctx.font = '12px sans-serif'; ctx.fillStyle = isAct ? '#ff40ff' : '#ffffff';
        ctx.fillText('#' + cr[i], q[0]*v.k + v.ox + 8, q[1]*v.k + v.oy - 6); ctx.restore(); } });
    if (sp.active_xy && !(sp.cand_index || []).includes(sp.active)) ring(sp.active_xy, '#ff40ff', 8, 3);
    ring(sp.he, '#ffeb3b', 10, 3); ring(sp.if, '#00e5ff', 9);
    // remember the side-panel layer toggles (checked state lives on the layer objects)
    const now = {}; for (const L of [].concat(S.images || [], S.points || [], S.polys || [])) if (L.id) now[L.id] = (L._userVis !== undefined ? L._userVis : L.visible);
    const js = JSON.stringify(now); if (js !== A._vis) { A._vis = js; store('vis', now); }
  };
  return A;
}
let lastScene = null, lastSel = null, API = null;
function onKey(e){ if (!API || API.mode !== 'review') return;
  try { const fe = window.frameElement; if (fe && !fe.getClientRects().length) return; } catch (err) {}   // tab not shown
  if (API.key(e)) { try { API.redraw(); } catch (err) {} } }
window.addEventListener('keydown', onKey);
try { const pd = window.parent && window.parent.document;            // shortcuts also work with the focus on the page around the viewer
  if (pd && pd.addEventListener) { if (window.parent.__crvKey) pd.removeEventListener('keydown', window.parent.__crvKey, true);
    window.parent.__crvKey = onKey; pd.addEventListener('keydown', onKey, true); } } catch (err) {}
window.addEventListener('message', ev => { const d = ev.data; if (!d || d.type !== 'streamlit:render') return; const a = d.args;
  if (a.scene_id !== lastScene) { lastScene = a.scene_id; lastSel = a.sel_id; a.scene.selected = a.selected || {}; API = makeAPI(a.scene, a.view_key || 'default');
    boot(a.scene, API); API.setSel(a.selected); API.redraw(); }
  else if (a.sel_id !== lastSel) { lastSel = a.sel_id; API.setSel(a.selected); API.redraw(); }
  send('streamlit:setFrameHeight', {height: a.height + 24}); });
send('streamlit:componentReady', {apiVersion: 1});
""".replace("__CLICK__", str(CLICK_CSS_PX)).replace("__DRAG__", str(DRAG_CSS_PX)).replace("__FOCUS__", str(FOCUS_UM)) \
   .replace("__KEYS__", json.dumps(KEY_ACTIONS))
_DIR = Path(__file__).resolve().parent / "interactive_cell_review"
_component = None


def pick_points_layer(layer_id, label, xy, idx, role, color, visible=True, radius=2.5):
    lay = viewer.points_layer(layer_id, label, xy, color, visible=visible, radius=radius)
    lay.update({"pick": role, "idx": [int(i) for i in np.asarray(idx).ravel()]})
    return lay


def contour_group(group_id, label, packed: dict, color, ids=None, at=None, visible=True, label_color=None, kind="nucleus", dash=None,
                  wmul=1.0):
    """Outlines drawn by the overlay (level of detail + viewport culling in the browser).  packed = Contours.pack(...).

    kind "nucleus" (solid, toggle 核輪郭 / 核ID) or "cell" (whole-cell outlines: dashed, toggle 細胞全体の輪郭 / 細胞ID)."""
    assert kind in ("nucleus", "cell"), kind
    g = {"id": group_id, "label": label, "color": color, "poly": packed["poly"], "idx": packed["idx"], "visible": visible, "kind": kind}
    if dash:
        g["dash"] = list(dash)
    if wmul != 1.0:
        g["wmul"] = float(wmul)
    if ids is not None:
        g["ids"] = [str(v) for v in ids]
        g["at"] = np.round(np.asarray(at, float).reshape(-1, 2), 2).tolist()
        if label_color:
            g["label_color"] = label_color
    return g


def segments_layer(layer_id, label, a, b, color, width=1.2, visible=True):
    """Many line segments drawn as one toggle (poly layer with `multi`)."""
    a, b = np.asarray(a, float).reshape(-1, 2), np.asarray(b, float).reshape(-1, 2)
    multi = [np.round([a[i, 0], a[i, 1], b[i, 0], b[i, 1]], 2).tolist() for i in range(len(a))]
    lay = viewer.poly_layer(layer_id, label, np.zeros((0, 2)), color, visible=visible, width=width, closed=False)
    lay["multi"] = multi
    return lay


def html(height: int = 720) -> str:
    return viewer._TEMPLATE.replace("__BOOT__", _BOOT).replace("__H__", str(int(height)))


def interactive(scene: dict, height: int = 720, key: str | None = None, view_key: str = "default"):
    """Returns the last event (see the module docstring) or None.  ``scene["selected"]`` is sent apart from the scene so that
    a new selection redraws in place; any other change re-renders the scene with the view (zoom / pan) and toggles kept."""
    global _component
    import streamlit.components.v1 as components
    if _component is None:
        _DIR.mkdir(exist_ok=True)
        f = _DIR / "index.html"
        h = html(height)
        if not f.is_file() or f.read_text() != h:
            f.write_text(h)
        _component = components.declare_component("regx_cell_review_viewer", path=str(_DIR))
    sc = {k: v for k, v in scene.items() if k != "selected"}
    sc["view_key"] = view_key                     # the template keeps zoom / pan per view_key across re-renders
    payload = json.dumps(sc, allow_nan=False)
    sel = json.dumps(scene.get("selected") or {}, allow_nan=False)
    sid = hashlib.sha1((payload + view_key).encode()).hexdigest()
    return _component(scene=json.loads(payload), scene_id=sid, selected=json.loads(sel), sel_id=hashlib.sha1(sel.encode()).hexdigest(),
                      height=height, view_key=view_key, key=key, default=None)
