"""Two-way viewer for the cell-correspondence review tab (research mode).

Reuses the canvas renderer of ``viewer.py`` (its HTML template and layer helpers, unchanged) with a different click API:
  view     zoom / pan
  pick_he  click near an H&E nucleus  -> {"kind": "nucleus", "role": "he", "index": <global HE index>}
  pick_if  click near an IF nucleus   -> {"kind": "nucleus", "role": "if", "index": <global IF index>}
Pickable point layers carry ``pick`` ("he" | "if") and ``idx`` (global indices in the order of ``xy``).
The existing viewer component and its modes are not modified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import viewer

_BOOT = r"""
function send(type, data){ window.parent.postMessage(Object.assign({isStreamlitMessage: true, type: type}, data || {}), '*'); }
function emit(data){ send('streamlit:setComponentValue', {value: Object.assign({nonce: Date.now() + '-' + Math.random().toString(16).slice(2)}, data), dataType: 'json'}); }
function makeAPI(S, key){
  const store = (k, v) => { try { if (v === undefined) return JSON.parse(sessionStorage.getItem('crv-' + key + k) || 'null'); sessionStorage.setItem('crv-' + key + k, JSON.stringify(v)); } catch (e) { return null; } };
  const A = {mode: store('mode') || 'pick_he'};
  const opt = Object.assign({contours: true, ids: false, centres: true, alpha: 0.45}, store('opt') || {});
  A.opt = opt;
  const groups = S.contours || [];
  for (const G of groups) G.bb = G.poly.map(P => { let a = 1e30, b = 1e30, c = -1e30, d = -1e30;
    for (let i = 0; i < P.length; i += 2) { a = Math.min(a, P[i]); c = Math.max(c, P[i]); b = Math.min(b, P[i+1]); d = Math.max(d, P[i+1]); } return [a, b, c, d]; });
  const applyCentres = () => { for (const L of S.points) { if (L._userVis === undefined) L._userVis = L.visible; L.visible = opt.centres ? L._userVis : false; } };
  A.panel = (side, el, redraw) => {
    side.appendChild(el('h4', {}, 'クリック操作'));
    const sel = el('select');
    for (const [k, v] of [['view', '表示（ズーム・パン）'], ['pick_he', 'H&E核を選択'], ['pick_if', '対応先のIF核を選択']]) sel.appendChild(el('option', {value: k}, v));
    sel.value = A.mode; side.appendChild(sel);
    const msg = el('div', {style: 'color:var(--mut);margin-top:4px'}); side.appendChild(msg);
    A.info = () => { msg.innerHTML = {view: 'ホイール: ズーム / ドラッグ: パン', pick_he: 'H&E核（変換後）をクリックすると、その核の対応候補を下に表示します。ホイールでズームできます',
      pick_if: 'IF核をクリックすると、選択中のH&E核の対応先候補として下に送ります'}[A.mode]; };
    sel.onchange = () => { A.mode = sel.value; store('mode', A.mode); A.info(); redraw(); };
    A.info();
    side.appendChild(el('h4', {}, 'セグメンテーション表示'));
    const chk = (k, label) => { const l = el('label'); const c = el('input', {type: 'checkbox', checked: !!opt[k]});
      c.onchange = () => { opt[k] = c.checked; store('opt', opt); if (k === 'centres') applyCentres(); redraw(); };
      l.appendChild(c); l.appendChild(document.createTextNode(label)); side.appendChild(l); };
    chk('contours', '核輪郭（拡大時に薄く表示）'); chk('ids', '細胞ID（さらに拡大時）'); chk('centres', '核中心点');
    side.appendChild(el('div', {style: 'color:var(--mut)'}, '輪郭の濃さ'));
    const ca = el('input', {type: 'range', min: 0.1, max: 1, step: 0.05, value: opt.alpha});
    ca.oninput = () => { opt.alpha = +ca.value; store('opt', opt); redraw(); }; side.appendChild(ca);
    side.appendChild(el('div', {style: 'color:var(--mut)'}, groups.length ? '選択中の核と候補の輪郭は常に強調表示します' : '輪郭データなし（中心点でレビュー）'));
    applyCentres();
  };
  A.down = () => {};
  A.move = () => false;
  A.up = (w) => {
    const role = A.mode === 'pick_he' ? 'he' : (A.mode === 'pick_if' ? 'if' : null); if (!role) return;
    const v = window.__regx.st.view; const tol = 12 / v.k;      // 12 screen px
    let best = -1, bd = tol * tol;
    for (const L of S.points) { if (L.pick !== role || !(L.visible || (!opt.centres && L._userVis))) continue;
      for (let i = 0, n = L.xy.length / 2; i < n; i++) { const dx = L.xy[2*i] - w[0], dy = L.xy[2*i+1] - w[1], d = dx*dx + dy*dy; if (d < bd) { bd = d; best = L.idx[i]; } } }
    if (best >= 0) emit({kind: 'nucleus', role: role, index: best});
  };
  A.overlay = (ctx, v) => {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    const ring = (q, c, r) => { if (!q) return; ctx.beginPath(); ctx.arc(q[0]*v.k + v.ox, q[1]*v.k + v.oy, r, 0, 6.2832); ctx.lineWidth = 2.5; ctx.strokeStyle = c; ctx.stroke(); };
    const sp = S.selected || {};
    const W = ctx.canvas.width, H = ctx.canvas.height;
    const vx0 = -v.ox / v.k, vy0 = -v.oy / v.k, vx1 = (W - v.ox) / v.k, vy1 = (H - v.oy) / v.k;
    const path = (P) => { ctx.beginPath(); for (let i = 0; i < P.length; i += 2) { const x = P[i]*v.k + v.ox, y = P[i+1]*v.k + v.oy; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.closePath(); };
    A.drawn = 0;
    if (opt.contours && v.k >= (S.contour_min_k || 1.2)) {            // level of detail: outlines only when zoomed in
      ctx.save(); ctx.globalAlpha = opt.alpha; ctx.lineWidth = Math.min(1.6, 0.5 + 0.08 * v.k);
      for (const G of groups) { if (G.visible === false) continue; ctx.strokeStyle = G.color;
        for (let j = 0; j < G.poly.length; j++) { const b = G.bb[j]; if (b[2] < vx0 || b[0] > vx1 || b[3] < vy0 || b[1] > vy1) continue; path(G.poly[j]); ctx.stroke(); A.drawn++; } }
      ctx.restore();
    }
    if (opt.ids && v.k >= (S.label_min_k || 4)) {
      ctx.save(); const fs = Math.max(9, Math.min(14, v.k * 1.1)); ctx.font = fs + 'px sans-serif'; ctx.textBaseline = 'bottom'; let nl = 0;
      for (const G of groups) { if (!G.ids) continue; ctx.fillStyle = G.label_color || G.color;
        for (let j = 0; j < G.ids.length && nl < 800; j++) { const q = G.at[j]; if (q[0] < vx0 || q[0] > vx1 || q[1] < vy0 || q[1] > vy1) continue;
          ctx.fillText(String(G.ids[j]), q[0]*v.k + v.ox + 3, q[1]*v.k + v.oy - 3); nl++; } }
      ctx.restore();
    }
    const hl = (P, c, w, dash) => { if (!P || !P.length) return; ctx.save(); ctx.setLineDash(dash || []); ctx.lineWidth = w; ctx.strokeStyle = c; path(P); ctx.stroke(); ctx.restore(); };
    for (const P of (sp.cand_polys || [])) hl(P, '#ffffff', 1.6, [4, 3]);
    hl(sp.he_poly, '#ffeb3b', 2.6); hl(sp.if_poly, '#00e5ff', 2.6);
    ring(sp.he, '#ffeb3b', 9); ring(sp.if, '#00e5ff', 9);
    for (const q of (sp.candidates || [])) ring(q, '#ffffff', 6);
  };
  return A;
}
let lastScene = null;
window.addEventListener('message', ev => { const d = ev.data; if (!d || d.type !== 'streamlit:render') return; const a = d.args;
  if (a.scene_id !== lastScene) { lastScene = a.scene_id; boot(a.scene, makeAPI(a.scene, a.view_key || 'default')); }
  send('streamlit:setFrameHeight', {height: a.height + 24}); });
send('streamlit:componentReady', {apiVersion: 1});
"""
_DIR = Path(__file__).resolve().parent / "interactive_cell_review"
_component = None


def pick_points_layer(layer_id, label, xy, idx, role, color, visible=True, radius=2.5):
    lay = viewer.points_layer(layer_id, label, xy, color, visible=visible, radius=radius)
    lay.update({"pick": role, "idx": [int(i) for i in np.asarray(idx).ravel()]})
    return lay


def contour_group(group_id, label, packed: dict, color, ids=None, at=None, visible=True, label_color=None):
    """Outlines drawn by the overlay (level of detail + viewport culling in the browser).  packed = Contours.pack(...)."""
    g = {"id": group_id, "label": label, "color": color, "poly": packed["poly"], "idx": packed["idx"], "visible": visible}
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
    global _component
    import streamlit.components.v1 as components
    if _component is None:
        _DIR.mkdir(exist_ok=True)
        f = _DIR / "index.html"
        h = html(height)
        if not f.is_file() or f.read_text() != h:
            f.write_text(h)
        _component = components.declare_component("regx_cell_review_viewer", path=str(_DIR))
    payload = json.dumps(scene, allow_nan=False)
    sid = hashlib.sha1((payload + view_key).encode()).hexdigest()
    return _component(scene=json.loads(payload), scene_id=sid, height=height, view_key=view_key, key=key, default=None)
