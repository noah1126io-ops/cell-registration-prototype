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
  };
  A.down = () => {};
  A.move = () => false;
  A.up = (w) => {
    const role = A.mode === 'pick_he' ? 'he' : (A.mode === 'pick_if' ? 'if' : null); if (!role) return;
    const v = window.__regx.st.view; const tol = 12 / v.k;      // 12 screen px
    let best = -1, bd = tol * tol;
    for (const L of S.points) { if (L.pick !== role || !L.visible) continue;
      for (let i = 0, n = L.xy.length / 2; i < n; i++) { const dx = L.xy[2*i] - w[0], dy = L.xy[2*i+1] - w[1], d = dx*dx + dy*dy; if (d < bd) { bd = d; best = L.idx[i]; } } }
    if (best >= 0) emit({kind: 'nucleus', role: role, index: best});
  };
  A.overlay = (ctx, v) => {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    const ring = (q, c, r) => { if (!q) return; ctx.beginPath(); ctx.arc(q[0]*v.k + v.ox, q[1]*v.k + v.oy, r, 0, 6.2832); ctx.lineWidth = 2.5; ctx.strokeStyle = c; ctx.stroke(); };
    const sp = S.selected || {};
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
