"""Side-by-side comparison of one H&E nucleus and its IF candidates (research, display only; 2026-11-18).

Left = H&E section, right = IF section, each in its **own, undeformed** frame (µm of that section; the original image crop
is never warped).  The global affine (H&E -> IF) is used only for the optional position sync of the two views.

Static component (components.html): zoom (wheel), pan (drag), local re-centre, contour / centre / ID / image toggles,
contour opacity.  Decisions are made with Streamlit buttons next to it, so no value is sent back.

2026-11-25: each pane is centred on its ``anchor`` (left: the selected H&E nucleus; right: the compared candidate, or the
Global prediction when there is none).  Zoom (relative to the window), pan and the toggles are kept in sessionStorage per
``view_key``: a re-render for the same nucleus keeps the view; another nucleus is centred again at the same zoom; another
candidate re-centres the right pane only.  Changing the window (30 / 60 / 120 µm) or 「選択核を中心に戻す」 resets the zoom.
Shortcut keys pressed inside the panes are forwarded to the page (the review viewer listens there).

Item styles (contours and points):
  sel        selected H&E nucleus                     yellow, thick
  active     candidate being compared                 magenta, thick
  cand       other candidates (rank label)            white, dashed
  target     IF nucleus picked in the main viewer     cyan, thick
  current    current partner in the review           green, thick
  faint      other nuclei in the window (neighbours)  thin, low opacity
Whole-cell outlines (item kind "cell", 2026-11-18; separate toggles 細胞輪郭 / 細胞ID, always dashed, never filled):
  cell_sel     cell(s) containing the selected / compared nucleus   orange, dashed, thick
  cell         other cells in the window                            light orange, dashed, thin, low opacity
"""
from __future__ import annotations

import base64
import json

import numpy as np

STYLES = {"sel": ("#ffeb3b", 2.6, []), "active": ("#ff40ff", 2.8, []), "cand": ("#ffffff", 1.5, [4, 3]), "target": ("#00e5ff", 2.6, []),
          "current": ("#00e676", 2.4, [2, 2]), "faint": ("#b0bec5", 0.9, []), "cell_sel": ("#ff9100", 2.4, [8, 4]),
          "cell": ("#ffcc80", 1.1, [6, 4])}

_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><style>
:root{--bg:#10161a;--fg:#e0e6ea;--mut:#90a4ae}
body{margin:0;background:var(--bg);color:var(--fg);font:12px sans-serif}
#bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:4px 6px}
#bar label{white-space:nowrap}
#panes{display:flex;gap:6px;padding:0 6px}
.pane{flex:1;min-width:0;position:relative}
.pane canvas{width:100%;height:__H__px;background:#000;border:1px solid #37474f;cursor:grab;display:block}
.ttl{font-weight:bold;margin:2px 0}
.note{color:var(--mut);margin:2px 0;min-height:1.2em}
#leg{color:var(--mut);padding:2px 6px}
.sw{display:inline-block;width:14px;height:3px;vertical-align:middle;margin:0 3px 0 8px}
button{background:#263238;color:var(--fg);border:1px solid #37474f;border-radius:4px}
</style></head><body>
<div id="bar"></div><div id="panes"></div><div id="leg"></div>
<script>
const D = __DATA__;
const VK = 'crv-cmp-' + (D.view_key || 'default');
const load = (k) => { try { return JSON.parse(sessionStorage.getItem(VK + k) || 'null'); } catch (e) { return null; } };
const save = (k, v) => { try { sessionStorage.setItem(VK + k, JSON.stringify(v)); } catch (e) {} };
const opt = Object.assign({img: true, contours: true, centres: true, ids: true, cells: true, cellids: false, sync: D.sync_default, alpha: 0.55},
  load('opt') || {});
const bar = document.getElementById('bar');
function chk(k, label){ const l = document.createElement('label'); const c = document.createElement('input'); c.type = 'checkbox'; c.checked = opt[k];
  c.onchange = () => { opt[k] = c.checked; save('opt', opt); if (k === 'sync' && opt.sync) { syncFrom(0); saveView(); } drawAll(); }; l.appendChild(c); l.appendChild(document.createTextNode(' ' + label)); bar.appendChild(l); }
chk('img', '元画像'); chk('contours', '核輪郭'); chk('centres', '核中心点'); chk('ids', '核ID');
if (D.panes.some(S => S.items.some(it => it.k === 'cell'))) { chk('cells', '細胞輪郭（破線）'); chk('cellids', '細胞ID'); } if (D.sync) chk('sync', '位置を同期（Global Affine）');
{ const l = document.createElement('label'); l.textContent = '周辺の輪郭の濃さ '; const r = document.createElement('input'); r.type = 'range'; r.min = 0.05; r.max = 1; r.step = 0.05; r.value = opt.alpha;
  r.oninput = () => { opt.alpha = +r.value; save('opt', opt); drawAll(); }; l.appendChild(r); bar.appendChild(l); }
{ const b = document.createElement('button'); b.textContent = '選択核を中心に戻す'; b.onclick = () => { P.forEach(p => p.fit()); if (opt.sync) syncFrom(0); saveView(); drawAll(); }; bar.appendChild(b); }
const leg = document.getElementById('leg');
leg.innerHTML = Object.entries(D.legend).map(([k, v]) => '<span class="sw" style="background:' + D.styles[k][0] + '"></span>' + v).join('') +
  '<br>ホイール: ズーム / ドラッグ: パン（左右とも、この表示を保ったまま候補・核を切り替えられます）。左右とも未変形の元画像（各切片の座標）。';
const P = [];
function Pane(i, S){
  const wrap = document.createElement('div'); wrap.className = 'pane';
  wrap.innerHTML = '<div class="ttl"></div><div class="note"></div>'; wrap.querySelector('.ttl').textContent = S.title; wrap.querySelector('.note').textContent = S.note || '';
  const cv = document.createElement('canvas'); wrap.appendChild(cv); document.getElementById('panes').appendChild(wrap);
  const ctx = cv.getContext('2d'); const me = {S: S, cv: cv, view: null};
  let im = null; if (S.image) { im = new Image(); im.onload = () => draw(); im.src = S.image.src; }
  me.size = () => { const r = cv.getBoundingClientRect(), d = window.devicePixelRatio || 1; cv.width = Math.max(2, r.width * d); cv.height = Math.max(2, r.height * d); };
  me.k0 = () => { const b = S.bounds; return Math.min(cv.width / (b[2] - b[0]), cv.height / (b[3] - b[1])); };   // device px per µm of the window
  me.anchor = () => S.anchor || [(S.bounds[0] + S.bounds[2]) / 2, (S.bounds[1] + S.bounds[3]) / 2];
  me.fit = () => { me.size(); me.setCentre(me.anchor(), me.k0()); };
  me.centre = () => { const v = me.view; return [(cv.width / 2 - v.ox) / v.k, (cv.height / 2 - v.oy) / v.k]; };
  me.setCentre = (c, k) => { me.view = {k: k, ox: cv.width / 2 - k * c[0], oy: cv.height / 2 - k * c[1]}; };
  const path = (Q) => { const v = me.view; ctx.beginPath(); for (let j = 0; j < Q.length; j += 2) { const x = Q[j] * v.k + v.ox, y = Q[j+1] * v.k + v.oy; j ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.closePath(); };
  function draw(){
    if (!me.view) me.fit(); const v = me.view;
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = '#000'; ctx.fillRect(0, 0, cv.width, cv.height);
    if (opt.img && im && im.complete) { const m = S.image.m; ctx.save(); ctx.setTransform(v.k, 0, 0, v.k, v.ox, v.oy); ctx.transform(m[0], m[1], m[2], m[3], m[4], m[5]);
      ctx.imageSmoothingEnabled = v.k * Math.hypot(m[0], m[1]) < 2; ctx.drawImage(im, 0, 0); ctx.restore(); }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    const lo = it => (it.s === 'faint' || it.s === 'cell') ? 0 : (it.k === 'cell' ? 1 : 2);
    const items = S.items.slice().sort((a, b) => lo(a) - lo(b));
    for (const it of items) { const st = D.styles[it.s]; const faint = it.s === 'faint' || it.s === 'cell';
      if (it.k === 'cell') {
        if (opt.cells && it.poly) { ctx.save(); ctx.globalAlpha = faint ? opt.alpha : 1; ctx.setLineDash(st[2]); ctx.lineWidth = st[1]; ctx.strokeStyle = st[0]; path(it.poly); ctx.stroke(); ctx.restore(); }
        if (opt.cells && opt.cellids && it.label && it.c) { const x = it.c[0] * v.k + v.ox, y = it.c[1] * v.k + v.oy;
          ctx.save(); ctx.font = 'italic 10px sans-serif'; ctx.lineWidth = 3; ctx.strokeStyle = '#000'; ctx.strokeText(it.label, x, y); ctx.fillStyle = st[0]; ctx.fillText(it.label, x, y); ctx.restore(); }
        continue; }
      if (opt.contours && it.poly) { ctx.save(); ctx.globalAlpha = faint ? opt.alpha : 1; ctx.setLineDash(st[2]); ctx.lineWidth = st[1]; ctx.strokeStyle = st[0]; path(it.poly); ctx.stroke(); ctx.restore(); }
      if (opt.centres && it.c) { const x = it.c[0] * v.k + v.ox, y = it.c[1] * v.k + v.oy; ctx.save(); ctx.globalAlpha = faint ? Math.max(0.35, opt.alpha) : 1;
        ctx.fillStyle = st[0]; ctx.beginPath(); ctx.arc(x, y, faint ? 2 : 3.5, 0, 6.2832); ctx.fill();
        if (!it.poly && !faint) { ctx.lineWidth = 2; ctx.strokeStyle = st[0]; ctx.beginPath(); ctx.arc(x, y, 9, 0, 6.2832); ctx.stroke(); }   // no contour: ring
        ctx.restore(); }
      if (opt.ids && it.label && it.c && (!faint || v.k > 6)) { const x = it.c[0] * v.k + v.ox, y = it.c[1] * v.k + v.oy;
        ctx.save(); ctx.font = (faint ? 10 : 12) + 'px sans-serif'; ctx.lineWidth = 3; ctx.strokeStyle = '#000'; ctx.strokeText(it.label, x + 5, y - 5);
        ctx.fillStyle = faint ? '#cfd8dc' : st[0]; ctx.fillText(it.label, x + 5, y - 5); ctx.restore(); } }
    ctx.fillStyle = '#90a4ae'; ctx.font = '11px sans-serif'; const sb = S.scale_um || 10; const L = sb * v.k;
    ctx.fillRect(10, cv.height - 14, L, 3); ctx.fillText(sb + ' µm', 12 + L, cv.height - 9);
  }
  me.draw = draw;
  let drag = null;
  cv.addEventListener('wheel', e => { e.preventDefault(); const r = cv.getBoundingClientRect(), d = window.devicePixelRatio || 1; const sx = (e.clientX - r.left) * d, sy = (e.clientY - r.top) * d;
    const f = Math.exp(-e.deltaY * 0.0015), v = me.view; v.ox = sx - (sx - v.ox) * f; v.oy = sy - (sy - v.oy) * f; v.k *= f; syncFrom(i); drawAll(); saveView(); }, {passive: false});
  cv.addEventListener('pointerdown', e => { drag = {x: e.clientX, y: e.clientY, ox: me.view.ox, oy: me.view.oy}; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener('pointermove', e => { if (!drag) return; const d = window.devicePixelRatio || 1; me.view.ox = drag.ox + (e.clientX - drag.x) * d; me.view.oy = drag.oy + (e.clientY - drag.y) * d; syncFrom(i); drawAll(); });
  cv.addEventListener('pointerup', () => { if (drag) saveView(); drag = null; });
  return me;
}
function syncFrom(i){
  if (!opt.sync || !D.sync) return; const a = P[i], b = P[1 - i]; const c = a.centre(); const A = D.sync.A, t = D.sync.t, s = D.sync.scale;
  if (i === 0) b.setCentre([A[0][0] * c[0] + A[0][1] * c[1] + t[0], A[1][0] * c[0] + A[1][1] * c[1] + t[1]], a.view.k / s);
  else { const Ai = D.sync.Ainv, ti = D.sync.tinv; b.setCentre([Ai[0][0] * c[0] + Ai[0][1] * c[1] + ti[0], Ai[1][0] * c[0] + Ai[1][1] * c[1] + ti[1]], a.view.k * s); }
}
function drawAll(){ P.forEach(p => p.draw()); }
// view state: zoom relative to the window (rel), centre offset from the anchor (off, µm); kept per nucleus / candidate
function saveView(){ save('view', {he: D.he, active: D.active, window: D.window, rel: P.map(p => p.view.k / p.k0()),
  off: P.map(p => { const c = p.centre(), a = p.anchor(); return [c[0] - a[0], c[1] - a[1]]; })}); }
function restore(){
  const s = load('view'); P.forEach(p => p.fit());
  if (!s || s.window !== D.window) return 'fit';
  P.forEach((p, i) => { const keep = s.he === D.he && (i === 0 || s.active === D.active);   // same nucleus (and candidate on the right): keep the pan
    const a = p.anchor(), o = keep ? s.off[i] : [0, 0]; p.setCentre([a[0] + o[0], a[1] + o[1]], p.k0() * (s.rel[i] || 1)); });
  return s.he === D.he ? (s.active === D.active ? 'kept' : 'kept-left') : 'zoom-kept';
}
D.panes.forEach((S, i) => P.push(Pane(i, S)));
const restored = restore(); if (opt.sync && restored !== 'kept') syncFrom(0); drawAll();
if (window.ResizeObserver) new ResizeObserver(() => { P.forEach(p => { if (!p.view) { p.fit(); return; } const c = p.centre(), rel = p.view.k / p.k0(); p.size(); p.setCentre(c, p.k0() * rel); });
  drawAll(); }).observe(document.body);
// shortcut keys pressed in this frame -> the page (the review viewer listens there); typing in inputs is left alone
const FWD = ['1', '2', '3', '4', 'Enter', 'Escape'];
window.addEventListener('keydown', e => { const t = e.target; if (t && (/^(TEXTAREA|SELECT)$/.test(t.tagName || '') || (t.tagName === 'INPUT' && !/^(checkbox|radio|range|button)$/i.test(t.type || 'text')))) return;
  const k = (e.key || '').toLowerCase(); if (!(FWD.includes(e.key) || (e.ctrlKey && (k === 'z' || k === 'y')))) return;
  e.preventDefault();
  try { window.parent.document.dispatchEvent(new KeyboardEvent('keydown', {key: e.key, ctrlKey: e.ctrlKey, shiftKey: e.shiftKey, bubbles: true})); } catch (err) {} });
window.__cmp = {D: D, P: P, opt: opt, restored: restored};
</script></body></html>"""

LEGEND = {"sel": "選択中のH&E核", "active": "比較中の候補", "cand": "その他の候補（#順位）", "current": "現在の対応先", "target": "クリックしたIF核",
          "faint": "周辺の核"}
CELL_LEGEND = {"cell_sel": "核を含む細胞（破線）", "cell": "周辺の細胞（破線）"}


def image_entry(rgb: np.ndarray, um_per_px: float, origin_xy) -> dict:
    """Crop placed in the section frame: pixel (u, v) top-left corner -> origin + (u, v) * um_per_px."""
    from . import imaging
    jpg = imaging.encode_jpeg(np.ascontiguousarray(rgb), 90)
    return {"src": "data:image/jpeg;base64," + base64.b64encode(jpg).decode(), "w": int(rgb.shape[1]), "h": int(rgb.shape[0]),
            "m": [float(um_per_px), 0.0, 0.0, float(um_per_px), float(origin_xy[0]), float(origin_xy[1])]}


def item(style, centre=None, poly=None, label=None, kind="nucleus") -> dict:
    assert style in STYLES, style
    assert kind in ("nucleus", "cell"), kind
    assert (kind == "cell") == style.startswith("cell"), (kind, style)
    d = {"s": style}
    if kind == "cell":
        d["k"] = "cell"
    if centre is not None:
        d["c"] = np.round(np.asarray(centre, float), 3).tolist()
    if poly is not None:
        d["poly"] = np.round(np.asarray(poly, float).ravel(), 3).tolist()
    if label is not None:
        d["label"] = str(label)
    return d


def pane(title, bounds, items, image=None, note="", scale_um=10, anchor=None) -> dict:
    """anchor: world point the pane is centred on (default: the centre of bounds); bounds give the window size."""
    d = {"title": title, "bounds": [float(v) for v in bounds], "items": items, "image": image, "note": note, "scale_um": scale_um}
    if anchor is not None:
        d["anchor"] = [float(v) for v in np.asarray(anchor, float).ravel()[:2]]
    return d


def sync_spec(M, t) -> dict:
    """Left (H&E) world -> right (IF) world for the optional position sync."""
    M = np.asarray(M, float); t = np.asarray(t, float)
    Mi = np.linalg.inv(M)
    return {"A": M.tolist(), "t": t.tolist(), "Ainv": Mi.tolist(), "tinv": (-Mi @ t).tolist(), "scale": float(np.sqrt(abs(np.linalg.det(M))))}


def html(left: dict, right: dict, sync: dict | None = None, height: int = 420, sync_default: bool = False, view_key: str | None = None,
         he=None, active=None, window=None) -> str:
    """view_key / he / active / window: identify the view state kept across re-renders (see the module docstring)."""
    has_cells = any(it.get("k") == "cell" for p in (left, right) for it in p["items"])
    data = {"panes": [left, right], "sync": sync, "sync_default": bool(sync_default and sync), "styles": STYLES,
            "legend": {**LEGEND, **(CELL_LEGEND if has_cells else {})}, "view_key": view_key,
            "he": None if he is None else int(he), "active": None if active is None else int(active),
            "window": None if window is None else float(window)}
    return _HTML.replace("__DATA__", json.dumps(data, allow_nan=False)).replace("__H__", str(int(height)))
