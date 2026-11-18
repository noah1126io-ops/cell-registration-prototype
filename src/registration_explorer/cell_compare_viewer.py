"""Side-by-side comparison of one H&E nucleus and its IF candidates (research, display only; 2026-11-18).

Left = H&E section, right = IF section, each in its **own, undeformed** frame (µm of that section; the original image crop
is never warped).  The global affine (H&E -> IF) is used only for the optional position sync of the two views.

Static component (components.html): zoom (wheel), pan (drag), local re-centre, contour / centre / ID / image toggles,
contour opacity.  Decisions are made with Streamlit buttons next to it, so no value is sent back.

Item styles (contours and points):
  sel        selected H&E nucleus                     yellow, thick
  active     candidate being compared                 magenta, thick
  cand       other candidates (rank label)            white, dashed
  target     IF nucleus picked in the main viewer     cyan, thick
  current    current partner in the review           green, thick
  faint      other nuclei in the window (neighbours)  thin, low opacity
"""
from __future__ import annotations

import base64
import json

import numpy as np

STYLES = {"sel": ("#ffeb3b", 2.6, []), "active": ("#ff40ff", 2.8, []), "cand": ("#ffffff", 1.5, [4, 3]), "target": ("#00e5ff", 2.6, []),
          "current": ("#00e676", 2.4, [2, 2]), "faint": ("#b0bec5", 0.9, [])}

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
const opt = {img: true, contours: true, centres: true, ids: true, sync: D.sync_default, alpha: 0.55};
const bar = document.getElementById('bar');
function chk(k, label){ const l = document.createElement('label'); const c = document.createElement('input'); c.type = 'checkbox'; c.checked = opt[k];
  c.onchange = () => { opt[k] = c.checked; if (k === 'sync' && opt.sync) syncFrom(0); drawAll(); }; l.appendChild(c); l.appendChild(document.createTextNode(' ' + label)); bar.appendChild(l); }
chk('img', '元画像'); chk('contours', '核輪郭'); chk('centres', '中心点'); chk('ids', 'ID'); if (D.sync) chk('sync', '位置を同期（Global Affine）');
{ const l = document.createElement('label'); l.textContent = '周辺の輪郭の濃さ '; const r = document.createElement('input'); r.type = 'range'; r.min = 0.05; r.max = 1; r.step = 0.05; r.value = opt.alpha;
  r.oninput = () => { opt.alpha = +r.value; drawAll(); }; l.appendChild(r); bar.appendChild(l); }
{ const b = document.createElement('button'); b.textContent = '選択核を中心に戻す'; b.onclick = () => { P.forEach(p => p.fit()); drawAll(); }; bar.appendChild(b); }
const leg = document.getElementById('leg');
leg.innerHTML = Object.entries(D.legend).map(([k, v]) => '<span class="sw" style="background:' + D.styles[k][0] + '"></span>' + v).join('') +
  '<br>ホイール: ズーム / ドラッグ: パン。左右とも未変形の元画像（各切片の座標）。';
const P = [];
function Pane(i, S){
  const wrap = document.createElement('div'); wrap.className = 'pane';
  wrap.innerHTML = '<div class="ttl"></div><div class="note"></div>'; wrap.querySelector('.ttl').textContent = S.title; wrap.querySelector('.note').textContent = S.note || '';
  const cv = document.createElement('canvas'); wrap.appendChild(cv); document.getElementById('panes').appendChild(wrap);
  const ctx = cv.getContext('2d'); const me = {S: S, cv: cv, view: null};
  let im = null; if (S.image) { im = new Image(); im.onload = () => draw(); im.src = S.image.src; }
  me.fit = () => { const r = cv.getBoundingClientRect(), d = window.devicePixelRatio || 1; cv.width = Math.max(2, r.width * d); cv.height = Math.max(2, r.height * d);
    const b = S.bounds, k = Math.min(cv.width / (b[2] - b[0]), cv.height / (b[3] - b[1]));
    me.view = {k: k, ox: cv.width / 2 - k * (b[0] + b[2]) / 2, oy: cv.height / 2 - k * (b[1] + b[3]) / 2}; };
  me.centre = () => { const v = me.view; return [(cv.width / 2 - v.ox) / v.k, (cv.height / 2 - v.oy) / v.k]; };
  me.setCentre = (c, k) => { me.view = {k: k, ox: cv.width / 2 - k * c[0], oy: cv.height / 2 - k * c[1]}; };
  const path = (Q) => { const v = me.view; ctx.beginPath(); for (let j = 0; j < Q.length; j += 2) { const x = Q[j] * v.k + v.ox, y = Q[j+1] * v.k + v.oy; j ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.closePath(); };
  function draw(){
    if (!me.view) me.fit(); const v = me.view;
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = '#000'; ctx.fillRect(0, 0, cv.width, cv.height);
    if (opt.img && im && im.complete) { const m = S.image.m; ctx.save(); ctx.setTransform(v.k, 0, 0, v.k, v.ox, v.oy); ctx.transform(m[0], m[1], m[2], m[3], m[4], m[5]);
      ctx.imageSmoothingEnabled = v.k * Math.hypot(m[0], m[1]) < 2; ctx.drawImage(im, 0, 0); ctx.restore(); }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    const items = S.items.slice().sort((a, b) => (a.s === 'faint') === (b.s === 'faint') ? 0 : (a.s === 'faint' ? -1 : 1));
    for (const it of items) { const st = D.styles[it.s]; const faint = it.s === 'faint';
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
    const f = Math.exp(-e.deltaY * 0.0015), v = me.view; v.ox = sx - (sx - v.ox) * f; v.oy = sy - (sy - v.oy) * f; v.k *= f; syncFrom(i); drawAll(); }, {passive: false});
  cv.addEventListener('pointerdown', e => { drag = {x: e.clientX, y: e.clientY, ox: me.view.ox, oy: me.view.oy}; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener('pointermove', e => { if (!drag) return; const d = window.devicePixelRatio || 1; me.view.ox = drag.ox + (e.clientX - drag.x) * d; me.view.oy = drag.oy + (e.clientY - drag.y) * d; syncFrom(i); drawAll(); });
  cv.addEventListener('pointerup', () => { drag = null; });
  return me;
}
function syncFrom(i){
  if (!opt.sync || !D.sync) return; const a = P[i], b = P[1 - i]; const c = a.centre(); const A = D.sync.A, t = D.sync.t, s = D.sync.scale;
  if (i === 0) b.setCentre([A[0][0] * c[0] + A[0][1] * c[1] + t[0], A[1][0] * c[0] + A[1][1] * c[1] + t[1]], a.view.k / s);
  else { const Ai = D.sync.Ainv, ti = D.sync.tinv; b.setCentre([Ai[0][0] * c[0] + Ai[0][1] * c[1] + ti[0], Ai[1][0] * c[0] + Ai[1][1] * c[1] + ti[1]], a.view.k * s); }
}
function drawAll(){ P.forEach(p => p.draw()); }
D.panes.forEach((S, i) => P.push(Pane(i, S)));
P.forEach(p => p.fit()); if (opt.sync) syncFrom(0); drawAll();
if (window.ResizeObserver) new ResizeObserver(() => { P.forEach(p => { const c = p.view ? p.centre() : null, k = p.view ? p.view.k : null; p.fit(); }); if (opt.sync) syncFrom(0); drawAll(); }).observe(document.body);
window.__cmp = {D: D, P: P, opt: opt};
</script></body></html>"""

LEGEND = {"sel": "選択中のH&E核", "active": "比較中の候補", "cand": "その他の候補（#順位）", "current": "現在の対応先", "target": "クリックしたIF核",
          "faint": "周辺の核"}


def image_entry(rgb: np.ndarray, um_per_px: float, origin_xy) -> dict:
    """Crop placed in the section frame: pixel (u, v) top-left corner -> origin + (u, v) * um_per_px."""
    from . import imaging
    jpg = imaging.encode_jpeg(np.ascontiguousarray(rgb), 90)
    return {"src": "data:image/jpeg;base64," + base64.b64encode(jpg).decode(), "w": int(rgb.shape[1]), "h": int(rgb.shape[0]),
            "m": [float(um_per_px), 0.0, 0.0, float(um_per_px), float(origin_xy[0]), float(origin_xy[1])]}


def item(style, centre=None, poly=None, label=None) -> dict:
    assert style in STYLES, style
    d = {"s": style}
    if centre is not None:
        d["c"] = np.round(np.asarray(centre, float), 3).tolist()
    if poly is not None:
        d["poly"] = np.round(np.asarray(poly, float).ravel(), 3).tolist()
    if label is not None:
        d["label"] = str(label)
    return d


def pane(title, bounds, items, image=None, note="", scale_um=10) -> dict:
    return {"title": title, "bounds": [float(v) for v in bounds], "items": items, "image": image, "note": note, "scale_um": scale_um}


def sync_spec(M, t) -> dict:
    """Left (H&E) world -> right (IF) world for the optional position sync."""
    M = np.asarray(M, float); t = np.asarray(t, float)
    Mi = np.linalg.inv(M)
    return {"A": M.tolist(), "t": t.tolist(), "Ainv": Mi.tolist(), "tinv": (-Mi @ t).tolist(), "scale": float(np.sqrt(abs(np.linalg.det(M))))}


def html(left: dict, right: dict, sync: dict | None = None, height: int = 420, sync_default: bool = False) -> str:
    data = {"panes": [left, right], "sync": sync, "sync_default": bool(sync_default and sync), "styles": STYLES, "legend": LEGEND}
    return _HTML.replace("__DATA__", json.dumps(data, allow_nan=False)).replace("__H__", str(int(height)))
