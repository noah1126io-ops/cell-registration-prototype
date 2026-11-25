"""Real-browser checks of the Cell Review operation (headless Chromium via Playwright; 2026-11-25).

usage: e2e.py ROOT OUT_JSON [--dpr 1|2] [--port P] [--tabs] [--cells] [--shots DIR]
Starts browser/review_app.py with Streamlit on 127.0.0.1:P (ROOT must hold one precomputed auto result, see prepare.py),
drives it with real pointer / wheel / keyboard events and writes one record per check to OUT_JSON.
Every check reads the state back from the page (viewer JS state, the judgement panel text, the saved review file).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PY = "/mnt/mib-dbia08_1/users/noa.ganeko/miniconda3/envs/cell-rg/bin/python"
CH = Path.home() / ".cache/ms-playwright/chromium_headless_shell-1187/chrome-linux"
# the gateway's broken /dev/nvidia0 makes Chromium's GPU probe take minutes: software rendering only
BROWSER_ENV = {"VK_ICD_FILENAMES": str(CH / "vk_swiftshader_icd.json"), "__EGL_VENDOR_LIBRARY_DIRS": "/nonexistent"}
BROWSER_ARGS = ["--no-sandbox", "--disable-gpu", "--use-gl=angle", "--use-angle=swiftshader"]

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("out")
ap.add_argument("--dpr", type=float, default=1.0)
ap.add_argument("--port", type=int, default=8597)
ap.add_argument("--tabs", action="store_true")
ap.add_argument("--cells", action="store_true")
ap.add_argument("--shots", default=None)
ap.add_argument("--sample", default="031-6_2")
ap.add_argument("--inspect", action="store_true")
args = ap.parse_args()
os.environ.update(BROWSER_ENV)
RESULTS = []


def record(name, ok, **detail):
    RESULTS.append({"check": name, "ok": bool(ok), "dpr": args.dpr, **detail})
    print(("PASS " if ok else "FAIL ") + name, json.dumps(detail, ensure_ascii=False, default=str)[:400], flush=True)


def start_server():
    env = dict(os.environ, CRO_ROOT=args.root, CRO_SAMPLE=args.sample, CRO_TABS="1" if args.tabs else "", CRO_CELLS="1" if args.cells else "",
               CELLREG_CELL_REVIEW="1", CELLREG_RESEARCH_MODE="1", PYTHONDONTWRITEBYTECODE="1")
    log = open(Path(args.out).with_suffix(".server.log"), "w")
    p = subprocess.Popen([PY, "-m", "streamlit", "run", str(HERE / "review_app.py"), "--server.port", str(args.port), "--server.address", "127.0.0.1",
                          "--server.headless", "true", "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false"],
                         cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(240):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{args.port}/_stcore/health", timeout=2).read() == b"ok":
                return p
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("server did not start")


def settle(page, timeout=180.0, quiet=1.0):
    """Wait until Streamlit has finished running (no status widget for `quiet` s)."""
    page.wait_for_timeout(250)
    t0, calm = time.time(), None
    while time.time() - t0 < timeout:
        running = page.locator('[data-testid="stStatusWidget"]').count() > 0
        if running:
            calm = None
        elif calm is None:
            calm = time.time()
        elif time.time() - calm >= quiet:
            return
        page.wait_for_timeout(100)
    raise TimeoutError("streamlit run did not finish")


def picker(page):
    for f in page.frames:
        if "regx_cell_review_viewer" in f.url:
            return f
    raise LookupError("picker frame")


def compare(page):
    for f in page.frames:
        try:
            if f.evaluate("() => !!window.__cmp"):
                return f
        except Exception:
            pass
    return None


def wait_js(frame_get, expr, arg=None, timeout=60.0):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        try:
            last = frame_get().evaluate(expr, arg)
            if last:
                return last
        except Exception as exc:          # frame replaced during a rerun
            last = repr(exc)
        time.sleep(0.15)
    return last


def screen_of(page, w):
    """World µm (IF frame) -> page CSS px (iframe offset + canvas rect + view transform / devicePixelRatio)."""
    f = picker(page)
    box = f.frame_element().bounding_box()
    x, y = f.evaluate("""(w) => { const v = window.__regx.st.view, cv = document.getElementById('cv'), r = cv.getBoundingClientRect(),
        d = window.devicePixelRatio || 1; return [r.left + (w[0]*v.k + v.ox)/d, r.top + (w[1]*v.k + v.oy)/d]; }""", w)
    return box["x"] + x, box["y"] + y


def view(page):
    return picker(page).evaluate("""() => { const v = window.__regx.st.view, cv = document.getElementById('cv');
        return {k: v.k, cx: (cv.width/2 - v.ox)/v.k, cy: (cv.height/2 - v.oy)/v.k, W: cv.width, H: cv.height, dpr: window.devicePixelRatio || 1,
                cssW: cv.getBoundingClientRect().width}; }""")


def sel(page):
    return picker(page).evaluate("() => (typeof API !== 'undefined' && API) ? {he: API.selHe(), active: (API.sel || {}).active, n_emit: 0} : null")


def quick_text(page):
    loc = page.locator("div.st-key-regx-crv-quick")
    return loc.inner_text() if loc.count() else ""


def wait_quick(page, needle, timeout=90.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if needle in quick_text(page):
            return True
        page.wait_for_timeout(150)
    return False


def review_file():
    revs = sorted(Path(args.root).glob(f"{args.sample}/reviews/*/*.json"))
    return json.loads(revs[-1].read_text()) if revs else None


MODES0 = []


def display_modes(page):
    """{label: shown value} of the display-mode selectboxes (H&E表示 / IF表示)."""
    return page.evaluate("""() => { const out = {}; for (const e of document.querySelectorAll('[data-testid="stSelectbox"]')) {
        const lab = (e.querySelector('label') || {}).innerText || ''; if (!/表示/.test(lab) || /輪郭/.test(lab)) continue;
        const inp = e.querySelector('input'); const box = e.querySelector('[data-baseweb="select"]');
        out[lab.trim()] = ((box && box.innerText) || (inp && (inp.value || inp.getAttribute('aria-label'))) || '').trim(); } return out; }""")


def blur(page):
    page.evaluate("() => { if (document.activeElement) document.activeElement.blur(); }")


def shot(page, name):
    if args.shots:
        Path(args.shots).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(Path(args.shots) / f"{name}_dpr{int(args.dpr)}.jpg"), type="jpeg", quality=70, full_page=False)


def main():
    srv = start_server()
    try:
        with sync_playwright() as pw:
            b = pw.chromium.launch(args=BROWSER_ARGS)
            ctx = b.new_context(viewport={"width": 1680, "height": 1000}, device_scale_factor=args.dpr)
            page = ctx.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{args.port}/", wait_until="domcontentloaded")
            settle(page)
            if args.inspect:
                inspect(page)
            elif args.tabs:
                run_tabs(page)
            else:
                run_all(page)
            record("no JavaScript errors on the page", not errors, errors=errors[:5])
            b.close()
    finally:
        srv.terminate()
        srv.wait(20)
        Path(args.out).write_text(json.dumps(RESULTS, ensure_ascii=False, indent=1))


def nuclei(page):
    """In-ROI H&E nuclei shown in the viewer: [[index, x, y], ...] and the candidate table."""
    return picker(page).evaluate("""() => { const S = window.__regx.S, out = [];
        for (const L of S.points) if (L.pick === 'he') for (let i = 0; i < L.idx.length; i++) out.push([L.idx[i], L.xy[2*i], L.xy[2*i+1]]);
        return {he: out, cands: S.cand_table, ids: S.he_ids}; }""")


def isolated(he, cands, k=3, min_sep=9.0, need_cands=2):
    """Nuclei with no other H&E nucleus within min_sep µm, at least need_cands candidates, the 2nd candidate away from it."""
    import math
    out = []
    for i, x, y in he:
        c = cands.get(str(i), [])
        if len(c) < need_cands:
            continue
        if any(j != i and math.hypot(x - u, y - v) < min_sep for j, u, v in he):
            continue
        if need_cands >= 2 and math.hypot(c[1][1] - x, c[1][2] - y) < 3.0:
            continue
        out.append((i, x, y))
    out.sort(key=lambda r: (r[1], r[2]))
    return out[:: max(1, len(out) // (k + 2))][:k] if len(out) > k else out


def click_at(page, xy, dx=2.0, dy=-2.0, **kw):
    x, y = screen_of(page, xy)
    page.mouse.click(x + dx, y + dy, **kw)


def run_all(page):
    picker(page).frame_element().scroll_into_view_if_needed()
    page.wait_for_timeout(500)
    v0 = view(page)
    MODES0.append(display_modes(page))
    record("canvas uses devicePixelRatio (device px = CSS px x dpr)", abs(v0["W"] - v0["cssW"] * args.dpr) <= 2 and v0["dpr"] == args.dpr, view=v0)
    N = nuclei(page)
    picks = isolated(N["he"], N["cands"], k=4)
    assert len(picks) >= 3, picks
    (h1, x1, y1), (h2, x2, y2), (h3, x3, y3) = picks[:3]
    ids = N["ids"]
    picker(page).evaluate("() => { window.__regx.__marker = 'boot-1'; }")
    # ---- 1. left click selects the nearest H&E nucleus, without a mode switch; 2. focus / zoom to ~60 µm
    click_at(page, [x1, y1])
    s_now = sel(page)
    v1 = view(page)
    k60 = min(v1["W"], v1["H"]) / 60.0
    record("left click selects the H&E nucleus (instant, in the viewer)", s_now and s_now["he"] == h1, expected=h1, got=s_now)
    record("selected nucleus centred and zoomed to ~60 um", abs(v1["cx"] - x1) < 0.5 and abs(v1["cy"] - y1) < 0.5 and abs(v1["k"] / k60 - 1) < 0.02,
           view=v1, nucleus=[x1, y1], k60=k60)
    ok = wait_quick(page, f"H&E核 {ids[str(h1)]}")
    settle(page)
    record("judgement panel shows the clicked nucleus after one rerun", ok, quick=quick_text(page)[:300])
    record("quick panel offers the 4 verdicts and the candidates", all(t in quick_text(page) for t in ("対応確認 [1]", "未特定 [2]", "非観測 [3]", "判断不能 [4]", "#1 IF")),
           quick=quick_text(page)[:400])
    v1b = view(page)
    record("view kept after the rerun (no re-initialisation)", abs(v1b["k"] - v1["k"]) < 1e-6 and abs(v1b["cx"] - v1["cx"]) < 0.05, before=v1, after=v1b)
    marker = picker(page).evaluate("() => window.__regx.__marker || null")
    record("selection change redraws in place (viewer not rebooted)", marker == "boot-1", marker=marker)
    cm = compare(page)
    d = cm.evaluate("() => ({he: __cmp.D.he, active: __cmp.D.active, a0: __cmp.D.panes[0].anchor, a1: __cmp.D.panes[1].anchor})") if cm else None
    record("left/right comparison follows the selected nucleus", d and d["he"] == h1 and d["active"] is not None, compare=d)
    shot(page, "01_selected")
    # ---- 3. drag pans and never selects (press on another nucleus)
    xs, ys = screen_of(page, [x1 + 8.0, y1 + 8.0])
    page.mouse.move(xs, ys)
    page.mouse.down()
    for i in range(1, 11):
        page.mouse.move(xs + 6 * i, ys + 3 * i)
    page.mouse.up()
    page.wait_for_timeout(600)
    v2 = view(page)
    moved = (v2["cx"] - v1b["cx"]) * v2["k"] / args.dpr, (v2["cy"] - v1b["cy"]) * v2["k"] / args.dpr
    s2 = sel(page)
    settle(page)
    record("drag pans by the pointer travel (CSS px)", abs(moved[0] + 60) < 1.5 and abs(moved[1] + 30) < 1.5, moved_css_px=moved)
    record("drag release selects nothing", s2["he"] == h1 and f"H&E核 {ids[str(h1)]}" in quick_text(page), sel=s2)
    # a press that starts on a nucleus and is dragged must not select it either
    xs, ys = screen_of(page, [x2, y2]) if abs(x2 - x1) < 200 else (None, None)
    if xs is not None:
        page.mouse.move(xs, ys); page.mouse.down()
        for i in range(1, 6):
            page.mouse.move(xs + 4 * i, ys)
        page.mouse.up(); page.wait_for_timeout(500); settle(page)
        record("drag that starts on another nucleus does not select it", sel(page)["he"] == h1, sel=sel(page))
    # ---- 4. wheel zoom, then click: selection works and the user's zoom is kept
    xs, ys = screen_of(page, [x1, y1])
    page.mouse.move(xs, ys)
    page.mouse.wheel(0, -240)
    page.wait_for_timeout(400)
    v3 = view(page)
    record("wheel zooms about the cursor", v3["k"] > v2["k"] * 1.1, k_before=v2["k"], k_after=v3["k"])
    zoom_user = v3["k"]
    # pan to h2 so that it is on screen, then click it
    picker(page).evaluate("(q) => { API.centreOn(q); API.redraw(); }", [x2 + 5.0, y2 + 5.0])
    click_at(page, [x2, y2], dx=3.0, dy=1.0)
    v4 = view(page)
    s4 = sel(page)
    record("click after wheel zoom selects the right nucleus", s4["he"] == h2, expected=h2, got=s4)
    record("user zoom kept when moving to the next nucleus", abs(v4["k"] - zoom_user) < 1e-6 and abs(v4["cx"] - x2) < 0.5, view=v4, zoom_user=zoom_user)
    wait_quick(page, f"H&E核 {ids[str(h2)]}")
    settle(page)
    c2 = N["cands"][str(h2)]
    # ---- 6. candidate switch by clicking its ring (rank 2)
    f2 = c2[1]
    click_at(page, [f2[1], f2[2]], dx=1.0, dy=1.0)
    s6 = sel(page)
    record("click on a candidate ring switches the compared candidate (instant)", s6["active"] == f2[0] and s6["he"] == h2, expected=f2[0], got=s6)
    ok = wait_quick(page, "比較中の IF候補")
    settle(page)
    d6 = compare(page).evaluate("() => ({he: __cmp.D.he, active: __cmp.D.active, a1: __cmp.D.panes[1].anchor})")
    record("comparison right pane centred on the chosen candidate", d6["active"] == f2[0] and abs(d6["a1"][0] - f2[1]) < 0.01 and abs(d6["a1"][1] - f2[2]) < 0.01,
           compare=d6, cand=f2)
    radio_checked = page.locator("div.st-key-regx-crv-quick input[type=radio]:checked").count()
    record("judgement panel radio follows the candidate click", radio_checked == 1 and f"#{f2[3]} IF" in quick_text(page), quick=quick_text(page)[:300])
    shot(page, "02_candidate")
    # ---- 7. comparison zoom / pan kept across a rerun for the same nucleus, re-centred for another candidate
    cf = compare(page)
    cf.frame_element().scroll_into_view_if_needed()
    page.wait_for_timeout(300)
    box = cf.frame_element().bounding_box()
    r = cf.evaluate("() => { const c = __cmp.P[0].cv.getBoundingClientRect(); return [c.left + c.width/2, c.top + c.height/2]; }")
    page.mouse.move(box["x"] + r[0], box["y"] + r[1])
    page.mouse.wheel(0, -300)
    page.wait_for_timeout(300)
    rel0 = cf.evaluate("() => __cmp.P.map(p => p.view.k / p.k0())")
    # rerun without a nucleus change: tick the auto-advance checkbox (a widget) -> the comparison iframe is rebuilt
    page.locator("div.st-key-regx-crv-quick label:has-text('次の未レビュー核へ進む')").click()
    settle(page)
    cf2 = wait_js(lambda: compare(page) or page.main_frame, "() => window.__cmp ? __cmp.P.map(p => p.view.k / p.k0()) : null")
    record("comparison zoom kept across a rerun", rel0[0] > 1.2 and isinstance(cf2, list) and abs(cf2[0] - rel0[0]) < 1e-3, rel_before=rel0, rel_after=cf2)
    # the judgement panel stays in view while the page is scrolled down to the comparison
    qb = page.locator("div.st-key-regx-crv-quick").bounding_box()
    vh = page.viewport_size["height"]
    record("judgement panel visible while scrolled to the comparison", qb is not None and -2 <= qb["y"] < 150 and qb["y"] + qb["height"] <= vh + 2,
           panel_box=qb, viewport_h=vh)
    shot(page, "04_scrolled_compare")
    # another candidate: right pane re-centred, zoom kept; another nucleus: both re-centred, zoom kept
    f3 = c2[2] if len(c2) > 2 else c2[0]
    picker(page).frame_element().scroll_into_view_if_needed()
    click_at(page, [f3[1], f3[2]], dx=1.0, dy=1.0)
    settle(page)
    st3 = wait_js(lambda: compare(page) or page.main_frame, "() => window.__cmp ? {r: __cmp.restored, rel: __cmp.P.map(p => p.view.k / p.k0()), active: __cmp.D.active} : null")
    record("comparison: another candidate keeps the zoom, re-centres the IF pane", isinstance(st3, dict) and st3["active"] == f3[0] and st3["r"] == "kept-left"
           and abs(st3["rel"][0] - rel0[0]) < 1e-3, state=st3)
    # ---- 8. verdict shortcut records to the selected nucleus (focus in the viewer)
    page.locator("div.st-key-regx-crv-quick label:has-text('次の未レビュー核へ進む')").click()   # untick again
    settle(page)
    click_at(page, [x2, y2], dx=2.0, dy=0.0)
    page.wait_for_timeout(300)
    vb = view(page)
    picker(page).evaluate("() => { window.__regx.__marker = 'before-verdict'; }")
    page.keyboard.press("2")
    ok = wait_quick(page, f"記録しました: H&E核 {ids[str(h2)]}")
    settle(page)
    record("key 2 records 未特定 for the selected nucleus", ok and "未特定" in quick_text(page), quick=quick_text(page)[:300])
    va = view(page)
    record("view and selection kept after a verdict (scene re-rendered)", abs(va["k"] - vb["k"]) < 1e-6 and abs(va["cx"] - vb["cx"]) < 0.05 and sel(page)["he"] == h2,
           before=vb, after=va, sel=sel(page))
    # ---- 9. typing in the comment box never triggers shortcuts
    inp = page.locator("input[aria-label^='コメント（']")
    inp.click()
    page.keyboard.type("1234 テスト")
    page.keyboard.press("Enter")
    settle(page)
    val = page.locator("input[aria-label^='コメント（']").input_value()
    record("typing digits / Enter in the comment box is text, not shortcuts", val == "1234 テスト" and "記録しました" not in quick_text(page)
           and "未特定" in quick_text(page), value=val, quick=quick_text(page)[:200])
    inp.fill("")
    blur(page)
    settle(page)
    # ---- 10. key from the page around the viewer (focus outside the iframe) + Ctrl+Z / Ctrl+Y
    page.locator("div.st-key-regx-crv-quick").click(position={"x": 4, "y": 4})
    page.keyboard.press("4")
    ok = wait_quick(page, "判断不能")
    settle(page)
    record("key 4 from the page (outside the viewer) records 判断不能", ok and f"記録しました: H&E核 {ids[str(h2)]}" in quick_text(page), quick=quick_text(page)[:300])
    page.keyboard.press("Control+z")
    ok = wait_quick(page, "取り消しました")
    settle(page)
    record("Ctrl+Z undoes the last verdict (back to 未特定)", ok and "人のレビュー: 未特定" in quick_text(page).replace("\n", " ").replace("：", ":"),
           quick=quick_text(page)[:300])
    page.keyboard.press("Control+y")
    ok = wait_quick(page, "やり直しました")
    settle(page)
    record("Ctrl+Y redoes it", ok and "判断不能" in quick_text(page), quick=quick_text(page)[:300])
    # ---- 11. absent without a comment: nothing saved, focus moved to the comment box
    page.locator("div.st-key-regx-crv-quick").click(position={"x": 4, "y": 4})
    ops_before = len(review_ops(page))
    page.keyboard.press("3")
    ok = wait_quick(page, "非観測にはコメントが必要です")
    settle(page)
    page.wait_for_timeout(400)
    foc = page.evaluate("() => (document.activeElement && document.activeElement.getAttribute('aria-label')) || null")
    record("key 3 without a comment saves nothing and focuses the comment box", ok and (foc or "").startswith("コメント（") and len(review_ops(page)) == ops_before,
           focus=foc)
    page.keyboard.type("IF切片の同じ位置に核が見えない")
    page.keyboard.press("Tab")
    settle(page)
    page.locator("div.st-key-regx-crv-quick").click(position={"x": 4, "y": 4})
    page.keyboard.press("3")
    ok = wait_quick(page, "記録しました")
    settle(page)
    record("key 3 with a comment records 非観測", ok and "非観測" in quick_text(page), quick=quick_text(page)[:300])
    # ---- 12. auto-advance only after a recorded verdict; Enter = next unreviewed; Esc = deselect
    page.locator("div.st-key-regx-crv-quick label:has-text('次の未レビュー核へ進む')").click()
    settle(page)
    click_at(page, [x3, y3]) if on_screen(page, [x3, y3]) else (picker(page).evaluate("(q) => { API.centreOn(q); API.redraw(); }", [x3, y3]), click_at(page, [x3, y3]))
    wait_quick(page, f"H&E核 {ids[str(h3)]}")
    settle(page)
    page.keyboard.press("1")
    ok = wait_quick(page, "次の未レビュー核")
    settle(page)
    s12 = sel(page)
    vq = view(page)
    nxt_xy = next(([x, y] for i, x, y in N["he"] if i == s12["he"]), None)
    record("after a recorded verdict the next unreviewed nucleus is selected and focused", ok and s12["he"] not in (h3, None)
           and nxt_xy is not None and abs(vq["cx"] - nxt_xy[0]) < 0.5, sel=s12, view=vq, next_xy=nxt_xy)
    page.keyboard.press("2")             # 未特定 on the nucleus reached by auto-advance
    wait_quick(page, "記録しました")
    settle(page)
    s_adv = sel(page)
    page.locator("div.st-key-regx-crv-quick label:has-text('次の未レビュー核へ進む')").click()
    settle(page)
    page.locator("div.st-key-regx-crv-quick").click(position={"x": 4, "y": 4})
    page.keyboard.press("Enter")
    page.wait_for_timeout(500)
    settle(page)
    s13 = sel(page)
    record("Enter moves to the next unreviewed nucleus without recording", s13["he"] not in (None, s_adv["he"]) and "記録しました" not in quick_text(page),
           before=s_adv, after=s13)
    # Enter with the focus on a judgement button must not press that button
    btn = page.locator("div.st-key-regx-crv-quick button:has-text('判断不能 [4]')").first
    btn.focus()
    n0 = len(review_ops(page))
    page.keyboard.press("Enter")
    page.wait_for_timeout(500)
    settle(page)
    record("Enter on a focused verdict button only moves on (button not pressed)", len(review_ops(page)) == n0 and sel(page)["he"] != s13["he"],
           ops_before=n0, ops_after=len(review_ops(page)))
    page.keyboard.press("Escape")
    page.wait_for_timeout(300)
    s14 = sel(page)
    settle(page)
    record("Esc deselects (viewer and panel)", s14["he"] is None and "左クリック" in quick_text(page), sel=s14, quick=quick_text(page)[:200])
    # ---- 13. contours (and cells) still drawn; layer toggles survive the re-render
    st_ = picker(page).evaluate("""() => { const S = window.__regx.S; return {groups: (S.contours || []).length, nuc: (S.contours || []).filter(g => g.kind !== 'cell').length,
        cell: (S.contours || []).filter(g => g.kind === 'cell').length, opt: API.opt}; }""")
    picker(page).evaluate("(q) => { API.centreOn(q, API.focusK(60)); API.redraw(); }", [x1, y1])
    drawn = picker(page).evaluate("() => API.drawn")
    record("nucleus contours (and cell outlines) still drawn when zoomed in", st_["nuc"] >= 2 and drawn > 0 and st_["opt"]["contours"]
           and (st_["cell"] >= 1 if args.cells else True), state=st_, drawn=drawn)
    # ---- 14. display mode untouched by nucleus changes (value of the H&E表示 / IF表示 boxes, start vs end)
    m_end = display_modes(page)
    record("display mode (RGB / 核強調) unchanged by the review", bool(m_end) and m_end == MODES0[0], start=MODES0[0], end=m_end)
    # ---- 15. the saved review: every verdict on the intended nucleus, through Review.apply (ops + history)
    page.locator("div.st-key-regx-crv-quick button:has-text('レビューを保存')").first.click()
    settle(page)
    d = review_file()
    ops = [(o["op"], o["he_index"]) for o in d["ops"][: d["cursor"]]]
    want = [("not_identified", h2), ("uncertain", h2), ("absent", h2)]
    ok = ops[:3] == want and any(h == h3 for _, h in ops) and len(d["history"]) >= len(d["ops"]) + 2
    record("saved review: verdicts on the intended nuclei, undo/redo in the history", ok, ops=ops, history_actions=[h["action"] for h in d["history"]],
           schema=d["schema"])
    shot(page, "03_after")


def review_ops(page):
    """Ops currently in effect, read from the review JSON download (no save needed)."""
    with page.expect_download() as dl:
        page.locator("button:has-text('レビューJSON')").first.click()
    d = json.loads(Path(dl.value.path()).read_text())
    return d["ops"][: d["cursor"]]


def on_screen(page, xy):
    v = view(page)
    sx, sy = (xy[0] - v["cx"]) * v["k"] + v["W"] / 2, (xy[1] - v["cy"]) * v["k"] + v["H"] / 2
    return 20 < sx < v["W"] - 20 and 20 < sy < v["H"] - 20


def inspect(page):
    """Debug: DOM around the judgement panel (classes, position, scroll containers) and the button labels."""
    picker(page).frame_element().scroll_into_view_if_needed()
    page.wait_for_timeout(500)
    N = nuclei(page)
    h, x, y = isolated(N["he"], N["cands"], k=1)[0]
    click_at(page, [x, y])
    wait_quick(page, f"H&E核 {N['ids'][str(h)]}")
    settle(page)
    print(page.evaluate("""() => { const q = document.querySelector('[class*="st-key-regx-crv-quick"]'); if (!q) return 'no quick';
       const out = []; let e = q; for (let i = 0; i < 12 && e; i++) { const cs = getComputedStyle(e); const r = e.getBoundingClientRect();
         out.push(e.tagName + ' [' + (e.getAttribute('data-testid') || '') + '] ' + String(e.className).slice(0, 90) + ' pos=' + cs.position + ' ov=' + cs.overflowY
                  + ' align=' + cs.alignSelf + ' h=' + r.height.toFixed(0) + ' top=' + r.top.toFixed(0)); e = e.parentElement; } return out.join('\\n'); }"""))
    print(page.evaluate("() => Array.from(document.querySelectorAll('div[class*=st-key-regx-crv-quick] button')).map(b => JSON.stringify(b.innerText))"))
    print(page.evaluate("() => Array.from(document.querySelectorAll('style')).filter(s => s.textContent.includes('regx-crv-quick')).length"))


def run_tabs(page):
    """The tab is second: with the first tab shown, shortcuts do nothing; on the review tab they work."""
    page.get_by_role("tab", name="細胞対応レビュー（研究）").click()
    settle(page)
    picker(page).frame_element().scroll_into_view_if_needed()
    page.wait_for_timeout(500)
    N = nuclei(page)
    h, x, y = isolated(N["he"], N["cands"], k=1)[0]
    click_at(page, [x, y])
    wait_quick(page, f"H&E核 {N['ids'][str(h)]}")
    settle(page)
    page.get_by_role("tab", name="別のタブ").click()
    page.wait_for_timeout(500)
    blur(page)
    page.keyboard.press("2")
    page.keyboard.press("Enter")
    page.wait_for_timeout(1500)
    settle(page)
    page.get_by_role("tab", name="細胞対応レビュー（研究）").click()
    settle(page)
    page.wait_for_timeout(500)
    record("shortcuts do nothing while another tab is shown", "記録しました" not in quick_text(page) and sel(page)["he"] == h, quick=quick_text(page)[:200])
    page.locator("div.st-key-regx-crv-quick").click(position={"x": 4, "y": 4})
    page.keyboard.press("2")
    ok = wait_quick(page, "記録しました")
    record("shortcuts work again on the review tab", ok)
    # the dummy text box on the other tab: digits are text
    page.get_by_role("tab", name="別のタブ").click()
    page.locator("input[aria-label='dummy input']").fill("")
    page.locator("input[aria-label='dummy input']").type("12")
    record("typing in another tab's text box is unaffected", page.locator("input[aria-label='dummy input']").input_value() == "12")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        RESULTS.append({"check": "script finished", "ok": False, "error": traceback.format_exc()[-1500:]})
        Path(args.out).write_text(json.dumps(RESULTS, ensure_ascii=False, indent=1))
        sys.exit(1)
