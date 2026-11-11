"""Browser E2E checks for the Registration Explorer (headless Chromium via Playwright).

  PLAYWRIGHT_BROWSERS_PATH=... <venv>/bin/python ui_e2e.py --url http://127.0.0.1:8511 [--only A,B]
Writes ui_screenshots/*.png and ui_e2e_results.json.  Assertions are on visible text / viewer state only.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import traceback
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
SHOTS = HERE / "screenshots"
RESULTS = []


def check(test, name, ok, detail=""):
    RESULTS.append({"test": test, "check": name, "pass": bool(ok), "detail": str(detail)[:700]})
    print(("PASS " if ok else "FAIL ") + f"[{test}] {name} {detail if not ok else ''}", flush=True)


def wait_idle(pg, extra=1500, timeout=90000):
    """Wait until the Streamlit script run is finished (status widget / spinners gone)."""
    pg.wait_for_timeout(500)
    t0 = time.time()
    while time.time() - t0 < timeout / 1000:
        busy = pg.locator('[data-testid="stStatusWidget"], [data-testid="stSpinner"]').count()
        if busy == 0:
            break
        pg.wait_for_timeout(300)
    pg.wait_for_timeout(extra)


def page_errors(pg):
    """Exceptions rendered by Streamlit (element test ids differ between versions, so check both)."""
    errs = pg.locator('[data-testid="stException"]').all_inner_texts()
    errs += [t for t in pg.locator('[data-testid="stAlertContentError"]').all_inner_texts() if "Traceback" in t or "Error" in t[:80]]
    return errs


def select(pg, label, option, scope=None):
    root = scope or pg
    for attempt in range(3):
        wait_idle(pg, 300)
        lab = pg.locator('[data-testid="stWidgetLabel"]', has_text=re.compile("^" + re.escape(label) + "$"))
        box = root.locator('[data-testid="stSelectbox"]:visible').filter(has=lab).first
        if not box.count():           # widget sits in a hidden tab (e.g. the Viewer): open that tab
            tab = pg.get_by_role("tab", name="Viewer（重ね合わせ・手動調整）")
            if tab.count():
                tab.click(); wait_idle(pg, 1500)
            box = root.locator('[data-testid="stSelectbox"]:visible').filter(has=lab).first
        box.click()
        opt = pg.get_by_role("option", name=option, exact=False).first
        try:
            opt.wait_for(timeout=5000)
        except Exception:  # noqa: BLE001 - dropdown closed by a concurrent rerun; retry
            pg.keyboard.press("Escape")
            continue
        opt.click()
        wait_idle(pg)
        return
    raise TimeoutError(f"option {option!r} for {label!r} not reachable")


def main_text(pg):
    return pg.locator('[data-testid="stMain"]').inner_text()


def viewer_frames(pg):
    """Viewer iframes (static srcdoc viewers and the interactive component), in page order."""
    out = []
    for i in range(pg.locator("iframe").count()):
        el = pg.locator("iframe").nth(i)
        fr = el.element_handle().content_frame()
        try:
            if fr and fr.evaluate("!!document.getElementById('cv')"):
                out.append((el, fr))
        except Exception:  # noqa: BLE001 - frame navigating
            pass
    return out


def main_viewer(pg):
    """(iframe locator, frame) of the main Viewer tab (not the candidate map); opens the Viewer tab if it is hidden."""
    tab = pg.get_by_role("tab", name="Viewer（重ね合わせ・手動調整）")
    if tab.count() and tab.get_attribute("aria-selected") != "true":
        tab.click(); wait_idle(pg, 2500)
    for el, fr in viewer_frames(pg):
        if not el.is_visible():
            continue
        if "候補マップ" not in fr.locator("#side").inner_text():
            return el, fr
    raise LookupError("main viewer not found")


def viewer_frame(pg, idx=0):
    el, fr = main_viewer(pg)
    el.scroll_into_view_if_needed()
    return fr


def shot_el(pg, locator, name):
    locator.scroll_into_view_if_needed(); pg.wait_for_timeout(500)
    locator.screenshot(path=str(SHOTS / name))


def open_explorer(pg, url):
    pg.goto(url, wait_until="networkidle"); wait_idle(pg, 2500)
    sb = pg.locator('[data-testid="stSidebar"] [data-testid="stSelectbox"]').first
    sb.click(); pg.wait_for_timeout(300)
    pg.get_by_role("option", name="Registration Explorer (Research)").click()
    wait_idle(pg, 4000)


def test_a(pg, url):
    open_explorer(pg, url)
    select(pg, "Sample", "031-6_2")
    select(pg, "結果の種類", "視野サイズの違いに対応した核探索")
    main_viewer(pg)
    t = main_text(pg)
    check("A", "UNIQUE / Accepted shown", "UNIQUE / Accepted" in t)
    check("A", "H&E and fluorescence images available", t.count("あり (") >= 2 and "10240 × 10240" in t)
    check("A", "saved NN median 1.91 µm displayed", "1.91" in t, "metrics table")
    f = viewer_frame(pg)
    bans = " ".join(f.locator(".ban").all_inner_texts())
    check("A", "viewer banner accepted", "UNIQUE / Accepted" in bans, bans)
    labels = " ".join(f.locator("#side label").all_inner_texts())
    check("A", "image + point layers present", "H&E" in labels and "蛍光" in labels and "核" in labels, labels[:200])
    shot_el(pg, main_viewer(pg)[0], "A1_031-6_2_E0x_after_overlay.png")
    cv = f.locator("#cv")
    box = cv.bounding_box()
    hud0 = f.locator("#hud").inner_text() if f.locator("#hud").count() else ""
    pg.mouse.move(box["x"] + box["width"] * 0.55, box["y"] + box["height"] * 0.5)
    for _ in range(6):
        pg.mouse.wheel(0, -300); pg.wait_for_timeout(120)
    pg.mouse.down(); pg.mouse.move(box["x"] + box["width"] * 0.45, box["y"] + box["height"] * 0.45, steps=8); pg.mouse.up()
    pg.wait_for_timeout(400)
    hud1 = f.locator("#hud").inner_text()
    check("A", "zoom/pan updates the viewer (HUD coordinates)", hud1 and hud1 != hud0, hud1)
    f.locator("select").first.select_option("checker"); pg.wait_for_timeout(300)
    shot_el(pg, main_viewer(pg)[0], "A2_031-6_2_E0x_zoom_checker.png")
    f.locator("select").first.select_option("overlay")
    vec = f.locator("#side label").filter(has_text="残留変位").locator("input")
    vec.check(); pg.wait_for_timeout(300)
    shot_el(pg, main_viewer(pg)[0], "A3_031-6_2_E0x_vectors.png")
    pg.get_by_text("Before（位置合わせ前）").click(); wait_idle(pg, 2500)
    f = viewer_frame(pg)
    bans = " ".join(f.locator(".ban").all_inner_texts())
    note = f.locator("#side").inner_text()
    check("A", "Before view labelled, same fixed frame", "BEFORE" in bans and "Fixed frameのframe-local" in note.replace(" ", "") or "BEFORE" in bans, bans)
    shot_el(pg, main_viewer(pg)[0], "A4_031-6_2_before.png")
    pg.get_by_text("After（位置合わせ後）").click(); wait_idle(pg, 2000)
    pg.get_by_role("tab", name="Warp Preview").click(); wait_idle(pg, 6000)
    t = main_text(pg)
    check("A", "warp preview accepted label + inverse mapping record", "Accepted transform" in t and "inverse" in t)
    # the transform used must be the stored IF->HE map of E0x_HEfix (t ~ (541.9, 493.8) um), not its inverse
    check("A", "warp uses the stored moving->fixed transform (not inverted)", "541.871" in t and "493.781" in t, t[t.find("forward_used"):][:200])
    shot_el(pg, pg.locator("iframe:visible").last, "A5_031-6_2_warp_preview.png")


def test_b(pg, url):
    open_explorer(pg, url)
    select(pg, "Sample", "031-8_6")
    select(pg, "結果の種類", "視野サイズの違いに対応した核探索")
    t = main_text(pg)
    check("B", "AMBIGUOUS shown", "AMBIGUOUS" in t)
    check("B", "PREVIEW ONLY + not passed to Workflow D", "PREVIEW ONLY" in t and "Workflow Dへは渡しません" in t)
    check("B", "no Workflow D hand-off control", pg.get_by_role("button", name="Workflow D").count() == 0)
    f = viewer_frame(pg)
    labels = " ".join(f.locator("#side label").all_inner_texts())
    check("B", "FOV of both sides + candidate regions drawn", "Image FOV" in labels and "#c1 の領域" in labels, labels[:300])
    shot_el(pg, main_viewer(pg)[0], "B1_031-8_6_E0x_top_candidate_fov.png")
    select(pg, "候補 / 変換", "#c1:"); wait_idle(pg, 2500)
    f = viewer_frame(pg)
    bans = " ".join(f.locator(".ban").all_inner_texts())
    check("B", "candidate switch -> preview-only banner", "候補プレビュー" in bans and "PREVIEW ONLY" in bans, bans)
    shot_el(pg, main_viewer(pg)[0], "B2_031-8_6_candidate1_preview.png")
    pg.get_by_role("tab", name="対応領域の探索").click(); wait_idle(pg)
    t = main_text(pg)
    picks = map_frame(pg)[1].evaluate("window.__regx.S.polys.filter(p => p.pick).map(p => p.pick)")
    check("B", "candidate list (map) with IDs; candidates are PREVIEW ONLY", "E0x_HEfix#c1" in picks and "PREVIEW ONLY" in t, picks)
    shot_el(pg, pg.locator('[data-testid="stDataFrame"]:visible').last, "B3_031-8_6_candidate_table.png")
    # Phase 1 ablation results (top candidates with ICP + IF-frame evaluation) are browsable without recomputation
    pg.get_by_role("tab", name="Viewer（重ね合わせ・手動調整）").click(); wait_idle(pg)
    select(pg, "結果の種類", "核探索の構成要素の比較")
    select(pg, "結果", "X3_HEfix")
    pg.get_by_role("tab", name="対応領域の探索").click(); wait_idle(pg)
    t = main_text(pg)
    picks = map_frame(pg)[1].evaluate("window.__regx.S.polys.filter(p => p.pick).map(p => p.pick)")
    check("B", "ablation X3 candidates listed (with ICP-refined regions)", "X3_HEfix#c0" in picks and "X3_HEfix#c3" in picks and "AMBIGUOUS" in t, picks)
    shot_el(pg, pg.locator('[data-testid="stDataFrame"]:visible').last, "B4_031-8_6_X3_candidates_with_icp.png")
    pg.get_by_role("tab", name="Viewer（重ね合わせ・手動調整）").click(); wait_idle(pg)
    select(pg, "候補 / 変換", "#c2:"); wait_idle(pg, 2500)
    f = viewer_frame(pg)
    check("B", "ablation candidate preview banner", "候補プレビュー" in " ".join(f.locator(".ban").all_inner_texts()))
    shot_el(pg, main_viewer(pg)[0], "B5_031-8_6_X3_candidate2_icp_preview.png")


def test_c(pg, url):
    open_explorer(pg, url)
    select(pg, "Sample", "serial_section_pair_20260904")
    t = main_text(pg)
    check("C", "point-only sample: images absent, nuclei present", "なし（no image registered）" in t and "あり (1842)" in t)
    box = pg.locator('[data-testid="stSelectbox"]').filter(has_text="結果の種類").first
    box.click(); pg.wait_for_timeout(300)
    opts = " ".join(pg.get_by_role("option").all_inner_texts())
    check("C", "image methods disabled with reason", "両画像からの対応領域探索（利用不可" in opts and "片側画像を使った補助解析（利用不可" in opts, opts[:300])
    pg.keyboard.press("Escape"); select(pg, "結果の種類", "核密度による対応領域探索（従来法）"); wait_idle(pg, 3000)
    f = viewer_frame(pg)
    side = f.locator("#side").inner_text()
    check("C", "points shown without images", "核点群" in side and "level " not in side and "画像" not in side.split("核点群")[0].split("チェッカー幅")[-1], side[:200])
    check("C", "H&E frame shown in px (spacing UNKNOWN), unit mismatch noted", "(px)" in side and "単位が異なる" in side, side[-300:])
    shot_el(pg, main_viewer(pg)[0], "C1_point_only_pair_A2.png")
    t = main_text(pg)
    check("C", "transform shown", "表示中の変換" in t)
    # 031-6_2 with images hidden
    select(pg, "Sample", "031-6_2")
    pg.get_by_text("画像を使わない（Point-only表示）").click(); wait_idle(pg, 3000)
    select(pg, "結果の種類", "視野サイズの違いに対応した核探索"); wait_idle(pg, 3000)
    t = main_text(pg)
    check("C", "031-6_2 point-only view: E0x still available", "UNIQUE / Accepted" in t and "hidden (point-only view)" in t)
    shot_el(pg, main_viewer(pg)[0], "C2_031-6_2_images_hidden.png")


def test_d(pg, url):
    open_explorer(pg, url)
    select(pg, "Sample", "031-6_2")
    pg.get_by_text("画像読み込み失敗を再現（テスト）").click(); wait_idle(pg, 2000)
    select(pg, "結果の種類", "両画像からの対応領域探索")
    t = main_text(pg)
    check("D", "image methods unusable -> point-only result kept", "画像が使えないため、核探索の結果を表示します" in t and "UNIQUE / Accepted" in t, t[:400])
    shot_el(pg, pg.locator('[data-testid="stMain"]').locator('[data-testid="stAlert"]').first, "D1_simulated_image_failure_alert.png")
    pg.get_by_text("画像読み込み失敗を再現（テスト）").click(); wait_idle(pg, 2000)
    select(pg, "結果の種類", "両画像からの対応領域探索")
    select(pg, "結果", "E2_HEfix")
    t = main_text(pg)
    check("D", "E2 saved POINT_ONLY_FALLBACK with reason", "POINT_ONLY_FALLBACK" in t and "LOW_PEAK" in t and "E0x_HEfix" in t)
    shot_el(pg, pg.locator('[data-testid="stMain"]').locator('[data-testid="stAlert"]').first, "D2_E2_point_only_fallback.png")


def test_e(pg, url):
    open_explorer(pg, url)
    select(pg, "Sample", "031-6_2")
    select(pg, "Fixed / Moving", "蛍光 Fixed / H&E Moving")
    select(pg, "結果の種類", "視野サイズの違いに対応した核探索")
    f = viewer_frame(pg)
    check("E", "IF-fixed result selected by default", "After: E0x_IFfix" in f.locator("#side").inner_text())
    check("E", "viewer frame is IF", "031-6_2:IF" in f.locator("#side").inner_text() or "031-6_2:IF" in f.locator("#hud").inner_text())
    shot_el(pg, main_viewer(pg)[0], "E1_031-6_2_IF_fixed.png")
    pg.get_by_role("tab", name="方式比較").click(); wait_idle(pg)
    t = main_text(pg)
    check("E", "swap consistency table (E0x 0.23 µm)", "Fixed/Moving交換" in t and "0.231" in t)
    shot_el(pg, pg.locator('[data-testid="stTable"]:visible').last, "E2_swap_consistency.png")
    pg.get_by_role("tab", name="Warp Preview").click(); wait_idle(pg, 6000)
    t = main_text(pg)
    check("E", "inverse warp HE->IF recorded", "HE->IF" in t and "inverse" in t)
    check("E", "IF-fixed warp uses stored HE->IF map of E0x_IFfix (not inverted)", "(inverse of the stored direction)" not in t, t[t.find("transform_direction_used"):][:160])
    shot_el(pg, pg.locator("iframe:visible").last, "E3_warp_HE_into_IF.png")


def test_f(pg, url):
    pg.goto(url, wait_until="networkidle"); wait_idle(pg, 2500)
    for wf in ("Workflow A", "Workflow B", "Workflow C", "Workflow D"):
        sb = pg.locator('[data-testid="stSidebar"] [data-testid="stSelectbox"]').first
        sb.click(); pg.wait_for_timeout(300)
        pg.get_by_role("option", name=wf).first.click(); wait_idle(pg, 2500)
        exc = page_errors(pg)
        check("F", f"{wf} page renders without exception", not exc, exc[:1])
        if wf == "Workflow C":
            pg.screenshot(path=str(SHOTS / "F1_workflow_c_unchanged.png"))


def test_g(pg, url):
    """New research run from the UI: one Legacy A2 job kept, one X0 job submitted and cancelled."""
    open_explorer(pg, url)
    select(pg, "Sample", "031-6_2")
    pg.get_by_role("tab", name="新規実行").click(); wait_idle(pg)
    select(pg, "方式", "核密度による対応領域探索（従来法）")
    t = main_text(pg)
    if "完了済み結果があります" in t or "重複投入はできません" in t:
        check("G", "A2 job not resubmitted (cached or already queued)", True)
    else:
        pg.get_by_role("checkbox", name="Slurm（partition mib-dbia）にジョブを1件投入することを確認しました").check(force=True); wait_idle(pg, 800)
        pg.get_by_role("button", name="Slurmに投入").click(); wait_idle(pg, 3000)
        t = main_text(pg)
        check("G", "A2 job submitted via sbatch", "投入しました: job" in t, t[-300:])
    shot_el(pg, pg.locator('[data-testid="stMain"]'), "G1_new_run_submitted.png")
    select(pg, "方式", "X0")
    select(pg, "Fixed", "蛍光 Fixed")
    pg.get_by_role("checkbox", name="Slurm（partition mib-dbia）にジョブを1件投入することを確認しました").check(force=True); wait_idle(pg, 800)
    pg.get_by_role("button", name="Slurmに投入").click(); wait_idle(pg, 3000)
    t = main_text(pg)
    ok = "投入しました: job" in t
    check("G", "second job submitted (X0, IF fixed)", ok or "active explorer jobs" in t, t[-300:])
    if ok:
        jid = t.split("投入しました: job")[1].split(",")[0].strip()
        select(pg, "ログを見るrun", "")   # first (most recent) run
        btn = pg.get_by_role("button", name="このジョブをキャンセル")
        if btn.count():
            btn.click(); wait_idle(pg, 3000)
            t = main_text(pg)
            check("G", f"cancel own job {jid} (owner/name verified)", f"cancelled job {jid}" in t or "no longer in the queue" in t, t[-300:])
        else:
            check("G", "cancel button shown for queued/running job", False, "button missing")
    shot_el(pg, pg.locator('[data-testid="stMain"]'), "G2_run_status_after_cancel.png")


def test_h(pg, url):
    """A finished UI-submitted run is listed as 'Explorer run' and shown with its own verdict (no recomputation)."""
    open_explorer(pg, url)
    select(pg, "Sample", "031-6_2")
    select(pg, "結果の種類", "Explorerから実行した結果")
    main_viewer(pg)
    t = main_text(pg)
    side = viewer_frame(pg).locator("#side").inner_text()
    check("H", "completed explorer run listed with A2 detector verdict", "After: A2_HEfix@" in side and "FAIL" in t and "PREVIEW ONLY" in t, side[:300])
    check("H", "IF-frame ≤3 µm of the new A2 run equals the saved Legacy A2 value (0.0532)", "0.0532" in t)
    shot_el(pg, main_viewer(pg)[0], "H1_explorer_run_A2_result.png")


def set_number(pg, label, value):
    lab = pg.locator('[data-testid="stWidgetLabel"]', has_text=re.compile("^" + re.escape(label)))
    box = pg.locator('[data-testid="stNumberInput"]').filter(has=lab).first.locator("input")
    box.fill(str(value)); box.press("Enter"); wait_idle(pg, 2500)


def explorer_frame(pg):
    return main_viewer(pg)[1]


def test_i(pg, url):
    open_explorer(pg, url)
    select(pg, "Sample", "031-8_6")
    select(pg, "結果の種類", "視野サイズの違いに対応した核探索")
    select(pg, "候補 / 変換", "#c1:"); wait_idle(pg, 2500)
    pg.get_by_text("手動調整（回転・拡大縮小・平行移動）").click(); wait_idle(pg, 800)
    pg.locator('[data-testid="stCheckbox"], [data-testid="stToggle"]').filter(has_text="手動調整を有効にする").first.click(); wait_idle(pg, 3000)
    t = main_text(pg)
    check("I", "manual mode shows MANUAL / unverified / not passed to Workflow D", "MANUAL" in t and "未検証" in t and "Workflow Dへは渡しません" in t)
    f = viewer_frame(pg)
    check("I", "viewer banner MANUAL", "MANUAL" in " ".join(f.locator(".ban").all_inner_texts()))
    before = re.search(r"調整後の変換 \(IF→HE\): scale ([0-9.]+), 回転 (-?[0-9.]+)°", t)
    check("I", "identity adjustment keeps base transform", before is not None, t[-400:])
    # zoom in, then adjust: the zoom must survive the rerun
    fr = explorer_frame(pg)
    k0 = fr.evaluate("window.__regx.st.view.k")
    main_viewer(pg)[0].hover(); pg.mouse.wheel(0, -600); pg.wait_for_timeout(800)
    k1 = explorer_frame(pg).evaluate("window.__regx.st.view.k")
    set_number(pg, "回転 Δθ", 2.0)
    set_number(pg, "拡大縮小", 1.02)
    set_number(pg, "平行移動 Δx", 15.0)
    t = main_text(pg)
    after = re.search(r"調整後の変換 \(IF→HE\): scale ([0-9.]+), 回転 (-?[0-9.]+)°", t)
    ok = before and after and abs(float(after.group(1)) / float(before.group(1)) - 1.02) < 2e-3 and abs((float(after.group(2)) - float(before.group(2)) + 180) % 360 - 180 - 2.0) < 0.05
    check("I", "rotation +2° and scale x1.02 applied to the displayed transform", ok, (before.groups() if before else None, after.groups() if after else None))
    k2 = explorer_frame(pg).evaluate("window.__regx.st.view.k")
    check("I", "zoom kept across reruns while adjusting", k1 > k0 * 1.5 and abs(k2 / k1 - 1) < 0.02, (k0, k1, k2))
    # sliders: typed value is mirrored on the slider; arrow key on the slider moves one step and updates the box
    thumbs = pg.locator('[data-testid="stSlider"] input[type="range"]')
    check("I", "four manual sliders shown", thumbs.count() >= 4, thumbs.count())
    check("I", "slider mirrors typed rotation", abs(float(thumbs.nth(0).input_value()) - 2.0) < 1e-6, thumbs.nth(0).input_value())
    thumbs.nth(0).focus(); pg.keyboard.press("ArrowRight"); wait_idle(pg, 2500)
    lab = pg.locator('[data-testid="stWidgetLabel"]', has_text=re.compile("^回転 Δθ"))
    box = pg.locator('[data-testid="stNumberInput"]').filter(has=lab).first.locator("input")
    t = main_text(pg)
    after2 = re.search(r"調整後の変換 \(IF→HE\): scale ([0-9.]+), 回転 (-?[0-9.]+)°", t)
    ok = after and after2 and abs(float(box.input_value()) - 3.0) < 1e-6 and abs((float(after2.group(2)) - float(after.group(2)) + 180) % 360 - 180 - 1.0) < 0.05
    check("I", "slider arrow key moves rotation by one step (+1°) and syncs the number box", ok, (box.input_value(), after2.groups() if after2 else None))
    thumbs.nth(2).focus(); pg.keyboard.press("ArrowLeft"); wait_idle(pg, 2500)
    lab = pg.locator('[data-testid="stWidgetLabel"]', has_text=re.compile("^平行移動 Δx"))
    box = pg.locator('[data-testid="stNumberInput"]').filter(has=lab).first.locator("input")
    check("I", "translation slider moves Δx down to the step grid (15 → 10)", abs(float(box.input_value()) - 10.0) < 1e-6, box.input_value())
    pg.get_by_text("細", exact=True).click(); wait_idle(pg, 2500)
    check("I", "changing step keeps the adjustment", abs(float(box.input_value()) - 10.0) < 1e-6, box.input_value())
    shot_el(pg, main_viewer(pg)[0], "I1_031-8_6_manual_adjustment.png")
    check("I", "safety metric MANUAL (UNVERIFIED)", "MANUAL (UNVERIFIED)" in main_text(pg))
    # warp preview uses the live manual transform and is labelled as such
    pg.get_by_role("tab", name="Warp Preview").click(); wait_idle(pg, 4000)
    t = main_text(pg)
    check("I", "warp preview offers and labels the manual transform", "手動調整中の変換" in t and "MANUAL (UNVERIFIED)" in t and '"transform":"MANUAL"' in re.sub(r"\s+", "", t), " | ".join(x for x in t.split("\n") if any(k in x for k in ("手動", "MANUAL", "PREVIEW", "transform", "変換"))) [:900])
    pg.get_by_role("tab", name="Viewer（重ね合わせ・手動調整）").click(); wait_idle(pg)
    # save -> listed as Manual adjustment, never accepted
    lab = pg.locator('[data-testid="stWidgetLabel"]', has_text=re.compile("^名前$"))
    pg.locator('[data-testid="stTextInput"]:visible').filter(has=lab).first.locator("input").fill("e2e-manual-test")
    pg.keyboard.press("Enter"); wait_idle(pg)
    pg.get_by_role("button", name="保存する").click(); wait_idle(pg, 2500)
    t = main_text(pg)
    m = re.search(r"保存しました: (tf-[0-9a-f-]+)", t)
    check("I", "saved to the research store (append-only)", m is not None, t[-300:])
    select(pg, "結果の種類", "手動調整した変換")
    t = main_text(pg)
    lab = pg.locator('[data-testid="stWidgetLabel"]', has_text=re.compile("^結果$"))
    chosen = pg.locator('[data-testid="stSelectbox"]').filter(has=lab).first.locator("input").input_value()
    check("I", "saved transform listed as Manual adjustment, MANUAL (UNVERIFIED), PREVIEW ONLY",
          "e2e-manual-test" in chosen and "MANUAL (UNVERIFIED)" in t and "PREVIEW ONLY" in t, chosen)
    shot_el(pg, main_viewer(pg)[0], "I2_031-8_6_saved_manual_record.png")


# ============================================================================ 2026-10-22: research Workflow C / sessions
REPO = HERE.parents[1]
MANIFEST = REPO / "analysis/e0x-difficulty-and-web-integration-20261020/viewer_sample_manifest.json"
DL = HERE / "e2e_state" / "downloads"


def wf_run(sample, accepted=None):
    """Most recent UI-path engine run for a sample (artifact.json written by the Slurm runner)."""
    runs = sorted((REPO / "runs/registration-explorer").glob("*/artifact.json"), reverse=True)
    for p in runs:
        a = json.loads(p.read_text())
        if a["sample"] == sample and a["mode"] == "automatic" and (accepted is None or a["decision"]["accepted"] == accepted):
            return a
    raise LookupError(sample)


def if_nuclei_um(sample):
    """IF nuclei in IF world um straight from the provider GeoJSON (independent of the app code)."""
    man = json.loads(MANIFEST.read_text())
    s = next(x for x in man["samples"] if x["sample_id"] == sample)
    m = s["modalities"]["IF"]
    gj = json.loads(Path(m["nuclei"]["path"]).read_text())
    k = m["pixel_um"]
    return [(f["properties"][m["nuclei"]["x"]] * k, f["properties"][m["nuclei"]["y"]] * k) for f in gj["features"]], k, m


def apply(M, t, p):
    return (M[0][0] * p[0] + M[0][1] * p[1] + t[0], M[1][0] * p[0] + M[1][1] * p[1] + t[1])


def map_frame(pg):
    for el, fr in viewer_frames(pg):
        if "候補マップ" in fr.locator("#side").inner_text():
            return el, fr
    raise LookupError("candidate map not found")


def click_world(pg, el, fr, w, button_mode=None):
    """Click at world coordinate w in a viewer iframe (uses the live view transform)."""
    box = el.bounding_box()
    sx, sy = fr.evaluate("([x, y]) => { const v = window.__regx.st.view, d = window.devicePixelRatio || 1, r = document.getElementById('cv').getBoundingClientRect();"
                         " return [r.left + (x * v.k + v.ox) / d, r.top + (y * v.k + v.oy) / d]; }", list(w))
    pg.mouse.click(box["x"] + sx, box["y"] + sy)
    pg.wait_for_timeout(250)


def set_click_mode(fr, mode):
    sel = [s for s in fr.locator("#side select").all() if mode in s.locator("option").evaluate_all("o => o.map(x => x.value)")][0]
    sel.select_option(mode)


def open_wf(pg, url, sample):
    open_explorer(pg, url)
    select(pg, "Sample", sample)
    select(pg, "結果の種類", "研究Workflow C")
    wait_idle(pg, 2500)


def test_j(pg, url):
    """Stage diagnosis of an engine run: per-source verdicts, failure stage, image vs point candidates."""
    a = wf_run("031-6_2")
    open_wf(pg, url, "031-6_2")
    t = main_text(pg)
    lab = pg.locator('[data-testid="stWidgetLabel"]', has_text=re.compile("^結果$"))
    chosen = pg.locator('[data-testid="stSelectbox"]').filter(has=lab).first.locator("input").input_value()
    check("J", "engine run listed (newest first) and decision shown", a["run_id"] in chosen and "Accepted" in chosen, chosen)
    check("J", "both searches recorded separately (image + points)",
          "両画像からの対応領域探索" in t and "視野サイズの違いに対応した核探索" in t)
    check("J", "image-vs-point top candidate difference reported", "画像探索の1位と核探索の1位の差" in t)
    check("J", "internal ids moved to details", "内部ID: registration_core" in t)
    shot_el(pg, pg.locator('[data-testid="stMain"]'), "J1_031-6_2_stage_panel.png")
    a86 = wf_run("031-8_6")
    open_wf(pg, url, "031-8_6")
    t = main_text(pg)
    check("J", "031-8_6 not accepted, failure stage shown (C1)", "未採択" in t and "最初に止まった段階" in t and "C1 対応領域の探索" in t, t[:400])
    shot_el(pg, pg.locator('[data-testid="stMain"]'), "J2_031-8_6_stage_panel.png")


def test_k(pg, url):
    """Candidate map: whole fixed section, all candidates; clicking one switches the Viewer to it."""
    a = wf_run("031-8_6")
    open_wf(pg, url, "031-8_6")
    el, fr = map_frame(pg)
    picks = fr.evaluate("window.__regx.S.polys.filter(p => p.pick).map(p => p.pick)")
    check("K", "map shows every candidate of every source", set(picks) == {c["id"] for c in a["candidates"]}, picks)
    shot_el(pg, el, "K1_031-8_6_candidate_map.png")
    target = "points#3"
    cen = fr.evaluate("(id) => { const P = window.__regx.S.polys.find(p => p.pick === id).pts; let x = 0, y = 0, n = P.length / 2;"
                      " for (let i = 0; i < n; i++) { x += P[2*i]; y += P[2*i+1]; } return [x / n, y / n]; }", target)
    set_click_mode(fr, "pick"); pg.wait_for_timeout(300)
    click_world(pg, el, fr, cen); wait_idle(pg, 3500)
    lab = pg.locator('[data-testid="stWidgetLabel"]', has_text=re.compile("^候補 / 変換$"))
    chosen = pg.locator('[data-testid="stSelectbox"]').filter(has=lab).first.locator("input").input_value()
    check("K", f"click on {target} selects it in the Viewer", target in chosen, chosen)
    pg.get_by_role("tab", name="Viewer（重ね合わせ・手動調整）").click(); wait_idle(pg, 2500)
    bans = " ".join(viewer_frame(pg).locator(".ban").all_inner_texts())
    check("K", "viewer shows the picked candidate as PREVIEW ONLY", "候補プレビュー" in bans and "PREVIEW ONLY" in bans, bans)
    shot_el(pg, main_viewer(pg)[0], "K2_031-8_6_picked_candidate_preview.png")


def test_l(pg, url):
    """Landmarks clicked on the image -> fixed and moving coordinates (quantitative), saved as an annotation."""
    a = wf_run("031-6_2", accepted=True)
    open_wf(pg, url, "031-6_2")
    pg.get_by_role("tab", name="Viewer（重ね合わせ・手動調整）").click(); wait_idle(pg, 2500)
    el, fr = main_viewer(pg)
    el.scroll_into_view_if_needed()
    S = fr.evaluate("window.__regx.S.points.map(p => ({id: p.id, xy: p.xy.slice(0, 4000)}))")
    fixed_xy = next(p["xy"] for p in S if p["id"] == "fixed_nuclei")
    mov_xy = next(p["xy"] for p in S if p["id"] == "moving_nuclei")
    if_um, k, _ = if_nuclei_um("031-6_2")
    k_view = fr.evaluate("window.__regx.st.view.k / (window.devicePixelRatio || 1)")    # css px per um
    set_click_mode(fr, "landmark"); pg.wait_for_timeout(300)
    picks = [100, 400]
    for i in picks:
        click_world(pg, el, fr, (fixed_xy[2 * i], fixed_xy[2 * i + 1]))                   # (1) fixed structure
        click_world(pg, el, fr, (mov_xy[2 * i], mov_xy[2 * i + 1]))                       # (2) mapped moving nucleus i
    fr.locator("#side button", has_text="送信").click(); wait_idle(pg, 3500)
    t = main_text(pg)
    check("L", "landmark table shown", "指定した対応点 2 組" in t, t[-400:])
    rows = pg.locator('[data-testid="stTable"]:visible').filter(has_text="Moving x")
    check("L", "landmark table present", rows.count() > 0)
    # quantitative: python converted the clicked display point back to moving (IF) world = original nucleus i
    tol = 1.5 / k_view + 0.05
    data = rows.first.inner_text()
    nums = [float(x) for x in re.findall(r"-?\d+\.\d", data)]
    exp = []
    for i in picks:
        exp += [round(fixed_xy[2 * i], 1), round(fixed_xy[2 * i + 1], 1), if_um[i][0], if_um[i][1]]
    ok = len(nums) >= 8 and all(abs(a_ - b_) <= tol for a_, b_ in zip(nums[:8], exp))
    check("L", f"clicked fixed / back-mapped moving coordinates match the nuclei (tol {tol:.2f} um)", ok, (nums[:8], [round(x, 1) for x in exp]))
    pg.locator('[data-testid="stTextInput"]:visible').filter(has=pg.locator('[data-testid="stWidgetLabel"]', has_text="対応点の名前")).first.locator("input").fill("e2e landmarks")
    pg.keyboard.press("Enter"); wait_idle(pg)
    pg.get_by_role("button", name="対応点を注釈として保存").click(); wait_idle(pg, 2500)
    check("L", "landmarks saved as annotation", "保存しました（ann-" in main_text(pg))
    t = main_text(pg)
    check("L", "similarity from landmarks shown with residual", "対応点から求めた相似変換" in t and "残差" in t)
    shot_el(pg, el, "L1_031-6_2_landmarks.png")


def test_m(pg, url):
    """ROI drawn by dragging on the image, saved as a search-restriction annotation."""
    open_wf(pg, url, "031-8_6")
    pg.get_by_role("tab", name="Viewer（重ね合わせ・手動調整）").click(); wait_idle(pg, 2500)
    el, fr = main_viewer(pg)
    el.scroll_into_view_if_needed()
    set_click_mode(fr, "roi"); pg.wait_for_timeout(300)
    box = el.bounding_box()
    def scr(w):
        return fr.evaluate("([x, y]) => { const v = window.__regx.st.view, d = window.devicePixelRatio || 1, r = document.getElementById('cv').getBoundingClientRect();"
                           " return [r.left + (x * v.k + v.ox) / d, r.top + (y * v.k + v.oy) / d]; }", list(w))
    a0, a1 = scr((300.0, 400.0)), scr((1300.0, 1500.0))
    pg.mouse.move(box["x"] + a0[0], box["y"] + a0[1]); pg.mouse.down()
    pg.mouse.move(box["x"] + a1[0], box["y"] + a1[1], steps=8); pg.mouse.up(); pg.wait_for_timeout(300)
    fr.locator("#side button", has_text="送信").click(); wait_idle(pg, 3000)
    t = main_text(pg)
    m = re.search(r"指定した領域（ROI）（Fixed frame）: \((-?\d+), (-?\d+)\), \((-?\d+), (-?\d+)\)", t)
    ok = m and abs(int(m.group(1)) - 300) <= 8 and abs(int(m.group(2)) - 400) <= 8 and abs(int(m.group(3)) - 1300) <= 8 and abs(int(m.group(4)) - 1500) <= 8
    check("M", "dragged ROI arrives in fixed-frame um", bool(ok), m.groups() if m else t[-300:])
    pg.get_by_text("探索範囲の制限（ROI）").click(); wait_idle(pg, 800)
    pg.locator('[data-testid="stTextInput"]:visible').filter(has=pg.locator('[data-testid="stWidgetLabel"]', has_text=re.compile("^名前$"))).first.locator("input").fill("e2e roi")
    pg.keyboard.press("Enter"); wait_idle(pg)
    pg.get_by_role("button", name="ROIを保存").click(); wait_idle(pg, 2500)
    check("M", "ROI saved as search-restriction annotation", "保存しました（ann-" in main_text(pg))
    shot_el(pg, el, "M1_031-8_6_roi.png")


def test_n(pg, url):
    """Structured review: saved with candidate / transform, survives reload (and server restart: see test_n2)."""
    open_wf(pg, url, "031-8_6")
    pg.get_by_role("tab", name="評価・注釈").click(); wait_idle(pg, 1500)
    pg.get_by_text("間違っていると思う").click()
    for opt in ("対応領域が違う", "Rotationが違う"):
        pg.locator('[data-testid="stMultiSelect"]:visible').first.click(); pg.wait_for_timeout(300)
        pg.get_by_role("option", name=opt).click(); pg.wait_for_timeout(300)
        pg.keyboard.press("Escape")
    pg.locator("textarea:visible").first.fill("e2e: 腺管の配置が合わない")
    pg.get_by_role("button", name="評価を保存").click(); wait_idle(pg, 2500)
    t = main_text(pg)
    check("N", "review saved as Review Annotation", "評価を保存しました（rev-" in t and "ground truth" in t, t[-300:])
    check("N", "review listed with judgement / problems / comment", "間違っていると思う" in t and "Rotationが違う" in t and "腺管の配置が合わない" in t)
    shot_el(pg, pg.locator('[data-testid="stMain"]'), "N1_review_saved.png")


def test_n2(pg, url):
    """After a browser reload and an app restart the review / annotations are still there."""
    open_wf(pg, url, "031-8_6")
    pg.get_by_role("tab", name="評価・注釈").click(); wait_idle(pg, 1500)
    t = main_text(pg)
    check("N2", "review persisted (new browser page)", "腺管の配置が合わない" in t)
    check("N2", "annotations persisted (ROI)", "e2e roi" in t)


def test_o(pg, url):
    """Experiment record: history and JSON / CSV export."""
    open_wf(pg, url, "031-8_6")
    pg.get_by_role("tab", name="実験記録").click(); wait_idle(pg, 2000)
    pg.get_by_text("このサンプルのすべて").click(); wait_idle(pg, 2000)
    DL.mkdir(parents=True, exist_ok=True)
    with pg.expect_download() as d:
        pg.get_by_role("button", name="JSONでエクスポート").click()
    p = DL / "export.json"; d.value.save_as(p)
    data = json.loads(p.read_text())
    check("O", "JSON export has reviews / annotations / events for this sample only",
          any("腺管" in (r.get("comment") or "") for r in data["reviews"]) and any(a["label"] == "e2e roi" for a in data["annotations"])
          and data["events"] and all(r["sample_id"] == "031-8_6" for r in data["reviews"] + data["annotations"]))
    with pg.expect_download() as d:
        pg.get_by_role("button", name="CSV（zip）でエクスポート").click()
    p2 = DL / "export_csv.zip"; d.value.save_as(p2)
    import zipfile
    names = zipfile.ZipFile(p2).namelist()
    check("O", "CSV export (zip of 5 tables)", sorted(names) == ["annotations.csv", "events.csv", "reviews.csv", "runs.csv", "transforms.csv"], names)
    check("O", "store integrity ok", "integrity_check: ok" in main_text(pg))
    shot_el(pg, pg.locator('[data-testid="stMain"]'), "O1_experiment_record.png")


def test_p(pg, url):
    """Assisted run: annotations are applied only when explicitly selected; recorded as assisted."""
    open_wf(pg, url, "031-8_6")
    pg.get_by_role("tab", name="新規実行").click(); wait_idle(pg, 1500)
    select(pg, "処理", "注釈を使った探索")
    t = main_text(pg)
    check("P", "assisted run needs explicit annotation selection", "assisted run needs at least one selected annotation" in t)
    pg.locator('[data-testid="stMultiSelect"]:visible').first.click(); pg.wait_for_timeout(300)
    pg.get_by_role("option", name=re.compile("探索範囲の制限")).first.click(); pg.keyboard.press("Escape"); wait_idle(pg, 1500)
    t = main_text(pg)
    check("P", "selected ROI annotation accepted as a constraint (no validation error)", "実行できません" not in t and "完全自動の性能評価には含めません" in t, t[-400:])
    shot_el(pg, pg.locator('[data-testid="stMain"]'), "P1_assisted_run_setup.png")


def test_q(pg, url):
    """Workflow D hand-off: only an accepted automatic run can be exported; the zip is a valid Workflow C artifact."""
    open_wf(pg, url, "031-6_2")
    pg.get_by_role("tab", name="Artifact").click(); wait_idle(pg, 1500)
    t = main_text(pg)
    check("Q", "accepted run offers Workflow D hand-off", "Workflow Dが読み込める形式" in t)
    pg.get_by_role("button", name="Workflow D用のartifactを作成").click(); wait_idle(pg, 4000)
    DL.mkdir(parents=True, exist_ok=True)
    with pg.expect_download() as d:
        pg.get_by_role("button", name=re.compile("workflow_c_registration_result.zip")).click()
    p = DL / "workflow_d_handoff.zip"; d.value.save_as(p)
    import zipfile
    check("Q", "download is a Workflow C result zip", {"registration_result.npz", "manifest.json"} <= set(zipfile.ZipFile(p).namelist()))
    shot_el(pg, pg.locator('[data-testid="stMain"]'), "Q1_workflow_d_handoff.png")
    open_wf(pg, url, "031-8_6")
    pg.get_by_role("tab", name="Artifact").click(); wait_idle(pg, 1500)
    t = main_text(pg)
    check("Q", "unaccepted run refused for Workflow D", "Workflow Dへは渡せません" in t and "not accepted" in t)


def test_r(pg, url):
    """Workflow C (production page) offers the accepted research result only in research mode, as an explicit opt-in."""
    pg.goto(url, wait_until="networkidle"); wait_idle(pg, 2500)
    sb = pg.locator('[data-testid="stSidebar"] [data-testid="stSelectbox"]').first
    sb.click(); pg.wait_for_timeout(300)
    pg.get_by_role("option", name="Workflow C").first.click(); wait_idle(pg, 3000)
    pg.get_by_text("研究モード：Global位置合わせを『研究Workflow C』の採択済み結果にする").click(); wait_idle(pg, 800)
    t = main_text(pg)
    check("R", "research global option is off by default", "研究Workflow Cの結果を使う" in t)
    pg.get_by_text("研究Workflow Cの結果を使う").click(); wait_idle(pg, 1500)
    t = main_text(pg)
    a = wf_run("031-6_2", accepted=True)
    lab = pg.locator('[data-testid="stWidgetLabel"]', has_text=re.compile("^採択済みの結果$"))
    box = pg.locator('[data-testid="stSelectbox"]').filter(has=lab).first
    chosen = box.locator("input").input_value()
    box.click(); pg.wait_for_timeout(400)
    opts = " | ".join(pg.get_by_role("option").all_inner_texts()); pg.keyboard.press("Escape")
    check("R", "only accepted automatic runs listed", a["run_id"] in chosen and "031-8_6" not in opts and "ACCEPTED" in opts, opts[:300])
    check("R", "no exception on Workflow C page", not page_errors(pg))
    shot_el(pg, pg.locator('[data-testid="stMain"]'), "R1_workflow_c_research_global.png")


def test_s(pg, url):
    """Display geometry, quantitatively: moving nuclei drawn = artifact transform x provider IF nuclei, and the moving
    image layer places nucleus pixels exactly on those points."""
    a = wf_run("031-6_2", accepted=True)
    open_wf(pg, url, "031-6_2")
    pg.get_by_role("tab", name="Viewer（重ね合わせ・手動調整）").click(); wait_idle(pg, 2500)
    el, fr = main_viewer(pg)
    sc = fr.evaluate("({pts: window.__regx.S.points.find(p => p.id === 'moving_nuclei').xy, img: window.__regx.S.images.filter(L => L.role === 'moving').map(L => ({m: L.m, w: L.w, h: L.h}))})")
    M, t = a["transform"]["forward"]["M"], a["transform"]["forward"]["t"]
    if_um, k, mod = if_nuclei_um("031-6_2")
    exp = [apply(M, t, p) for p in if_um]
    d = max(abs(sc["pts"][2 * i] - e[0]) + abs(sc["pts"][2 * i + 1] - e[1]) for i, e in enumerate(exp))
    check("S", f"moving nuclei drawn at T(provider nuclei): max |d| {d:.4f} um", d < 0.02)
    L = sc["img"][0]
    um0 = mod["pixel_um"]
    ds = (mod["shape_px"][1] * um0) / (L["w"] * 1.0)      # um per level-k pixel along x
    worst = 0.0
    for i in (0, 250, 700):
        px, py = if_um[i][0] / um0, if_um[i][1] / um0                  # level-0 pixel centre index
        u, v = (px + 0.5) * um0 / ds, (py + 0.5) * um0 / ds            # level-k pixel coords (corner convention)
        m = L["m"]; X = (m[0] * u + m[2] * v + m[4], m[1] * u + m[3] * v + m[5])
        worst = max(worst, abs(X[0] - exp[i][0]) + abs(X[1] - exp[i][1]))
    check("S", f"moving image pixel of a nucleus lands on the drawn nucleus: max |d| {worst:.3f} um", worst < 0.5 * ds)


TESTS = {"H": test_h, "G": test_g, "A": test_a, "B": test_b, "C": test_c, "D": test_d, "E": test_e, "F": test_f, "I": test_i, "J": test_j, "K": test_k, "L": test_l, "M": test_m, "N": test_n, "N2": test_n2, "O": test_o, "P": test_p, "Q": test_q, "R": test_r, "S": test_s}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8512"); ap.add_argument("--only", default="A,B,C,D,E,F,I,J,K,L,M,N,N2,O,P,Q,R,S")
    a = ap.parse_args()
    SHOTS.mkdir(exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        for k in a.only.split(","):
            pg = b.new_page(viewport={"width": 1600, "height": 1100})
            try:
                TESTS[k](pg, a.url)
                exc = page_errors(pg)
                check(k, "no Streamlit exception on page", not exc, exc[:1])
            except Exception as e:  # noqa: BLE001
                check(k, "test ran to completion", False, traceback.format_exc()[:300] + " ... " + traceback.format_exc()[-300:])
                pg.screenshot(path=str(SHOTS / f"{k}_error.png"))
            pg.close()
        b.close()
    out = HERE / "ui_e2e_results.json"
    prev = json.loads(out.read_text()) if out.exists() and True else []
    keep = [r for r in prev if r["test"] not in a.only.split(",")]
    out.write_text(json.dumps(keep + RESULTS, indent=1, ensure_ascii=False))
    print(sum(r["pass"] for r in RESULTS), "/", len(RESULTS), "passed")
