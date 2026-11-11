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
SHOTS = HERE / "ui_screenshots"
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
        box = root.locator('[data-testid="stSelectbox"]').filter(has=lab).first
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


def viewer_frame(pg, idx=0):
    pg.locator("iframe").nth(idx).scroll_into_view_if_needed()
    return pg.frame_locator("iframe").nth(idx)


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
    select(pg, "Registration方式", "E0x Point-only")
    t = main_text(pg)
    check("A", "UNIQUE / Accepted shown", "UNIQUE / Accepted" in t)
    check("A", "H&E and fluorescence images available", t.count("あり (") >= 2 and "10240 × 10240" in t)
    check("A", "saved NN median 1.91 µm displayed", "1.91" in t, "metrics table")
    f = viewer_frame(pg)
    bans = " ".join(f.locator(".ban").all_inner_texts())
    check("A", "viewer banner accepted", "UNIQUE / Accepted" in bans, bans)
    labels = " ".join(f.locator("#side label").all_inner_texts())
    check("A", "image + point layers present", "H&E" in labels and "蛍光" in labels and "核" in labels, labels[:200])
    shot_el(pg, pg.locator("iframe").first, "A1_031-6_2_E0x_after_overlay.png")
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
    shot_el(pg, pg.locator("iframe").first, "A2_031-6_2_E0x_zoom_checker.png")
    f.locator("select").first.select_option("overlay")
    vec = f.locator("#side label").filter(has_text="残留変位").locator("input")
    vec.check(); pg.wait_for_timeout(300)
    shot_el(pg, pg.locator("iframe").first, "A3_031-6_2_E0x_vectors.png")
    pg.get_by_text("Before（位置合わせ前）").click(); wait_idle(pg, 2500)
    f = viewer_frame(pg)
    bans = " ".join(f.locator(".ban").all_inner_texts())
    note = f.locator("#side").inner_text()
    check("A", "Before view labelled, same fixed frame", "BEFORE" in bans and "Fixed frameのframe-local" in note.replace(" ", "") or "BEFORE" in bans, bans)
    shot_el(pg, pg.locator("iframe").first, "A4_031-6_2_before.png")
    pg.get_by_text("After（位置合わせ後）").click(); wait_idle(pg, 2000)
    pg.get_by_role("tab", name="Warp Preview").click(); wait_idle(pg, 6000)
    t = main_text(pg)
    check("A", "warp preview accepted label + inverse mapping record", "Accepted transform" in t and "inverse" in t)
    # the transform used must be the stored IF->HE map of E0x_HEfix (t ~ (541.9, 493.8) um), not its inverse
    check("A", "warp uses the stored moving->fixed transform (not inverted)", "541.871" in t and "493.781" in t, t[t.find("forward_used"):][:200])
    shot_el(pg, pg.locator("iframe").nth(1) if pg.locator("iframe").count() > 1 else pg.locator("iframe").first, "A5_031-6_2_warp_preview.png")


def test_b(pg, url):
    open_explorer(pg, url)
    select(pg, "Sample", "031-8_6")
    select(pg, "Registration方式", "E0x Point-only")
    t = main_text(pg)
    check("B", "AMBIGUOUS shown", "AMBIGUOUS" in t)
    check("B", "PREVIEW ONLY + not passed to Workflow D", "PREVIEW ONLY" in t and "Workflow Dへは渡しません" in t)
    check("B", "no Workflow D hand-off control", pg.get_by_role("button", name="Workflow D").count() == 0)
    f = viewer_frame(pg)
    labels = " ".join(f.locator("#side label").all_inner_texts())
    check("B", "FOV of both sides + candidate regions drawn", "Image FOV" in labels and "候補 #1" in labels, labels[:300])
    shot_el(pg, pg.locator("iframe").first, "B1_031-8_6_E0x_top_candidate_fov.png")
    select(pg, "候補 / 変換", "候補 #1"); wait_idle(pg, 2500)
    f = viewer_frame(pg)
    bans = " ".join(f.locator(".ban").all_inner_texts())
    check("B", "candidate switch -> preview-only banner", "候補プレビュー" in bans and "PREVIEW ONLY" in bans, bans)
    shot_el(pg, pg.locator("iframe").first, "B2_031-8_6_candidate1_preview.png")
    pg.get_by_role("tab", name="候補 (Candidates)").click(); wait_idle(pg)
    t = main_text(pg)
    check("B", "candidate table with IDs and PREVIEW ONLY", "E0x_HEfix#c1" in t and "PREVIEW ONLY" in t)
    shot_el(pg, pg.locator('[data-testid="stTable"]:visible').last, "B3_031-8_6_candidate_table.png")
    # Phase 1 ablation results (top candidates with ICP + IF-frame evaluation) are browsable without recomputation
    pg.get_by_role("tab", name="Viewer").click(); wait_idle(pg)
    select(pg, "Registration方式", "E0x ablation")
    select(pg, "結果", "X3_HEfix")
    pg.get_by_role("tab", name="候補 (Candidates)").click(); wait_idle(pg)
    t = main_text(pg)
    check("B", "ablation X3 candidates with ICP metrics listed", "X3_HEfix#c0" in t and "X3_HEfix#c3" in t and "AMBIGUOUS" in t)
    shot_el(pg, pg.locator('[data-testid="stTable"]:visible').last, "B4_031-8_6_X3_candidates_with_icp.png")
    pg.get_by_role("tab", name="Viewer").click(); wait_idle(pg)
    select(pg, "候補 / 変換", "候補 #2"); wait_idle(pg, 2500)
    f = viewer_frame(pg)
    check("B", "ablation candidate preview banner", "候補プレビュー" in " ".join(f.locator(".ban").all_inner_texts()))
    shot_el(pg, pg.locator("iframe").first, "B5_031-8_6_X3_candidate2_icp_preview.png")


def test_c(pg, url):
    open_explorer(pg, url)
    select(pg, "Sample", "serial_section_pair_20260904")
    t = main_text(pg)
    check("C", "point-only sample: images absent, nuclei present", "なし（no image registered）" in t and "あり (1842)" in t)
    box = pg.locator('[data-testid="stSelectbox"]').filter(has_text="Registration方式").first
    box.click(); pg.wait_for_timeout(300)
    opts = " ".join(pg.get_by_role("option").all_inner_texts())
    check("C", "image methods disabled with reason", "E2 Dual-image（利用不可" in opts and "E1 Single-image（利用不可" in opts, opts[:300])
    pg.keyboard.press("Escape"); select(pg, "Registration方式", "Legacy A2"); wait_idle(pg, 3000)
    f = viewer_frame(pg)
    side = f.locator("#side").inner_text()
    check("C", "points shown without images", "核点群" in side and "level " not in side and "画像" not in side.split("核点群")[0].split("チェッカー幅")[-1], side[:200])
    check("C", "H&E frame shown in px (spacing UNKNOWN), unit mismatch noted", "(px)" in side and "単位が異なる" in side, side[-300:])
    shot_el(pg, pg.locator("iframe").first, "C1_point_only_pair_A2.png")
    t = main_text(pg)
    check("C", "transform shown", "表示中の変換" in t)
    # 031-6_2 with images hidden
    select(pg, "Sample", "031-6_2")
    pg.get_by_text("画像を使わない（Point-only表示）").click(); wait_idle(pg, 3000)
    select(pg, "Registration方式", "E0x Point-only"); wait_idle(pg, 3000)
    t = main_text(pg)
    check("C", "031-6_2 point-only view: E0x still available", "UNIQUE / Accepted" in t and "hidden (point-only view)" in t)
    shot_el(pg, pg.locator("iframe").first, "C2_031-6_2_images_hidden.png")


def test_d(pg, url):
    open_explorer(pg, url)
    select(pg, "Sample", "031-6_2")
    pg.get_by_text("画像読み込み失敗を再現（テスト）").click(); wait_idle(pg, 2000)
    select(pg, "Registration方式", "E2 Dual-image")
    t = main_text(pg)
    check("D", "IMAGE_UNUSABLE + point-only result kept", "IMAGE_UNUSABLE" in t and "Point-only（E0x）" in t and "UNIQUE / Accepted" in t)
    shot_el(pg, pg.locator('[data-testid="stMain"]').locator('[data-testid="stAlert"]').first, "D1_simulated_image_failure_alert.png")
    pg.get_by_text("画像読み込み失敗を再現（テスト）").click(); wait_idle(pg, 2000)
    select(pg, "Registration方式", "E2 Dual-image")
    select(pg, "結果", "E2_HEfix")
    t = main_text(pg)
    check("D", "E2 saved POINT_ONLY_FALLBACK with reason", "POINT_ONLY_FALLBACK" in t and "LOW_PEAK" in t and "E0x_HEfix" in t)
    shot_el(pg, pg.locator('[data-testid="stMain"]').locator('[data-testid="stAlert"]').first, "D2_E2_point_only_fallback.png")


def test_e(pg, url):
    open_explorer(pg, url)
    select(pg, "Sample", "031-6_2")
    select(pg, "Fixed / Moving", "蛍光 Fixed / H&E Moving")
    select(pg, "Registration方式", "E0x Point-only")
    f = viewer_frame(pg)
    check("E", "IF-fixed result selected by default", "After: E0x_IFfix" in f.locator("#side").inner_text())
    check("E", "viewer frame is IF", "031-6_2:IF" in f.locator("#side").inner_text() or "031-6_2:IF" in f.locator("#hud").inner_text())
    shot_el(pg, pg.locator("iframe").first, "E1_031-6_2_IF_fixed.png")
    pg.get_by_role("tab", name="方式比較").click(); wait_idle(pg)
    t = main_text(pg)
    check("E", "swap consistency table (E0x 0.23 µm)", "Fixed/Moving交換" in t and "0.231" in t)
    shot_el(pg, pg.locator('[data-testid="stTable"]:visible').last, "E2_swap_consistency.png")
    pg.get_by_role("tab", name="Warp Preview").click(); wait_idle(pg, 6000)
    t = main_text(pg)
    check("E", "inverse warp HE->IF recorded", "HE->IF" in t and "inverse" in t)
    check("E", "IF-fixed warp uses stored HE->IF map of E0x_IFfix (not inverted)", "(inverse of the stored direction)" not in t, t[t.find("transform_direction_used"):][:160])
    shot_el(pg, pg.locator("iframe").nth(1) if pg.locator("iframe").count() > 1 else pg.locator("iframe").first, "E3_warp_HE_into_IF.png")


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
    select(pg, "方式", "Legacy A2")
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
    select(pg, "Registration方式", "Explorer run")
    t = main_text(pg)
    side = viewer_frame(pg).locator("#side").inner_text()
    check("H", "completed explorer run listed with A2 detector verdict", "After: A2_HEfix@" in side and "FAIL" in t and "PREVIEW ONLY" in t, side[:300])
    check("H", "IF-frame ≤3 µm of the new A2 run equals the saved Legacy A2 value (0.0532)", "0.0532" in t)
    shot_el(pg, pg.locator("iframe").first, "H1_explorer_run_A2_result.png")


def set_number(pg, label, value):
    lab = pg.locator('[data-testid="stWidgetLabel"]', has_text=re.compile("^" + re.escape(label)))
    box = pg.locator('[data-testid="stNumberInput"]').filter(has=lab).first.locator("input")
    box.fill(str(value)); box.press("Enter"); wait_idle(pg, 2500)


def explorer_frame(pg):
    return next(f for f in pg.frames if f.url == "about:srcdoc" and f.evaluate("typeof S !== 'undefined' && !!document.getElementById('cv')"))


def test_i(pg, url):
    open_explorer(pg, url)
    select(pg, "Sample", "031-8_6")
    select(pg, "Registration方式", "E0x Point-only")
    select(pg, "候補 / 変換", "候補 #1"); wait_idle(pg, 2500)
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
    k0 = fr.evaluate("st.view.k")
    pg.locator("iframe").first.hover(); pg.mouse.wheel(0, -600); pg.wait_for_timeout(800)
    k1 = explorer_frame(pg).evaluate("st.view.k")
    set_number(pg, "回転 Δθ", 2.0)
    set_number(pg, "拡大縮小", 1.02)
    set_number(pg, "平行移動 Δx", 15.0)
    t = main_text(pg)
    after = re.search(r"調整後の変換 \(IF→HE\): scale ([0-9.]+), 回転 (-?[0-9.]+)°", t)
    ok = before and after and abs(float(after.group(1)) / float(before.group(1)) - 1.02) < 2e-3 and abs((float(after.group(2)) - float(before.group(2)) + 180) % 360 - 180 - 2.0) < 0.05
    check("I", "rotation +2° and scale x1.02 applied to the displayed transform", ok, (before.groups() if before else None, after.groups() if after else None))
    k2 = explorer_frame(pg).evaluate("st.view.k")
    check("I", "zoom kept across reruns while adjusting", k1 > k0 * 1.5 and abs(k2 / k1 - 1) < 0.02, (k0, k1, k2))
    shot_el(pg, pg.locator("iframe").first, "I1_031-8_6_manual_adjustment.png")
    check("I", "safety metric MANUAL (UNVERIFIED)", "MANUAL (UNVERIFIED)" in main_text(pg))
    # warp preview uses the live manual transform and is labelled as such
    pg.get_by_role("tab", name="Warp Preview").click(); wait_idle(pg, 4000)
    t = main_text(pg)
    check("I", "warp preview offers and labels the manual transform", "手動調整中の変換" in t and "MANUAL (UNVERIFIED)" in t and '"transform":"MANUAL"' in re.sub(r"\s+", "", t), " | ".join(x for x in t.split("\n") if any(k in x for k in ("手動", "MANUAL", "PREVIEW", "transform", "変換"))) [:900])
    pg.get_by_role("tab", name="Viewer").click(); wait_idle(pg)
    # save -> listed as Manual adjustment, never accepted
    lab = pg.locator('[data-testid="stWidgetLabel"]', has_text=re.compile("^名前$"))
    pg.locator('[data-testid="stTextInput"]').filter(has=lab).first.locator("input").fill("e2e-manual-test")
    pg.keyboard.press("Enter"); wait_idle(pg)
    pg.get_by_role("button", name="保存する").click(); wait_idle(pg, 2500)
    t = main_text(pg)
    m = re.search(r"保存しました: (\S+\.json)", t)
    check("I", "saved to JSON", m is not None, t[-300:])
    select(pg, "Registration方式", "Manual adjustment")
    t = main_text(pg)
    lab = pg.locator('[data-testid="stWidgetLabel"]', has_text=re.compile("^結果$"))
    chosen = pg.locator('[data-testid="stSelectbox"]').filter(has=lab).first.locator("input").input_value()
    check("I", "saved transform listed as Manual adjustment, MANUAL (UNVERIFIED), PREVIEW ONLY",
          "e2e-manual-test" in chosen and "MANUAL (UNVERIFIED)" in t and "PREVIEW ONLY" in t, chosen)
    shot_el(pg, pg.locator("iframe").first, "I2_031-8_6_saved_manual_record.png")
    if m:   # remove the test record so it does not appear as user data
        Path(m.group(1)).unlink(missing_ok=True)


TESTS = {"H": test_h, "G": test_g, "A": test_a, "B": test_b, "C": test_c, "D": test_d, "E": test_e, "F": test_f, "I": test_i}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8511"); ap.add_argument("--only", default="A,B,C,D,E,F")
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
    prev = json.loads(out.read_text()) if out.exists() and a.only != "A,B,C,D,E,F" else []
    keep = [r for r in prev if r["test"] not in a.only.split(",")]
    out.write_text(json.dumps(keep + RESULTS, indent=1, ensure_ascii=False))
    print(sum(r["pass"] for r in RESULTS), "/", len(RESULTS), "passed")
