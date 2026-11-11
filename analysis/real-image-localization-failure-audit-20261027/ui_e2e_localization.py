"""Browser E2E for the 2026-10-27 localization diagnostics (reuses the 10-22 harness helpers).

  FM  feature maps the image search compares are shown (both modalities, occ / cleft included)
  RF  a saved manual transform is designated as the reference; candidate-vs-reference table; survives a reload
  LR  3 landmark pairs with type / confidence -> leave-one-out shown -> saved + designated as reference
  NR  new-run panel offers the image feature (default occ+cleft)
Run against the isolated test server (port 8512, own store) after the 10-22 test I (creates 'e2e-manual-test').
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "workflow-c-unification-and-registration-review-20261022"
sys.path.insert(0, str(OLD))
import ui_e2e as H  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

H.SHOTS = HERE / "screenshots"


def goto_map(pg, url, sample):
    H.open_wf(pg, url, sample)
    pg.get_by_role("tab", name="対応領域の探索").click(); H.wait_idle(pg, 2500)


def test_fm(pg, url):
    goto_map(pg, url, "031-8_6")
    pg.get_by_text("画像探索が実際に比較している画像（特徴マップ）").click(); H.wait_idle(pg, 800)
    pg.locator('[data-testid="stCheckbox"]').filter(has_text="特徴マップを計算して表示する").first.click(); H.wait_idle(pg, 20000, timeout=240000)
    t = H.main_text(pg)
    n = pg.locator('[data-testid="stImage"]:visible').count()
    H.check("FM", "8 feature images shown (tissue / nuclear / occ / cleft x HE / IF)", n >= 8, n)
    H.check("FM", "explains old IF tissue is nuclei-dominated and the new occ / cleft maps", "核（Hoechst）が支配的" in t and "occ（2026-10-27）" in t and "cleft（2026-10-27）" in t)
    H.check("FM", "quality table distinguishes readable / usable", "nuclear_map_auc_own_nuclei_vs_tissue" in t)
    H.shot_el(pg, pg.locator('[data-testid="stExpander"]').filter(has_text="特徴マップ").first, "FM1_031-8_6_feature_maps.png")


def test_rf(pg, url):
    goto_map(pg, url, "031-8_6")
    t = H.main_text(pg)
    H.check("RF", "reference panel states it is not ground truth", "Ground Truthではありません" in t)
    H.select(pg, "参照に使う変換", "e2e-manual-test")
    btn = pg.get_by_role("button", name="参照変換に指定")
    if btn.is_enabled():
        btn.click(); H.wait_idle(pg, 3000)
    t = H.main_text(pg)
    H.check("RF", "reference designated and described", "現在の参照" in t and "e2e-manual-test" in t, t[-600:])
    # st.dataframe renders on a canvas: check the heading, the verdict caption and the grid's row count
    grid = pg.locator('[data-testid="stDataFrame"]:visible')
    H.check("RF", "candidate vs reference table and verdict caption", "自動推定の候補と参照変換の比較" in t and ("参照領域に一致する候補" in t), t[-400:])
    H.check("RF", "comparison grids present (candidate list + reference table)", grid.count() >= 2, grid.count())
    fr = [f for _, f in H.viewer_frames(pg)]
    ids = fr[0].evaluate("window.__regx.S.polys.map(p => p.label)") if fr else []
    H.check("RF", "reference footprint drawn on the candidate map", any("参照変換" in x for x in ids), ids[-4:])
    pg.reload(); H.wait_idle(pg, 3000)
    goto_map(pg, url, "031-8_6")
    H.check("RF", "reference designation persists after reload", "現在の参照" in H.main_text(pg) and "e2e-manual-test" in H.main_text(pg))
    pg.get_by_text("自動推定の候補と参照変換の比較").first.scroll_into_view_if_needed(); pg.wait_for_timeout(500)
    pg.screenshot(path=str(H.SHOTS / "RF1_031-8_6_reference_table.png"))


def test_lr(pg, url):
    H.open_wf(pg, url, "031-6_2")
    pg.get_by_role("tab", name="Viewer（重ね合わせ・手動調整）").click(); H.wait_idle(pg, 2500)
    el, fr = H.main_viewer(pg)
    el.scroll_into_view_if_needed()
    S = fr.evaluate("window.__regx.S.points.map(p => ({id: p.id, xy: p.xy.slice(0, 4000)}))")
    fx = next(p["xy"] for p in S if p["id"] == "fixed_nuclei"); mx = next(p["xy"] for p in S if p["id"] == "moving_nuclei")
    H.set_click_mode(fr, "landmark"); pg.wait_for_timeout(300)
    for i in (100, 400, 700):
        H.click_world(pg, el, fr, (fx[2 * i], fx[2 * i + 1])); H.click_world(pg, el, fr, (mx[2 * i], mx[2 * i + 1]))
    fr.locator("#side button", has_text="送信").click(); H.wait_idle(pg, 3500)
    t = H.main_text(pg)
    H.check("LR", "3 pairs listed with a type / confidence editor", "指定した対応点 3 組" in t and pg.locator('[data-testid="stDataFrame"]:visible').count() > 0)
    H.check("LR", "robust fit with leave-one-out error shown", "外れ値を除外" in t and "leave-one-out" in t, t[-500:])
    pg.locator('[data-testid="stTextInput"]:visible').filter(has=pg.locator('[data-testid="stWidgetLabel"]', has_text="対応点の名前")).first.locator("input").fill("e2e reference landmarks")
    pg.keyboard.press("Enter"); H.wait_idle(pg)
    pg.get_by_role("button", name="保存して参照変換に指定").click(); H.wait_idle(pg, 3000)
    H.check("LR", "saved as landmark transform and designated as reference", "参照変換に指定しました（tf-" in H.main_text(pg))
    H.open_wf(pg, url, "031-6_2")                      # next render of the viewer shows the newly designated reference
    el, fr = H.main_viewer(pg)
    H.check("LR", "reference footprint drawn in the viewer", any("参照変換" in x for x in fr.evaluate("window.__regx.S.polys.map(p => p.label)")))


def test_nr(pg, url):
    H.open_wf(pg, url, "031-8_6")
    pg.get_by_role("tab", name="新規実行").click(); H.wait_idle(pg, 2000)
    lab = pg.locator('[data-testid="stWidgetLabel"]', has_text="画像探索で比べる画像特徴")
    box = pg.locator('[data-testid="stSelectbox"]').filter(has=lab).first
    H.check("NR", "image feature selector present, default occ+cleft", box.count() > 0 and "組織の占有率と間隙の網目" in box.locator("input").input_value(), box.locator("input").input_value() if box.count() else None)


TESTS = {"FM": test_fm, "RF": test_rf, "LR": test_lr, "NR": test_nr}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8512"); ap.add_argument("--only", default="FM,RF,LR,NR")
    a = ap.parse_args()
    H.SHOTS.mkdir(exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        for k in a.only.split(","):
            pg = b.new_page(viewport={"width": 1600, "height": 1100})
            try:
                TESTS[k](pg, a.url)
                exc = H.page_errors(pg)
                H.check(k, "no Streamlit exception on page", not exc, exc[:1])
            except Exception:  # noqa: BLE001
                H.check(k, "test ran to completion", False, traceback.format_exc()[:300] + " ... " + traceback.format_exc()[-500:])
                pg.screenshot(path=str(H.SHOTS / f"{k}_error.png"))
            pg.close()
        b.close()
    (HERE / "ui_e2e_localization_results.json").write_text(json.dumps(H.RESULTS, indent=1, ensure_ascii=False))
    print(sum(r["pass"] for r in H.RESULTS), "/", len(H.RESULTS), "passed")
