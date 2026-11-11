"""Render ui_test_results.md from ui_e2e_results.json (+ the pytest summary lines passed on the command line).

  python make_ui_test_results.py "<unit tests summary>" "<full suite summary>" "<gpu parity summary>"
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = {"A": "Test A：031-6_2（E0x、UNIQUE、画像・点群、Before/After、Zoom/Pan、Warp Preview）",
         "B": "Test B：031-8_6（AMBIGUOUS、上位候補、切片サイズの違い、候補の切り替え、PREVIEW ONLY）",
         "C": "Test C：Point-only（画像なしのサンプル、画像機能の無効化、単位の扱い）",
         "D": "Test D：Fallback（画像読み込み失敗の再現、保存済みのPOINT_ONLY_FALLBACK）",
         "E": "Test E：Fixed/Movingの交換（両方向、逆Warp、座標の整合）",
         "F": "Test F：既存アプリとの互換性（Workflow A〜Dのページ）",
         "G": "Test G：新規実行（Slurm投入、重複防止、キャンセル）",
         "H": "Test H：UIから実行した結果の閲覧",
         "I": "Test I：手動調整（回転・拡大縮小・平行移動、表示の維持、Warp Preview、保存と再表示）（2026-10-22追加）"}
SHOTS = {"A": ["A1_031-6_2_E0x_after_overlay.png", "A2_031-6_2_E0x_zoom_checker.png", "A3_031-6_2_E0x_vectors.png", "A4_031-6_2_before.png", "A5_031-6_2_warp_preview.png"],
         "B": ["B1_031-8_6_E0x_top_candidate_fov.png", "B2_031-8_6_candidate1_preview.png", "B3_031-8_6_candidate_table.png", "B4_031-8_6_X3_candidates_with_icp.png", "B5_031-8_6_X3_candidate2_icp_preview.png"],
         "C": ["C1_point_only_pair_A2.png", "C2_031-6_2_images_hidden.png"], "D": ["D1_simulated_image_failure_alert.png", "D2_E2_point_only_fallback.png"],
         "E": ["E1_031-6_2_IF_fixed.png", "E2_swap_consistency.png", "E3_warp_HE_into_IF.png"], "F": ["F1_workflow_c_unchanged.png"],
         "G": ["G1_new_run_submitted.png", "G2_run_status_after_cancel.png"], "H": ["H1_explorer_run_A2_result.png"],
         "I": ["I1_031-8_6_manual_adjustment.png", "I2_031-8_6_saved_manual_record.png"]}


def main():
    res = json.loads((HERE / "ui_e2e_results.json").read_text())
    unit, full, gpu = (sys.argv[1:4] + ["(未記録)"] * 3)[:3]
    n_ok = sum(r["pass"] for r in res)
    out = ["# UIテスト結果（2026-10-20、2026-10-22更新）", "",
           f"- **ブラウザE2E**（headless Chromium + Playwright、`ui_e2e.py`、開発サーバー http://127.0.0.1:8511）：**{n_ok} / {len(res)} 通過**",
           f"- **Explorerの単体テスト**（`tests/test_registration_explorer.py`）：{unit}",
           f"- **既存テスト一式**（`pytest tests`、CPUノード）：{full}",
           f"- **GPU parityテスト**（GPUを割り当てて再実行）：{gpu}", "",
           "スクリーンショットは `ui_screenshots/` にあります。判定は、画面に表示された文字とviewerの状態だけで行っています。", ""]
    for k, title in NAMES.items():
        rows = [r for r in res if r["test"] == k]
        if not rows:
            continue
        out += [f"## {title}", "", "| 確認項目 | 結果 |", "|---|---|"]
        out += [f"| {r['check']} | {'PASS' if r['pass'] else 'FAIL: ' + r['detail'][:120].replace('|', '/')} |" for r in rows]
        out += ["", "スクリーンショット：" + "、".join(f"`ui_screenshots/{s}`" for s in SHOTS.get(k, []) if (HERE / 'ui_screenshots' / s).exists()), ""]
    (HERE / "ui_test_results.md").write_text("\n".join(out))
    print("wrote ui_test_results.md", n_ok, "/", len(res))


if __name__ == "__main__":
    main()
