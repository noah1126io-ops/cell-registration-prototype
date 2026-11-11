"""Render ui_test_results.md from ui_e2e_results.json (+ pytest summaries on the command line)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = {"A": "031-6_2：既存結果の表示（Before/After・重ね合わせ・Warp）", "B": "031-8_6：AMBIGUOUS・候補（PREVIEW ONLY）",
         "C": "Point-onlyサンプル・画像を隠した表示", "D": "画像読み込み失敗とfallback", "E": "Fixed/Movingの交換・逆Warp",
         "F": "既存Workflow A〜Dのページ", "H": "旧Runnerの結果の表示", "I": "手動調整（回転・倍率・平行移動）とストアへの保存",
         "J": "研究Workflow C：段階の診断（C1/C2/C3）と画像探索・核探索の比較", "K": "候補マップと、クリックによる候補の選択",
         "L": "対応点のクリックと座標の定量確認・注釈として保存", "M": "ROIのドラッグと保存", "N": "構造化された評価の保存",
         "N2": "ブラウザ再読込・アプリ再起動の後の保持", "O": "実験記録とJSON / CSVのエクスポート", "P": "注釈を使った探索の設定（明示的な選択）",
         "Q": "Workflow Dへの受け渡し（採択済みのみ）", "R": "Workflow C（Production画面）の研究モードの接続", "S": "表示座標の定量確認（画像と核）"}
res = json.loads((HERE / "ui_e2e_results.json").read_text())
unit, full = (sys.argv[1:3] + ["(未記録)"] * 2)[:2]
ok = sum(r["pass"] for r in res)
out = ["# UIテスト結果（2026-10-22）", "",
       f"- **ブラウザE2E**（headless Chromium + Playwright、`ui_e2e.py`。テスト専用サーバー http://127.0.0.1:8512 と、専用の研究ストア `e2e_state/research_e2e.sqlite` を使用）：**{ok} / {len(res)} 通過**",
       f"- **新しい単体テスト**（`tests/test_registration_core.py`、`tests/test_research_store.py`、`tests/test_registration_explorer.py`）：{unit}",
       f"- **テスト一式**（`pytest tests`、`--gres=gpu:1`）：{full}", "",
       "- 手順：A〜N を実行 → テストサーバーを再起動（アプリ再起動）→ N2〜S を実行。N2で、再起動前に保存した評価・ROIが残っていることを確認しています。",
       "- 判定は、画面に表示された文字、viewerの状態（`window.__regx`）、ダウンロードしたファイルだけで行っています。",
       "- 座標の確認（L・S）は、提供者のGeoJSONから直接読んだ核の座標と、artifactの変換行列を使い、アプリのコードとは独立に計算しています。", ""]
for k, title in NAMES.items():
    rows = [r for r in res if r["test"] == k]
    if not rows:
        continue
    out += [f"## Test {k}：{title}", "", "| 確認項目 | 結果 |", "|---|---|"]
    out += [f"| {r['check']} | {'PASS' if r['pass'] else 'FAIL: ' + r['detail'][:150].replace('|', '/')} |" for r in rows]
    shots = sorted(p.name for p in (HERE / "screenshots").glob(f"{k}[0-9]_*.png"))
    if shots:
        out += ["", "スクリーンショット：" + "、".join(f"`screenshots/{s}`" for s in shots)]
    out.append("")
(HERE / "ui_test_results.md").write_text("\n".join(out))
print("wrote", ok, "/", len(res))
