# テスト結果（2026-11-25、HEAD 10b0fba + 作業ツリーの変更）

## 1. 既存のテストと、この研究のテスト（Slurm job 1001375、`sbatch pytest.sbatch all`）

| スイート | 結果 |
|---|---|
| tests/（既存） | 270 passed, 1 skipped |
| graph-matching-adoption-separation-20261014（凍結） | 10 passed |
| contextual-graph-uot-20261008（凍結） | 7 passed |
| region-cell-matching-20261113 tests_rcm | 13 passed |
| cell-review-explorer-20261117 tests_crx | 24 passed, 1 skipped（node の無い計算ノード） |
| cell-review-annotation-20261118 tests_cra | 15 passed, 1 skipped（同上） |
| whole-cell-segmentation-20261118 tests_wcs | 11 passed, 1 skipped（同上） |
| **この研究 tests_cro** | 4 passed, 5 skipped（同上） |

計算ノードでスキップした JS のテストは、ログインノード（node v12）で実行しました。対象は次の 14 件で、すべて合格しました。
- tests_cro の slow 以外の 8 件（JS 5 件を含む）
- tests_crx のクリック選択（旧 API）
- tests_wcs の輪郭の切り替え
- tests_cra で名前が一致したもの

## 2. 要求されたテスト項目と、対応するテスト

E2E は実際のブラウザ（Chromium）、JS は node の単体テスト、AppTest は Streamlit のテストです。

| 項目 | 実際のブラウザ（e2e_results.csv） | JS 単体（node） / AppTest |
|---|---|---|
| 左クリックで正しい核 ID が選ばれる | 「left click selects…」、「judgement panel shows the clicked nucleus」 | `test_click_selects_nearest_he_without_mode_and_focuses`、AppTest |
| 選んだ核へ自動で拡大する | 「selected nucleus centred and zoomed to ~60 um」、「user zoom kept…」 | 同じ JS のテスト |
| ドラッグで移動でき、核を誤って選ばない | 「drag pans…」、「drag release selects nothing」、「drag that starts on another nucleus…」 | `test_drag_pans_and_never_selects` |
| ホイールでズームした後も選べる | 「wheel zooms…」、「click after wheel zoom selects the right nucleus」 | — |
| 高解像度の画面でも座標が一致する | 全項目を dpr 2 で実行し、合格。「canvas uses devicePixelRatio」 | 許容範囲を dpr 1・2 で確認（JS） |
| 左右比較が選んだ核に追従する | 「left/right comparison follows…」、「comparison right pane centred on the chosen candidate」、「comparison zoom kept…」、「another candidate keeps the zoom…」 | `test_compare_viewer_view_state`、AppTest（右の枠の anchor が候補） |
| IF 候補の切り替えが正しく動く | 「click on a candidate ring…」、「judgement panel radio follows…」 | `test_candidate_ring_and_shift_click`、AppTest |
| キーの判定が正しい核に記録される | 「key 2 records 未特定…」、「key 4 from the page…」、「saved review: verdicts on the intended nuclei」 | `test_keyboard_shortcuts_and_typing`、AppTest（stale な核には記録しない） |
| 入力欄での文字入力を妨げない | 「typing digits / Enter in the comment box…」、「typing in another tab's text box…」 | JS（text / number / textarea / contentEditable では無視、checkbox では有効） |
| 取り消し・やり直し・自動で次へ | 「Ctrl+Z…」、「Ctrl+Y…」、「after a recorded verdict the next unreviewed…」、「Enter moves…」、「Enter on a focused verdict button…」 | AppTest（非観測が拒否されたときは次へ進まない） |
| 核輪郭・細胞輪郭が表示されたまま | 「nucleus contours (and cell outlines) still drawn」（cells の条件で細胞輪郭も確認） | `test_selection_from_python_focuses_only_on_change_and_toggles_persist` |
| 既存のレビュー形式と計算結果が変わらない | 「saved review…」（schema /2） | `test_protected_files_unchanged`（65 ファイル）、AppTest（形式、自動推定の sha）、既存のスイートすべて |
| 非観測にはコメントが必要（既存の規則） | 「key 3 without a comment saves nothing and focuses the comment box」 | AppTest |
| 別のタブでは効かない | 「shortcuts do nothing while another tab is shown」 | — |
| 画像の表示モードが変わらない | 「display mode (RGB / 核強調) unchanged」 | — |

## 3. 実際のブラウザでの確認の条件

- **環境**：Slurm の計算ノード mib-dbia-g02 で、Streamlit 1.62 と Chromium 140 headless shell（Playwright 1.55）を使いました。
- **画面**：ビューポートは 1680×1000、devicePixelRatio は 1 と 2 です。
- **データ**：実データ 031-6_2 で、ROI は既定の ROI の中心 ±110 µm（11-18 のテストと同じ）です。自動推定は、変更していないコードで作りました。
- **実行**：4 つの条件（dpr1 / dpr2 / cells / tabs）を、それぞれ新しいサーバーと新しいレビューの保存先で実行しました。

途中で見つかった問題と修正:
1. sticky を判定パネルに付けていましたが、効いていませんでした。親のラッパーが同じ高さで、動ける余地が無かったためです。sticky を親のラッパー（`stLayoutWrapper`）に付け直しました。
2. 確認スクリプト側の問題を直しました。
   - `body` のクリックを blur に変えました。
   - ボタンの DOM 要素が 2 つずつあるので、`.first` を使うようにしました。
   - 表示モードを、ラベルではなく値で比べるようにしました。

## 4. 未確認

- 対象外のブラウザ・入力機器：Firefox、Safari、Windows/macOS のブラウザ、タッチパッド、タッチ操作。
- Explorer の全体のページ（`page.py`）での、実際のブラウザでの操作。
- 左右比較の中で押したキーの、実際のブラウザでの転送。
- 人による操作感の評価。
