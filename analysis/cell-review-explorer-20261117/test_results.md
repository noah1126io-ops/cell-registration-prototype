# テスト結果（2026-11-17）

テストは Slurm（mib-dbia、GPU 1 枚、`pytest.sbatch all`、job 932554）で実行した。ログは `logs/`（git に含めない）。

既存のテストは変更も削除もしていない。

| 対象 | 作業前（11-13 の記録。同じコードの e8a68f6） | 作業後（job 932554） |
|---|---|---|
| 既存のテスト `tests/` | 270 passed, 1 skipped | **270 passed, 1 skipped** |
| 凍結済み 10-14 `graph-matching-adoption-separation-20261014/tests` | 10 passed | **10 passed** |
| 凍結済み 10-08 `contextual-graph-uot-20261008/tests` | 7 passed | **7 passed** |
| 11-13 `region-cell-matching-20261113/tests_rcm` | 13 passed | **13 passed** |
| 今回の `tests_crx`（slow を含む） | — | **23 passed, 1 skipped** |

- skip の 1 件（`test_viewer_js_syntax_and_nucleus_pick`）は、計算ノードに `node` がないため。ゲートウェイでは passed。
- 作業前の時点で、`src/` と `tests/` は 11-13 のコミット `c6e0b39` と同じ（`e8a68f6` で追加したのはケースの一覧だけ）。そのため、作業前の値には 11-13 の記録を使った。

## tests_crx（24 件）

| テスト | 確かめること（仕様 Phase 7 との対応） |
|---|---|
| `test_existing_files_unchanged_except_allowed` | 本番 Workflow C / D、engine、凍結済みの研究コード、viewer / imaging / scene、Density Flow / Joint Flow の 40 ファイルが作業前と同じ。違うのは `page.py`（差し込み口）と `cell_correspondence.py`（`runs` を返すように追加）だけ（**既存の Workflow C / D の出力が変わらない**） |
| `test_tab_off_by_default`、`test_page_hook_keeps_existing_tabs`、`test_full_explorer_page_with_and_without_tab`（slow ×2） | 新しいタブは既定で無効。有効でも無効でも、既存の 8 つのタブの並びと動作は変わらない |
| `test_run_pipeline_frames_unchanged_by_runs_key` | 返り値に `runs` を追加しても、B0 / B1 / B3 / B2 の結果は変わらない |
| `test_ids_preserved` | **核 ID が正しく保持される**（インデックスとは違う文字列や数値の ID を使って確かめた）。ambiguous / unmatched の核にも上位 5 候補がある |
| `test_auto_quality_on_synthetic`、`test_truth_not_an_input` | 合成データで、B2 の matched の 9 割以上が正しい。正解は入力にならない |
| `test_auto_written_once_and_never_overwritten` | **自動推定の結果は手動修正で上書きされない**。2 回目の保存は拒否される。レビューは版ごとに別ファイルになる。改ざんは sha256 で検出する |
| `test_review_operations_undo_redo_history` | 支持・除外・変更・追加・未確定・取り消し・やり直し。履歴（元の推定、使用者、日時、理由）。JSON から同じ状態を再現できる |
| `test_invalid_ids_and_conflicts` | **不正な ID や重複した対応を検出する**。衝突した対応は、指定すればアンカーから除ける |
| `test_fields_applied_and_improve_on_synthetic` | 編集していないレビューの変位場は、自動推定のものと同じ。アンカーにしなかった核で、Affine より真の位置に近づく |
| `test_edits_change_the_field` | **変更・追加・除外が変位場に反映される**（アンカーの数、変位場、点の移動量） |
| `test_too_few_or_badly_spread_anchors_fall_back_to_affine` | アンカーが 20 組未満なら Affine に戻る。アンカーが 1 つのセルに偏っていても Affine に戻る |
| `test_dangerous_edits_are_not_applied` | 組織の半分の対応を 25 µm ずらしても、既存の安全性の監査が棄却する。画像も `NOT_APPLIED` になる（**危険な変形は正式には適用されない**） |
| `test_image_follows_the_same_transform_as_the_nuclei` | 核の位置にぼかした点を描いた画像を、Affine・人工の変位場・B2＋RBF の変位場で変形した。点の重心と、核の順写像の位置の差は、中央値 0.25 px 未満、95% が 0.6 px 未満（0.5 µm/px）（**同じ変換をかけた核の座標と、画像上の核の位置が一致する**）。読み込んだのは ROI の領域だけ |
| `test_inverse_not_converged_is_not_reported_as_applied` | 折り返しのある変位場では `INVERSE_NOT_CONVERGED` となり、適用済みとは表示しない |
| `test_export_bundle` | 出力される 11 種類のファイル。Workflow D には送っていない |
| `test_ui_flow`（slow ×4） | AppTest。**両側の画像**（031-6_2）、**片側の画像**（H&E だけ）、**点群のみ**（画像を隠す）、点群だけのサンプル（serial_section_pair）で、計算 → 核の選択 → 除外 → 取り消し・やり直し → 変位場 → 画像 → 出力までの操作が例外なく動く |
| `test_viewer_js_syntax_and_nucleus_pick` | ビューアーで核を選ぶ JS の動作（Node で実行）。クリックした位置の最寄りの核の、全体でのインデックスを返す。表示モードでは選ばない |

## 実データでの通しの実行（`smoke_real.py`、Slurm job 932546 / 932547）
- 結果は `smoke_031-6_2.json`、`smoke_031-8_6.json`。
- 図は `figures/fig1_*.png`（`make_figures.py`、job 932548）。

## ブラウザでの確認について
- AppTest は Streamlit の Python 側だけを動かすので、JS の部品の中のクリックは操作できない。
- そこで、部品の JS は Node で単体テストした。
- ヘッドレスブラウザ（Playwright）の導入は、このセッションの権限で拒否されたため、実際のブラウザでのクリックは自動では確認していない。
- 開発サーバー（8511、`CELLREG_CELL_REVIEW=1`）を起動してあるので、ユーザーの環境で確認できる。

## 追記（2026-11-18）：表示モードの共有
- `test_display_modes_shared_and_colours_identical`（slow）を追加した。tests_crx は全部で 25 件。
- 確かめること：両タブの既定は RGB。Viewer で変えるとレビューに、レビューで変えると Viewer に反映される。同じ元画像と同じモードなら、IF・H&E とも画像レイヤーの JPEG のバイト列（＝色）と配置行列が一致する。
- 対象を絞った実行（job 939033）：8 passed（新しいテスト、UI フロー 4 件、ページ全体 2 件、既存ファイル不変）。全体の結果は下に記す。
- 全体（`pytest.sbatch all`、job 939035）：
  - `tests/` 270 passed, 1 skipped
  - 凍結済みの研究テスト 10 ＋ 7 passed
  - 11-13 のテスト 13 passed
  - tests_crx 24 passed, 1 skipped（skip は node がないためで、前回と同じ）
