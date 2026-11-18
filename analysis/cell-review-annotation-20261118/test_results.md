# テスト結果（2026-11-18）

Slurm（mib-dbia、GPU 1 枚）の `pytest.sbatch all`（job 939047）で実行した。ログは `logs/`（git に含めない）。既存のテストは、変更も削除もしていない。

| 対象 | 結果 |
|---|---|
| `tests/` | 270 passed, 1 skipped |
| 凍結済み 10-14 / 10-08 | 10 / 7 passed |
| 11-13 `tests_rcm` | 13 passed |
| 11-17 `tests_crx`（slow を含む） | 24 passed, 1 skipped（node なし） |
| 今回の `tests_cra`（slow を含む） | 15 passed, 1 skipped（node なし。ゲートウェイでは passed） |

- 1 回目（job 939043）は、`tests/` の GPU の一致テスト 2 件が失敗した。sbatch に `--gres=gpu:1` を入れ忘れたためで、GPU を付けた再実行（job 939044）と job 939047 では通った。
- 1 回目は、UI のテストも 1 件失敗した。変位場を計算しないと出力ボタンが出ないためで、表の CSV を出すボタンを追加し、テストの手順を直した。

## tests_cra（16 件）
| テスト | 確かめること（完了条件との対応） |
|---|---|
| `test_protected_files_unchanged` | Workflow C / D、engine、Graph / UOT / B2、Density Flow / Joint Flow、viewer / scene / imaging / catalog / page、サンプルの manifest の 41 ファイルが、作業前と同じ（**回帰なし**） |
| `test_contours_join_by_id_and_detect_problems`（CSV と GeoJSON） | ID での結合、輪郭なし・ID の重複・中心が輪郭の外・核に結べない輪郭の検出。座標の倍率が中心点と同じ。空間の検索と抽出（**ID と座標系の整合性**） |
| `test_contours_frame_mismatch_detected` | 別の解像度の輪郭を、座標系の不一致として検出する |
| `test_points_only_sample_has_no_contours` | 輪郭がないサンプルでは None になる（中心点でのレビュー） |
| `test_real_contours_match_the_centre_points`（6_2、8_6） | 実データで、すべての核の中心点が、同じ ID の輪郭の内側にある。Affine で写した輪郭も、写した中心点を含む（**輪郭が正しい位置に表示される**） |
| `test_review_states_are_separate_from_the_algorithm` | 対応確認・未特定・非観測（コメント必須、方向と範囲を記録）・判断不能＋タグ。unmatched は未レビューのまま。判定の付け直し・未レビューに戻す・取り消し・やり直し。保存と再読み込み（schema /2）。自動推定は変わらない（**4 種類の区別、元の推定の保持**） |
| `test_schema_v1_reviews_still_load` | 11-17 の形式（/1）のレビューを読める（**後方互換**） |
| `test_conflicts_are_detected_not_resolved` | 同じ IF 核への複数の対応を検出し、自動では消さない |
| `test_annotation_table_metrics_and_agreement` | 1 核 1 行の表（元の ID、アルゴリズムの状態、人の状態、対応先の順位、一致）。精度・再現率・非観測の率・2 つのレビューの一致と κ（**評価できる形式**） |
| `test_export_contains_annotations` | 出力に `review_annotations.csv` と `review_metrics.json` が入る |
| `test_picker_contours_level_of_detail_and_culling` | 全体ビューアーの JS（Node）：縮小時は輪郭を描かない（LOD）。画面外は描かない。ID は拡大時だけ。表示を切ると描かない |
| `test_compare_viewer_html` | 左右比較の HTML と JS の構文、同期用の逆変換 |
| `test_ui_contours_compare_and_review_states`（slow） | 実データ 031-6_2 の AppTest。全体ビューアーの輪郭（写した輪郭が中心を含む）、左右比較（未変形の切り出し、選択核と候補の輪郭が中心を含む、周辺の核）。候補の切り替え → 対応確認。非観測（コメントなしは拒否）、判断不能＋タグ、未特定、未レビューに戻す、取り消し。保存 → 別のセッションで再読み込み → 同じ判定。出力の表。自動推定の sha256 は変わらない（**完了条件 1〜6**） |
| `test_ui_points_only_sample_reviews_without_contours`（slow） | 点群だけのサンプルで、輪郭なしでレビューできる |

## 表示の確認（ブラウザの代わり）
`make_figures.py`（job 939046）で、`figures/fig_compare_031-6_2.png` と `figures/fig_compare_031-8_6.png` を作った。輪郭が画像上の核に重なることを目で確かめた。
