# テスト結果（2026-11-18、HEAD 1a66073 + 作業ツリーの変更）

## 実行

Slurm（mib-dbia、`--gres=gpu:1`）で次を実行しました。

- `sbatch pytest.sbatch all`（job 939066）
- `sbatch pytest.sbatch new`（job 939068）

| スイート | 結果 |
|---|---|
| tests/（既存） | 270 passed, 1 skipped |
| graph-matching-adoption-separation-20261014（凍結） | 10 passed |
| contextual-graph-uot-20261008（凍結） | 7 passed |
| region-cell-matching-20261113 tests_rcm | 13 passed |
| cell-review-explorer-20261117 tests_crx | 24 passed, 1 skipped（node が無い計算ノード） |
| cell-review-annotation-20261118 tests_cra | 15 passed, 1 skipped（同上） |
| **この研究 tests_wcs** | **11 passed, 1 skipped**（同上）。skip した JS のテストはログインノード（node あり）で **2 passed** |

- 939066 で UI のテストが 1 件失敗しました。
- 原因はテストの前提で、テスト用の ROI で 031-8_6（隣り合う切片）に自動の対応が 0 組でした。
- 1 位候補を使うようにテストを直し、939068 で合格しました。

## 要求された 12 項目と対応するテスト（tests_wcs/test_whole_cell.py）

| # | 項目 | テスト |
|---|---|---|
| 1 | 既存の核輪郭が変わらない | `test_protected_files_unchanged`（nuclear_contours.py を含む 42 ファイルの sha256） |
| 2 | 細胞輪郭の読み込みと座標 | `test_cell_contours_load_coordinates_and_query`、`test_real_full_section_cells_relate_to_existing_nucleus_ids`（実データ: 「細胞に属す」核の中心が、その細胞の輪郭の内側にある割合 > 95%） |
| 3 | ID の追跡 | `test_nucleus_ids_tracked_not_renumbered`（元の ID で結び直す。順序が変わったら検出する。消えた ID は推測しない）、同じく実データ |
| 4 | 複数核・核なし・細胞なし | `test_association_cases`（複数核、核の無い細胞、細胞の無い核、中心が細胞外、2 つの細胞の間で競合） |
| 5 | 中心点だけのサンプル | `test_points_only_sample_and_unavailable_methods`、`test_ui_review_works_without_cell_outlines` |
| 6 | 左右の位置 | `test_ui_cell_outlines_compare_switch_and_save`（右の枠の細胞輪郭が、比較中の IF 核の中心を含む。IF frame）、`test_compare_viewer_cell_items` |
| 7 | 方式の切替 | 同じ UI テスト（if_mem_full → voronoi → mesmer（理由を表示、輪郭は無し）→ if_mem_full）、`test_picker_cell_groups_toggle_independently` |
| 8 | 方式と版をレビューに保存 | `test_review_cell_record_saved_and_old_formats_load`、UI テスト（保存したファイルが /3 で、method・version・cell_ids・reliability を持つ） |
| 9 | 古いレビュー形式の読み込み | 同じ（/1 と /2 を読める。cell の列は空欄） |
| 10 | 注釈・評価の表 | 同じ（`cell_*` の列） |
| 11 | Graph/UOT/B2・変形場・Workflow C/D が変わらない | `test_protected_files_unchanged`、`test_auto_fields_and_warp_identical_to_head`（変更前の HEAD の cell_review.py と、自動推定・変形場・H&E の変形が数値で一致） |
| 12 | 実際のブラウザで確認できないもの | **未確認**: 実際のブラウザでの破線の見え方・色・操作感。AppTest は表示データだけを、node のハーネスは描画の呼び出しだけを確認した |

## そのほか

- dev server（127.0.0.1:8511、`CELLREG_RESEARCH_MODE=1 CELLREG_CELL_REVIEW=1`）を再起動し、応答を確認しました。
- 実際のブラウザでの操作は、まだ確認していません。
