# 実装前の調査（2026-11-18）

作業前の HEAD は `59c0b07`（feature/gpu-slurm）。変更してはいけないファイルの sha256 は `baseline_manifest.json` に記録した。

## 1. Cell Review / Explorer の既存の UI 構造
- タブ「細胞対応レビュー（研究）」（`cell_review_ui.py`）。既定では無効で、`CELLREG_CELL_REVIEW=1` で有効になる。研究モードは `CELLREG_RESEARCH_MODE=1`。
- タブは次の 4 つのセクションからなる。
  1. Global 変換と ROI を選び、自動推定を計算して保存する。
  2. ビューアーで核を選び、レビューする。
  3. 変位場。
  4. 画像の変形と出力。
- ビューアー（`cell_review_viewer.py`）は、`viewer._TEMPLATE` に独自のクリック API を載せたもの。操作はズーム・パン・層ごとの表示切り替え・透明度・「H&E 核を選択」「IF 核を選択」。表示は IF frame で、H&E は Global 変換後の位置に置く。
- 既存のレビュー操作は 5 つ：支持（support）・除外（exclude）・候補または任意の IF 核への変更（reassign）・追加（add）・未確定（unset）。ほかに、取り消し・やり直し・版ごとの保存・保存済みレビューの再読み込み・同じ IF 核への複数対応（衝突）の検出がある。

## 2. 核のセグメンテーションデータの保存形式と読み込み処理
| サンプル | 切片 | 中心点（`catalog.load_nuclei` が読むもの） | 輪郭 | 対応付けのキー |
|---|---|---|---|---|
| 031-6_2 / 031-8_6 | H&E | StarDist の `he_nuclei.csv`（`centroid_x_px`、`centroid_y_px`、level-0 px）。ID は行の番号 | 同じディレクトリの `he_nuclei.geojson`（Polygon、32 頂点）。`point_id` を持つ | `point_id` = CSV の行の番号（= `point_id` 列） |
| 031-6_2 / 031-8_6 | IF | 提供元の `*_seg_seqif_level0.geojson` の `centroid_x/y`（level-0 px）。ID は `segment_id` | 同じファイルの Polygon | `segment_id` |
| serial_section_pair | 両方 | npy の点だけ | なし | — |

- 既存のコードは、輪郭を一度も読んでいない（中心点だけ）。
- 実データで確認した結果（4 つのファイルすべて）：
  - ID の対応付けは完全。
  - すべての中心点が、自分の輪郭の内側にある。
  - 中心点と、多角形の面積の重心との差は、等価な半径の最大 0.63 倍（StarDist の中心点は、面積の重心ではないため）。
  - 穴のある多角形はない。
- 座標：輪郭も中心点も、level-0 の画素の位置。world は px × `pixel_um`（中心点と同じ規約）。

## 3. セグメンテーション ID・重心座標・対応推定 ID の関係
- 自動推定（`AutoResult`）とレビューは、核を **全体でのインデックス**（`load_nuclei` が返す配列の順番）で持つ。元の ID（`he_id` / `if_id`）は表に保存している。
- 輪郭は、元の ID で中心点と結ぶ。結べなかったものは「輪郭なし」とし、ID の重複は検出して報告する。

## 4. 画像座標と核座標の変換
- 方向は H&E（moving）→ IF（fixed）で、`x_IF = M x_HE + t`（`auto.meta["global"]`）。
- 画像の切り出しには `scene.roi_image` を使う。画素の規約は `imaging.level_corner_um` で、ピラミッドのタイルだけを読む。
- 比較用の画面では、**変形していない元画像**を、それぞれの切片の frame（H&E µm / IF µm）のまま表示する。位置の同期にだけ Global Affine を使う。

## 5. 対応候補・confidence・状態の保存方式
- `auto_correspondence.csv` に、核ごとに次の値を保存している：`status`（matched / ambiguous / unmatched）、`b2_match_prob`、`b2_unmatched_prob`、top1。
- `auto_candidates.csv` に、上位 5 候補（`rank`、`mass_share`、`graph_support`、`position_graph_cost`、`distance_um`）を保存している。
- これらは一度だけ書き込まれ、sha256 で改ざんを検出する。今回も変更しない。

## 6. 既存の手動レビュー・注釈の保存機能
- `runs/cell_review/<sample>/reviews/<review_id>/vNNNN.json`（schema `cellreg-cell-review-edits/1`）。
- 中身は、操作の列・取り消しの位置・追記だけの履歴（使用者・日時・理由・元の推定）。
- **足りないもの**：
  - 未特定・非観測・判断不能の区別がない（除外と未確定は、どちらも単に「対応なし」）。
  - 補助タグがない。
  - 非観測の対象（切片・方向・観察範囲）を記録していない。
  - 使ったセグメンテーションの識別子を記録していない。
  - 評価用の 1 核 1 行の表がない。
- Explorer の「評価・注釈」タブの注釈（SQLite）は変換・候補に対するもので、核単位ではない。今回は使わない（research.sqlite には書き込まない）。

## 7. 既存のテストと研究用機能の有効化
- テスト：`tests/`、凍結済み 10-14 / 10-08、11-13 `tests_rcm`、11-17 `tests_crx`。
- 11-17 の `tests_crx` は、レビューの既存の操作・ボタンのキー（`-sup`、`-exc`、`-uns`、`-candset`、`-ifset`、`-undo`、`-redo`、`-save`）・出力ファイルを前提にしている。互換性を保つため、これらは残す。

## 8. 実装上の制約（この調査から）
- 輪郭は、既存の GeoJSON から読む（新しいセグメンテーションは行わない）。輪郭のないサンプル（点群だけ）では、中心点だけでレビューする。
- レビューは H&E 核を起点とする（H&E → IF）。
  - 非観測は「IF 切片で、観察した範囲に対応する核が見えない」という判断として記録する。
  - IF 核を起点とする判定（H&E に核がない）は今回は対象外。データの形式には、方向の欄を用意する。
- 一つの H&E 核の対応先は、データ構造上 1 つ。同じ IF 核に複数の H&E 核が対応する場合は、衝突として検出し、自動では解決しない。
- 変位場と画像の変形（3・4 章）は変えない。新しい判定（未特定・非観測・判断不能）の核は、対応先を持たないので、REVIEWED のアンカーにならない。これは既存の「除外」と同じ扱い。
