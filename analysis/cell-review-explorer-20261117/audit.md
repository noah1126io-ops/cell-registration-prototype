# 既存機能の監査（コード変更前、2026-11-17）

作業前の HEAD は `e8a68f6`（feature/gpu-slurm）で、未コミットの変更はない。監査した各ファイルの sha256 は `baseline_manifest.json` に記録した。

## 1. 監査の結果：仕様の Phase ごとに見た、実装済みの処理と不足している処理

| Phase | 実装済み（再利用するもの） | 不足していたもの（今回追加） |
|---|---|---|
| 1 対応領域 | engine C1（画像／片側画像／点群の探索、`registration_core.workflow`）。Explorer「対応領域の探索」タブ（候補の一覧と地図）。「Viewer」タブの手動の回転・倍率・平行移動（`manual.compose`）、ランドマーク（`reference.fit_reference`）、ROI 指定。Viewer は表示中の変換を `st.session_state["regx-ctx"]` に書く | 選んだ変換を核の対応計算に渡す経路（新タブで、Viewer の変換・結果・候補から選べるようにした） |
| 2 核対応 | 凍結済みの B0 / B1 / B3 ＋ B2（`registration_research.cell_correspondence.run_pipeline`、10-14 の実装を import）。B1 の UOT 輸送計画（`UOTRun.plan`）、`ranked_indices`、graph support | 核ごとの上位 5 候補（順位・輸送量の割合・graph support・コスト）の表。元の核 ID の保持。採用の根拠の文。衝突の検出。ROI 単位の計算。`run_pipeline` が `runs` を返さなかった（返り値にキーを追加。既存のキーは不変） |
| 3 手動レビュー | なし（既存の「評価・注釈」は、変換や候補に対する評価で、核単位の編集はない） | 支持・除外・変更・追加・未確定・取り消し・やり直し。履歴（元の推定、変更内容、使用者、日時、理由）。自動推定とは別のファイルに版ごとに保存 |
| 4 局所変形 | linear RBF（`validation_lib.fit_linear_rbf`、`deformation_experiment._rbf_from_anchors`）、UOT conservative（`validation_lib.run_uot`）、安全性の監査（`common_audit`、SAFETY の閾値）、アンカー不足のときに Affine に戻す処理（`MIN_ANCHORS`=20） | 任意の対応ペアからアンカーを作る処理（`anchors_from_pairs`）。アンカー分布のゲート（被覆率 0.20 未満なら Affine のみ。新しく加えたゲートで、判定を厳しくする方向にしか働かない）。レビューを変えたときの再計算 |
| 5 画像変形 | Affine の逆写像（`imaging.warp_preview`）。本番の H&E の逆写像（`pointset_registration.warp_he_image_to_world`、非剛体の逆は `_inverse_fine_warp` の固定点 8 回）。ピラミッドの ROI 読み込み（`imaging.read_region`） | **非剛体の逆写像の収束チェックがなかった**（本番は 8 回で打ち切る）。研究用に、残差を確かめる逆写像を追加した（`invert_field`。収束しなければ `INVERSE_NOT_CONVERGED` とし、適用済みと表示しない）。ROI のタイルだけを読む非剛体の再標本化 |
| 6 保存・比較 | 研究ストア（SQLite）。今回は書き込まない | ファイルでの出力（CSV、変位場の npz、画像の PNG、安全性、設定、履歴、修正による差分） |

## 2. 座標系と変換の方向
- 方向は **H&E（moving）→ IF（fixed）** に固定した。Workflow D の `he_to_if` と同じ。
- Viewer の変換が IF → H&E の向きで表示されている場合は、逆変換して使う（`to_he_to_if`、`results.oriented`）。
- 順写像（点）：`x_IF = A(x_HE) + d(A(x_HE))`。`d` は IF frame の格子（4 µm）上の変位場で、bilinear で標本化する。格子の外では、境界の値を 50 µm かけて直線的に 0 まで減らす（`TAPER_UM`。境界で写像が不連続になると、画像の逆写像が振動して収束しないため。ROI の核はすべて格子の内側にある）。
- 逆写像（画像）：IF の画素中心 `p` について、まず `q + d(q) = p` を固定点反復で解き、残差を確かめる。次に `x_HE = A⁻¹(q)` を求め、H&E のピラミッドの画素で標本化する。
  - 画素の規約は既存と同じ。level 0 の画素中心は整数の位置にあり、world = index × pixel_um。
- 点と画像は、同じ `forward_displacement` を使う（テストで一致を確認した）。

## 3. 変更しないもの
- Production Workflow C / D
- engine（`registration_core`）、既存の C3
- 凍結済みの Graph / UOT / B2 と UOT conservative
- Density Flow / Joint Flow
- 既存の viewer.py / imaging.py / scene.py
- 過去の analysis ディレクトリ

既存のファイルで変えたのは次の 2 つだけ。
- `page.py`：タブの差し込み口。既定で無効
- `cell_correspondence.py`：返り値に `runs` を追加
