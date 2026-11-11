# C1（対応領域の探索）実装監査 — 2026-10-27

対象：`src/registration_core`（共通エンジン。研究Runner・Registration Explorer・Workflow C研究モードの3経路が同じコードを呼ぶ）。
変更前のコードは `baseline/features.py.before` と `baseline/workflow.py.before` に保存しています。

座標の約束：`fixed_world = M @ moving_world + t`。world は各画像の level-0 画素 × pixel_um（µm）で、原点は各画像の左上（frame-local）です。2つの画像に共通する物理原点はUNKNOWNとして扱います。

| # | 処理 | 実装（ファイル：関数） | 入力 → 出力 | 座標系 | 特徴量・制約 | 想定される失敗条件 |
|---|---|---|---|---|---|---|
| 1 | H&E読み込み | `features.py: read_level`, `extract` | OME-TIFFのピラミッド → RGB配列 | 画素（level k）。`src_um = pixel_um × downsample` | 2 µm格子に対して1.3倍以上オーバーサンプルするlevelを選ぶ（031-8_6：level 2、0.755 µm/px） | pixel_um・ピラミッドがない場合は画像を使わない（`image_status`） |
| 2 | 蛍光読み込み | 同上 | 表示用RGB（seqIFを合成したもの） | 同上（level 3、0.80 µm/px） | チャンネルの意味は R = Desmoplakin、G = 膜系、B = Hoechst | 原チャンネルではなく表示用RGB。合成の仕方に依存する |
| 3 | 蛍光チャンネルの選択 | `extract` | — | — | **変更前**：`tissue = max(R, G, B)`、`nuclear = B − R`。**追加**：`occ = max(G, R)`（Bを除外） | **変更前の `tissue` はB（核）が支配的**で、組織構造ではなく核のテクスチャを表していた（`image_feature_audit.md`） |
| 4 | 前処理 | `extract` | — | 2 µm格子（`world = (col,row) × 2 µm`） | HE：光学密度の合計、ヘマトキシリン（`rgb2hed`）。σ 6 µm（tissue）、σ 2 µm（nuclear）で平滑化。パーセンタイル（1〜99.5%）で正規化 | Otsuの `tissue_bin` は、IFでは「核・膜」と「細胞質」を分けてしまい、組織と間隙を分けない（031-8_6 IFの組織率が0.40と出た） |
| 5 | 画像特徴量の生成 | `extract` → `FeatureMaps.maps` | `tissue, nuclear, edt, tissue_bin, envelope`（+ `occ, cleft`） | 2 µm格子 | 探索で使うのは `WorkflowConfig.image_feature`（変更前：`tissue+nuclear`）だけ。`edt` は使っていない | — |
| 6 | 解像度の調整 | `features.coarsen`, `pipeline.build_layers` | 2 µm → 4 µm（精密化）→ キャンバス上で 8 µm（粗探索） | 各frameの格子 | 両画像とも物理単位（µm）で揃える。画素サイズの違い（0.189 / 0.100 µm）は格子化で吸収する | pixel_umの誤りはそのまま倍率の誤りになる（今回は不具合なし。`coordinate_system_audit.md`） |
| 7 | 組織マスク・核密度マップ | `search.Canvas.hull_mask / raster_points`, `pipeline._fov_mask` | 核点 → σ 8 µmのガウス密度。画像 → 撮像範囲の全体をマスクにする | キャンバス（Fixed中心、一辺 = 2つの視野の対角の大きい方） | 核探索のマスクは点群の凸包 | — |
| 8 | 回転・倍率・平行移動の探索 | `search.PoseSearch.run` | 回転 0〜360°（4°刻み）× 倍率 {0.95, 1.00, 1.05} × chirality（proper）。平行移動はmasked NCC（FFT）で全範囲を調べる | Fixedキャンバス | 精密化：上位6候補を ±4°（1°刻み）× 倍率 ±3%、平行移動 ±40 µm の範囲で再探索。**倍率の範囲は約0.92〜1.08** | 倍率が範囲外のときはピークがぼやける。ただし今回の参照（手動で1.10）は過大で、画像の最適値は1.00〜1.05だった（Case Aは否定） |
| 9 | 候補領域の生成 | `PoseSearch._peaks` | 姿勢ごとにNCCのzスコア地図 → 局所最大（窓 16 µm）の上位3つ | — | 最小重なり：小さい方の視野の50% | — |
| 10 | スコアの計算 | `PoseSearch._maps` | 層ごとにmasked NCC → ロバストz（中央値 / MAD） | — | 層は `img_<feature>` か `points`。生の輝度は比べない | **構造が似た別の領域・180°回転した配置**が高得点になる（031-8_6） |
| 11 | 候補の順位付け | `_combine`（`zsum`） | 層のzの合計 | — | — | 情報のない層（隣接切片の核）がノイズとして加算される |
| 12 | 非最大抑制・Top-k | `PoseSearch.distinct`（プローブ点の中央値 < 40 µm なら同じ候補）、`n_fine = 6` | 粗探索の候補 → 異なる上位6つ → 精密化 → 判定 | — | 判定：margin < 0.10 なら AMBIGUOUS、z < 6 × 層数 なら LOW_PEAK | 正しい候補が粗探索で上位6位に入らないと、二度と評価されない（変更前の031-8_6：参照領域は261位） |
| 13 | C2への受け渡し | `workflow.refine_and_validate` → `cropped_icp` | 探索pose（M_search, t）→ 重なり内の核でICP（相似 → アフィン） | Fixed frame | 画像探索の候補でICPが信頼範囲を出たら、探索poseを維持する | 隣接切片では核が対応しないため、ICPに情報がない |
| 14 | Web Viewerの座標変換 | `registration_explorer.results.oriented`, `scene.apply` | 保存した変換を表示方向に合わせる（逆方向は逆行列） | 表示側のFixed frame | — | 往復誤差 3e-13 µm（不具合なし） |

## 「画像を使った」ことは「組織構造を比べた」ことを意味しない

変更前の画像探索（`tissue+nuclear`）で、031-8_6のIF側が比べていたのは次の2つです。

- `tissue`：max(R, G, B) をσ 6 µmで平滑化したもの。Hoechstの核が明るい粒として支配的です（`figures/feature_map_comparison_031-8_6.png`）。
- `nuclear`：B − R、つまり核です。

どちらも実質的に**核の配置**です。031-8_6のHEは別の切片（`031-8_6_HE_reference_031-7_XY06`）なので、核の配置は対応しません。人が見ている間隙（組織間の白い / 暗い帯の網目）は、IFの `tissue` では核の粒に埋もれていました。
