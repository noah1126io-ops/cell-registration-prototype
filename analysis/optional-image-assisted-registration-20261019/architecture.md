# Optional Image-Assisted Registration：研究用アーキテクチャ（2026-10-19）

analysis専用のprototypeです（`oir/`）。Production Workflow C/D・UI・A2/ICP・Graph/UOT/B2・safety gateは変更せず、importして呼ぶだけにしています。

## 1. 全体の流れ

```text
             ┌──────────── Sample (frames + nuclei + optional images) ────────────┐
             │ HE frame  : pixel_um, shape, origin(UNKNOWN→frame-local), image?   │
             │ IF frame  : pixel_um, shape, origin(UNKNOWN→frame-local), image?   │
             │ nuclei    : 両側とも world µm（frame内）                             │
             └───────────────┬────────────────────────────────────────────────────┘
                             │ hide_images=(...) で実験ごとに見せる入力を制御
                ┌────────────┴─────────────┐
          image visible?                 points only
          ┌────┴────┐                        │
     features.extract（modality別）          │
     tissue / nuclear / edt + quality        │
          │ usable?  ──no──► IMAGE_INCOMPATIBLE ─► POINT_ONLY_FALLBACK（E0x）
          ▼                                  ▼
   PoseSearch（共通エンジン）: layer = points | image | cross
   chirality × rotation × scale の格子、translationはmasked NCC（既存A2の関数）
   → candidates（robust z、distinct化、margin、verdict UNIQUE/AMBIGUOUS/LOW_PEAK）
                             │
          cropped production ICP（icp_similarity → affine_icp、相手の撮像範囲内の点だけ）
          image系はtrust region（scale ±3%、回転 ±2°、shift 15 µm）を外れたらICPを棄却
                             │
          evaluate（常にIF frameで）・held-out・acceptance → TransformArtifact
```

## 2. モードとメソッド

| Mode | 入力 | メソッド | 内容 |
|---|---|---|---|
| P0 | 点群のみ | **E0** | 既存 `init_lib.method_a2`（A2 + production refine）を**無変更で呼ぶ**。画像を渡しても無視する |
| P0 | 点群のみ | E0x | 点群密度layerのみ。canvasを両視野の和集合サイズにし（大きい側が切れない）、ICPは相手の撮像範囲内の点に限る |
| P1 | 点群＋片側画像 | E1 | points layer ＋ cross layer（画像側のtissue占有 vs 画像のない側の核密度）。robust zの和で候補を順位付け |
| P2 | 点群＋両画像 | E2 | image layer（構造表現）のみで探索し、production ICPで精密化（trust region内に限る） |
| P2 | 点群＋両画像 | E3 | image layer ＋ points layerを結合。結合規則 `zsum`／`zmin`／`rerank` を比較 |

- E0は既存の点群Pipelineそのもので、互換性確認の基準です（Phase B）。
- E0xは「大きさの違う視野」を扱えるよう点群探索を一般化したもので、画像は使いません。
- 画像scoreと点群scoreを生のまま足すことはしません。各NCC mapをrobust z（median/MAD）に変換してから、明示した規則で結合します。

## 3. 座標フレームと変換（`oir/frames.py`）

- `Frame`：`pixel_to_world` / `world_to_pixel`、`extent_um`、`footprint_um`。
  - 撮像範囲は「画像サイズ × pixel spacing」で求め、点群のbounding boxは使いません。
  - pixel spacingが不明（`None`）なら変換を拒否します（例外）。
  - originはOMEにstage座標がないためUNKNOWNとし、frame内の(0,0)を使います。
- `Affine2D(M, t, source_frame, target_frame)`：
  - `apply`（点の順写像）、`inverse`、`compose`（frame名が合わなければエラー）、`params`、`pixel_matrix`（pixel→pixelの3×3行列）を持ちます。
- **計算方向と出力frameは分離しています。**
  - どちらをFixedにしても、評価は `to_if()` でHE→IFの写像に揃えて行います。
  - 画像のresamplingは逆写像（target pixel → source pixel、`metrics.warp_map_into`）で行い、点の順写像とは別の経路です。
- `TransformArtifact` が持つ項目：
  - `source_frame` / `target_frame`、`forward_transform` / `inverse_transform`
  - `transform_direction`、`output_frame`、`model`
  - `roi`、`validity_region_target_um`（moving footprintをtargetへ写したもの）
  - `inputs`（探索に使った画像特徴量）、`image_status`
  - `confidence`、`failure_reasons`、`accepted`
  - `resampling`（逆写像、bilinear、範囲外は定数0で外挿なし、有効領域）

## 4. 画像由来の構造表現（`oir/features.py`）

生の輝度は比較しません。各modalityを、共通の物理grid（既定 2 µm/px）上の同じ表現に写します。

| 表現 | HE | IF（表示用RGB：R=Desmoplakin、G=Dystrophin1、B=Hoechst+Desmoplakin） |
|---|---|---|
| tissue | optical density（RGB合計）、σ 6 µm、percentile正規化 | 全channelの最大値、σ 6 µm、percentile正規化 |
| nuclear | hematoxylin（色分解） | B−R（Hoechst推定）。入力監査でAUC 0.89〜0.90を確認してから採用 |
| edt | tissue envelopeまでの符号付き距離（±60 µm） | 同左 |

- `tissue_bin`（同じmapのOtsu二値化）と `envelope` は、maskやDiceの計算に使います。
- **quality**（`usable`の判定条件）：
  - tissueがFOVに占める割合が0.02〜0.995。
  - 自分の核位置での核mapの値が、組織内のランダムな位置に対してAUC > 0.7。
- `usable=False` の場合は `IMAGE_INCOMPATIBLE` とし、点群のみに戻ります。

## 5. Fallbackと状態

| 状態 | 条件 |
|---|---|
| IMAGE_AVAILABLE | 実験で画像が見えている |
| IMAGE_USABLE | 見えている画像がすべてquality条件を満たす |
| IMAGE_INCOMPATIBLE | quality条件を満たさない → 画像融合を中止 |
| IMAGE_REGISTRATION_UNCERTAIN | 探索verdictがUNIQUEでない（AMBIGUOUS / LOW_PEAK）、または核配置の幾何学的整合が不合格 |
| POINT_ONLY_FALLBACK | 上の2つのいずれかのとき、同じsampleのE0x（なければE0）の結果を採用し、その事実をartifactに記録 |

- 幾何学的整合の判定（事前固定）：≤3 µmの近接率 ≥ 0.25、かつ近接率が鏡像nullの3倍以上。
  - 閾値はA2の既存検出器と同じ値で、今回のsampleに合わせた調整はしていません。
- 画像scoreが高くても、核配置が不整合なら採択しません。
- 画像融合に失敗しても、点群のみの結果は失いません。

## 6. Workflow C/Dへ統合する場合の境界

- **Workflow C**（変換推定）：
  - 入力parserを `Sample`（frames＋nuclei＋任意の画像）に置き換えます。
  - 探索の入口を `PoseSearch` に差し替えます。
  - 出力を `TransformArtifact` に拡張します。
  - ICP本体・A2・safety gateは、そのまま呼び出します。
- **Workflow D**（変換の適用）：
  - artifactから `inverse_transform`・frame（pixel_um, origin）・validity regionを読み、既存の逆写像resamplingを適用するだけです。
  - Workflow Dは変換を再推定しません。
  - 現在のWorkflow D（`registration_result.npz`）との差分：
    - frameの明示（source/target、pixel spacing）
    - 逆変換
    - validity polygon
    - image_status / accepted
- **Point-onlyの経路**：artifactの `inputs.images_used_in_search=[]`、`mode=P0` として、既存と同じ挙動を維持します。

## 7. 制約の記録

- Production・既存analysisは変更していません（`freeze/` に作業前後のSHA-256を保存）。
- commit / pushはしていません。
- 非剛体変形は推定しません（Similarity / Affineのみ）。
