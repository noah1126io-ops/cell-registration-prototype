# 入力監査：両画像付きサンプル 031-6_2 / 031-8_6（2026-10-19）

機械可読の結果は `sample_manifest.json` と `coordinate_consistency.csv` にあります。図は `input_image_overview.png` と `image_nuclei_overlay.png` です。監査スクリプトは `run_audit.py` です。

## 1. サンプルの所在

| sample | H&E画像 | 蛍光画像 | 核データ |
|---|---|---|---|
| 031-6_2 | `projects/031-6_2/031-6_2_HE_031-6_XY02.ome.tif` | `projects/031-6_2/031-6_2_seqif.ome.tif` | `031-6_2_seg_{he,seqif_level0,seqif_level1}.geojson` |
| 031-8_6 | `projects/031-8_6/031-8_6_HE_reference_031-7_XY06.ome.tif` | `projects/031-8_6/031-8_6_seqif.ome.tif` | `031-8_6_seg_{he,seqif_level0,seqif_level1}.geojson` |

- コピー元は `/home/atsushi.doi/nagasaki_lab/cardio_vascular/tmp/031-*_backgrounds/` です（2026-10-19にコピーし、SHA-256の一致を確認済み）。
- **031-8_6のH&Eは031-7（別の切片）です。** 提供者のメモに「REFERENCE ONLY - 031-8 has no HE; not registered to this field」とあります。同じ切片ではないので、細胞の同一性は期待できません。

## 2. 画像データ

| 項目 | H&E（両sample） | 蛍光 seqIF（両sample） |
|---|---|---|
| 形式 | OME-TIFF、pyramid（1, 2, 4, 8, 16） | OME-TIFF、pyramid（1, 2, 4, 8, 16） |
| Width × Height（level 0） | 6_2：11243 × 11411、8_6：11215 × 11400 | 10240 × 10240 |
| チャンネル | RGB明視野（uint8） | RGB表示用レンダリング（uint8）。生のチャンネル強度はない |
| Pixel spacing | 0.18872 µm/px | 0.1002 µm/px |
| 物理的な撮像範囲 | 約2120 × 2150 µm | 約1026 × 1026 µm |
| 原点・crop | **UNKNOWN**（OMEにstage座標がない）→ frame内の(0,0)を使う | **UNKNOWN** → 同左 |
| 向き | x右、y下（画像座標） | x右、y下 |

**蛍光チャンネルの意味**：R=Desmoplakin、G=Dystrophin1、B=Hoechst+Desmoplakin（提供者の `assets.json`）。

**核染色チャンネル**：純粋な核チャンネルはありません。DAPIなどがあるとは仮定せず、実測で確かめました。指標は、核の中心と周囲のリング（10 px L1）で値を比べたAUCです。

| channel | 6_2 AUC | 8_6 AUC | 判定 |
|---|---:|---:|---|
| R（Desmoplakin） | 0.43 | 0.48 | 核ではない |
| G（Dystrophin1） | 0.37 | 0.35 | 核ではない |
| B（Hoechst+Desmoplakin） | 0.86 | 0.88 | 核に近い |
| **B−R（Hoechstの推定）** | **0.89** | **0.90** | 核に近い → 蛍光側のnuclear mapに採用 |

## 3. 核データ

| 項目 | H&E側 | 蛍光側 |
|---|---|---|
| 出所 | StarDist `2D_versatile_he`（今回検出、prob 0.69、nms 0.3、downsample 2） | 提供者のGeoJSON `*_seg_seqif_level0.geojson` |
| 核数 | 6_2：1,242、8_6：4,800 | 6_2：1,091、8_6：1,552 |
| 座標 | HE level-0 px → world = px × 0.18872 µm | seqIF level-0 px → world = px × 0.1002 µm |
| ID | 連番 | `segment_id`（float） |
| 画像内に収まる割合 | 1.00 | 1.00 |

座標の整合性チェック（`coordinate_consistency.csv`）：

- **蛍光GeoJSON ↔ 蛍光画像**：核mask内でB−R信号が最大になる位置のずれは0.2 µm（1 px L1）でした。GeoJSONは画像に正しく登録されています。
- **level1 GeoJSON**：level0/2と一致しました（中央値の差0.7 px L0）。
- **centroid property ↔ polygon重心**：中央値4.8〜5.7 px L0（約0.5 µm）、p95 11〜14 px のずれがあります。評価にはpolygonの重心を使います（CHECK）。
- **H&E StarDist ↔ H&E画像**：重心の下でhematoxylinが最大になる位置のずれは0 µmでした。xyの順序とframeは一致しています。
- **提供者の `seg_he.geojson`**：蛍光座標を純粋なscale＋translationで写したもの（残差：6_2で0.005 px、8_6で0.37 px）です。
  - scaleは物理spacing比の0.985倍（6_2）／0.9825倍（8_6）で、提供者の値と一致します。
  - つまり、この座標は**提供者が推定した変換で蛍光の核を写したもの**であり、H&Eから独立して検出された核ではありません。そのため、H&E側の核には今回のStarDist検出を使いました。

## 4. 撮像領域

- 画像サイズの違い（約11k px vs 10k px）と、物理的な範囲の違い（約2.1 mm vs 1.0 mm）は区別しています。撮像範囲は「画像サイズ × spacing」で求め、点群のbounding boxは使いません。
- 提供者の配置では、蛍光のfootprintが100% H&Eの画像内に入ります（両sample）。**partial overlapの問題（小さい視野が大きい視野の中のどこにあるか）**として扱えます。
- 組織が占める割合（2 µm grid、Otsu）：
  - 6_2：HE 0.18、IF 0.36。組織はまばらです。
  - 8_6：HE 0.63、IF 0.40。組織は密で、似た構造が繰り返しています。

## 5. 参照変換（正解ではない）

| sample | 参照 | 位置づけ |
|---|---|---|
| 031-6_2 | 提供者による画像NCCの変換（回転0固定、scale 0.975〜0.995の格子） | **同じ切片**。独立していない参照で、ground truthではない。`ref_median_um` は、この参照に対する核位置の差 |
| 031-8_6 | 視野中心へのおおまかな配置（登録されていない） | **別の切片**。参照として使えない。正解の領域は未注釈 |

## 6. 監査の結論

| 項目 | 031-6_2 | 031-8_6 |
|---|---|---|
| pixel spacing | 既知 | 既知 |
| 原点・crop | UNKNOWN（frame-localで扱う） | UNKNOWN |
| 画像 ↔ 核の対応 | 両側で確認済み | 両側で確認済み |
| 蛍光の核情報 | B−Rで代用（AUC 0.89） | B−Rで代用（AUC 0.90） |
| 画像融合 | **可**（IMAGE_USABLE） | **可**（IMAGE_USABLE）。ただし別切片 |
| 正解 | 独立でない参照のみ | なし |

- 原点がUNKNOWNなので、2つの画像の絶対的な位置関係は確定していません。位置関係はすべてregistrationで推定し、結果を「画像座標系の一致を確認した」とは主張しません。
- 画像と核の対応は両側で確認できたので、画像融合は止めていません。
