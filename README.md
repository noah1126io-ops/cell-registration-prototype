# Cell Registration Prototype

外部ツールで作成済みの細胞・核データを読み込み、registration、matching、QC visualization を行うためのローカル Streamlit アプリです。

本アプリは研究用プロトタイプです。診断、治療方針決定、臨床判断、その他の医療用途には使用しないでください。

## アプリの位置づけ

このアプリは、segmentation 前の raw image を処理するアプリではありません。

Cellpose、StarDist、QuPath、既存解析パイプラインなどで作成済みの以下のデータを入力として使います。

- 点群 `.npy`
- 点群 CSV
- integer label mask
- GeoJSON nuclei segmentation
- 任意の QC 背景画像

raw image は必須入力ではなく、QC 表示用の背景として扱います。

## 統合方針

今後の基本方針は、UI では workflow を選べる形を維持しつつ、内部データモデルを共通化することです。

中心になる共通データは normalized point table です。

```text
point_id, centroid_x, centroid_y, source
```

各入力形式は、まずこの点群テーブルへ変換します。

```text
npy / csv / mask / GeoJSON
        ↓
normalized point table
        ↓
registration
        ↓
matching / QC visualization
        ↓
CSV / PNG / JSON export
```

mask 由来の特徴量や GeoJSON の world-µm 座標など、入力ごとの追加情報は保持しつつ、registration と visualization は可能な限り共通部品を使います。

## 起動方法

```bash
pip install -r requirements.txt
streamlit run app.py
```

8501 など他の Streamlit アプリとポートが重なる場合は、別ポートを指定します。

```bash
streamlit run app.py --server.port 8504 --server.address 127.0.0.1
```

OneDrive 配下で Streamlit のファイル監視が不安定な場合は、ファイル監視を切って起動します。

```bash
streamlit run app.py --server.port 8504 --server.address 127.0.0.1 --server.fileWatcherType none
```

## Workflows

### Workflow A: Point registration

点群データを直接入力する primary workflow です。画像なしで完結します。

| 項目 | 内容 |
| --- | --- |
| 入力 | fixed points `.npy` / `.csv`, moving points `.npy` / `.csv`, optional fixed/moving background image |
| 主な処理 | point table preview, density map generation, affine registration, transformed moving points, matching, scatter QC |
| 出力 | `matched_points.csv`, `transform_summary.json`, density map PNG, scatter/match QC PNG |
| 未実装 | non-rigid registration, moving-only unmatched point export, batch processing |

対応する CSV 列:

- `x, y`
- `centroid_x, centroid_y`
- `point_id, centroid_x, centroid_y`

`.npy` は shape `(n, 2)` を想定します。座標順は `xy` / `yx` を選択できます。

### Workflow B: Mask-derived point registration

integer label mask から細胞中心と特徴量を抽出し、その後は point registration と同じ考え方で registration / matching を行う secondary workflow です。

| 項目 | 内容 |
| --- | --- |
| 入力 | fixed mask, moving mask, optional fixed/moving image |
| 主な処理 | mask preview, cell feature extraction, centroid density map, affine registration, transformed moving mask/image/centroids, cell matching |
| 出力 | `fixed_cell_features.csv`, `moving_cell_features.csv`, `cell_correspondence.csv`, `transformation_summary.json`, density map PNG, overlay PNG |
| 未実装 | アプリ内 Cellpose 実行, production-grade non-rigid registration, moving-only unmatched cell export |

mask は `0` を背景、正の整数値を cell ID とする label image を想定します。

### Workflow C: HE-GeoJSON alignment

HE nuclei `.npy` と fluorescence nuclei GeoJSON を使う特殊座標系 workflow です。通常の画像表示は左上原点である一方、GeoJSON world-µm 座標は下向きでない座標系として扱われることがあるため、特に Y 方向の扱いを明示します。ICP、fine center-snap warp、Jacobian QC を扱うため、他の workflow とは独立して維持します。

| 項目 | 内容 |
| --- | --- |
| 入力 | HE nuclei centers `.npy`, fluorescence nuclei GeoJSON, optional HE image |
| 主な処理 | GeoJSON centroid loading, HE point loading, Y-flip centered orientation handling, optional X/Y-flip candidate selection, similarity ICP, affine ICP, robust fine center-snap warp, Jacobian QC, world-µm scatter QC |
| 出力 | transformed HE centers CSV, local translation anchors CSV, HE-GeoJSON transform summary JSON, affine/fine scatter QC PNG, warped HE image PNG |
| 未実装 | full-resolution tiled raster warp export, GeoJSON polygon overlay QC, production-grade warp report |

HE 側の `.npy` は StarDist などで事前検出済みの核中心を想定します。StarDist 由来ファイルでは座標順が `xy` か `yx` かを必ず確認してください。

Workflow C では、registration QC scatter の表示向きと warped HE image の出力向きを別々に指定できます。registration QC図だけが上下反転して見える場合は、`Registration QC display origin` を切り替えて確認します。warped HE image が上下反転して見える場合は、`Warped HE output origin` を `lower-left` / `upper-left` で切り替えて確認します。

## 既存 HE-to-GeoJSON 研究パイプラインの設計メモ

参考資料の既存パイプラインでは、1枚の HE image と fluorescence nuclei GeoJSON を対応させます。

- HE image を fluorescence nuclei GeoJSON の world-µm 座標系へ warp する
- HE 側の核中心は StarDist などで検出済みの `.npy` を使う
- fluorescence 側 GeoJSON は nuclei segmentation と centroid を world-µm 座標で保持する
- 通常のHE画像は左上原点なので、GeoJSON world-µm 座標と比較する際は主に Y-flip の有無を確認する
- 必要に応じて X flip も候補として比較できる
- global alignment として similarity ICP と affine ICP を行う
- fine alignment として mutual nearest-neighbor nuclei pair を使った center-snap warp を行う
- 一定距離を超える対応、信頼度が低い対応、局所的な変位と矛盾する対応は除外できる
- density map の局所patch相関から translation anchor を推定し、smooth displacement field を作成できる
- fine warp が reject された場合でも attempted displacement field と attempted point alignment をQC表示できる
- conservative / balanced / aggressive / debug の local translation preset を選択できる
- overlay 画像、warp JSON、warped HE image を出力する
- Jacobian min などで変形の破綻、特に fold-over を確認する

## 実装済みの主な部品

- `.npy` / `.csv` point loader
- GeoJSON centroid / polygon loader
- label mask からの cell feature extraction
- point table から minimal feature table への変換
- centroid density map generation
- affine registration
- similarity ICP / affine ICP
- Y-flip candidate selection
- optional X-flip candidate selection
- fine center-snap warp
- robust pair filtering for fine center-snap warp
- local translation field fine alignment
- local translation anchors CSV export
- attempted/applied fine alignment diagnostics
- warped HE image PNG export for QC
- Jacobian QC
- NaN area / eccentricity に対応した matching
- scatter plot / match overlay / density overlay
- CSV / PNG / JSON export

## segmentation source の方針

このアプリは Cellpose 前提ではありません。segmentation source は抽象化し、外部ツール由来のデータを normalized point table に変換して扱います。

想定する source:

- integer label mask
- StarDist 由来 nuclei center `.npy`
- Cellpose など外部ツール由来の mask / CSV
- QuPath / fluorescence segmentation 由来 GeoJSON
- centroid table CSV

## 注意

- 出力された transform、matching、QC は必ず目視確認してください。
- affine registration が失敗した場合は identity transform に fallback します。
- Workflow C の warped HE image は QC 用のMVP出力です。大きな画像の本格的な tiled export は今後の課題です。
- fine center-snap warp は Jacobian min が 0 以下の場合、局所的な fold-over の可能性があります。
- fine snap を強くしすぎると局所変形が破綻する可能性があるため、`Jacobian min` と overlay QC を確認してください。
