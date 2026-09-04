# Cell Registration Prototype

外部ツールで作成済みの細胞・核データを読み込み、registration、matching、QC visualization を行うためのローカル Streamlit アプリです。

本アプリは研究用プロトタイプです。診断、治療方針決定、臨床判断、その他の医療用途には使用しないでください。

## 現在の実装状況

Phase 4 まで実装済みです。Workflow C の実験的な Density Flow と Joint Density + Tissue-Structure Flow は、同一の計算実装を NumPy/SciPy（CPU）または CuPy/cupyx（CUDA）で実行できます。Joint FlowもStage A/Bの両方で既存Density Flow backendを再利用します。CPU が既定で、CUDA は Python API の `device="cuda"` または `device="auto"` から選択します。現時点の Streamlit UI はバックエンド選択を公開していないため、UIからの実行はCPUです。

Phase 4 には、float64 の CPU/CUDA バックエンド、Density FlowおよびJoint Flow Stage A/B反復中の配列のGPU常駐、バックエンド provenance、同期を含む処理時間の計測、CPU/CUDA parity test が含まれます。Joint Flowの中間HE/mask/structure前処理は明示的なCPU境界です。対象範囲とホスト・デバイス間転送の境界は [Density Flow array residency](docs/DENSITY_FLOW_BACKEND.md) を参照してください。

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

CUDA バックエンドを利用する環境では、通常の依存関係に加えて任意依存を導入します。`requirements-cuda.txt` は CUDA 13 系向けです。

```bash
pip install -r requirements.txt
pip install -r requirements-cuda.txt
python scripts/cupy_smoke_test.py
```

Python API から Density Flow のバックエンドを指定できます。

```python
from src.density_flow import tissue_aware_density_flow_registration

result = tissue_aware_density_flow_registration(
    fixed_points,
    moving_points,
    device="cuda",  # "cpu"（既定）/ "cuda" / "auto"
)
```

`cuda` は利用可能なCUDAデバイスとCuPyを必須とし、利用不能な場合は明示的にエラーになります。`auto` は起動時に一度だけ判定し、CUDAが利用可能ならCUDA、そうでなければCPUを選択します。CPU/CUDA parity を確立するため、現在は両方とも `float64` 固定です。

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

### Workflow C: Point Registration

HE nuclei `.npy` を fluorescence nuclei GeoJSON の world-µm 座標へ登録し、その結果を再利用可能な artifact として保存する workflow です。HE画像や tissue mask を fine registration の計算補助に使う場合はありますが、最終 HE raster image の warp と出力は行いません。

| 項目 | 内容 |
| --- | --- |
| 入力 | HE nuclei centers `.npy`, fluorescence nuclei GeoJSON, optional HE image |
| 主な処理 | GeoJSON centroid loading, HE point loading, Y-flip centered orientation handling, optional X/Y-flip candidate selection, similarity ICP, affine ICP, cluster-anchor / local translation / experimental tissue-aware density-flow point warp, Jacobian QC, world-µm scatter QC |
| 出力 | transformed HE centers CSV, displacement fields, point-set metrics, Jacobian/safety QC, scatter QC PNG, `workflow_c_registration_result.zip` |
| 未実装 | GeoJSON polygon overlay QC, production-grade registration report |

HE 側の `.npy` は StarDist などで事前検出済みの核中心を想定します。StarDist 由来ファイルでは座標順が `xy` か `yx` かを必ず確認してください。

`workflow_c_registration_result.zip` には `registration_result.npz`、`metrics.json`、`parameters.json`、`provenance.json`、`manifest.json` が含まれます。NPZには fixed/original/affine/attempted/applied points、affine transform、attempted/applied displacement field、grid、bounds が保存され、Workflow D は registration を再計算せず利用します。

`Tissue-aware density flow [Experimental]` は独立実装の実験方式です。Workflow C では点群と変位場を計算し、画像への inverse raster mapping は Workflow D が担当します。Density Flow の反復グリッド計算はCPU/CUDAに対応していますが、前処理、点群metric、plot、artifact exportなどはCPUで実行します。設計は [docs/DENSITY_FLOW_METHOD.md](docs/DENSITY_FLOW_METHOD.md)、独立実装の来歴は [docs/DENSITY_FLOW_PROVENANCE.md](docs/DENSITY_FLOW_PROVENANCE.md)、バックエンド境界は [docs/DENSITY_FLOW_BACKEND.md](docs/DENSITY_FLOW_BACKEND.md) を参照してください。STalignとの同等性や生物学的精度向上は主張していません。

### Workflow D: Raster Deformation

Workflow C が保存した registration result と元の HE image を入力し、既存の inverse raster mapping と raster QC を実行します。固定 GeoJSON 点は移動せず、登録パラメータや変位場も再推定しません。

| 項目 | 内容 |
| --- | --- |
| 入力 | raw HE image, `workflow_c_registration_result.zip` |
| 主な処理 | affine HE生成, attempted/applied inverse raster warp, safety-gated affine fallback, checkerboard, edge overlay, difference/Jacobian/warp-grid QC |
| 出力 | affine HE PNG, attempted warped HE PNG, final applied HE PNG, raster QC JSON, warp metadata JSON, Workflow D result ZIP |
| 未実装 | full-resolution tiled raster export, GPU backend, Slurm integration |

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
- tissue-aware density-flow point registration と displacement field 計算（Workflow C）
- Density Flow の NumPy/SciPy CPU backend と CuPy/cupyx CUDA backend
- Joint Density + Tissue-Structure Flow Stage A/B のCUDA backend再利用
- CPU/CUDA backend provenance と runtime breakdown
- 保存済み displacement field による inverse raster mapping（Workflow D）
- local translation anchors CSV export
- attempted/applied fine alignment diagnostics
- Workflow C result artifact export と Workflow D raster result export
- Jacobian QC
- NaN area / eccentricity に対応した matching
- scatter plot / match overlay / density overlay
- CSV / PNG / JSON export

## テスト

CPUを含む全テストは次のコマンドで実行します。CUDAデバイスがない環境ではGPU専用テストがskipされます。

```bash
python -m pytest -q
```

CUDAノードでは、まずsmoke testを実行し、その後にバックエンドとparityのテストを実行します。

```bash
python scripts/cupy_smoke_test.py
python -m pytest -q \
  tests/test_array_backend.py \
  tests/test_density_flow_backend.py \
  tests/test_density_flow_gpu_parity.py
```

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
- Workflow C は最終 HE raster を生成しません。保存した result artifact と元画像を Workflow D へ入力してください。
- Workflow D の warped HE image は QC 用のMVP出力です。大きな画像の本格的な tiled export は今後の課題です。
- CUDA対応はWorkflow CのDensity FlowおよびJoint Flow Stage A/B反復グリッド計算が対象です。Workflow Dのraster deformation、Streamlit UIからのGPU選択、Slurmジョブ投入は未実装です。
- fine center-snap warp は Jacobian min が 0 以下の場合、局所的な fold-over の可能性があります。
- fine snap を強くしすぎると局所変形が破綻する可能性があるため、`Jacobian min` と overlay QC を確認してください。
