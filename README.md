# Cell Registration Prototype

外部ツールで作成済みの細胞・核データを読み込み、registration、matching、QC visualization を行うためのローカル Streamlit アプリです。

本アプリは研究用プロトタイプです。診断、治療方針決定、臨床判断、その他の医療用途には使用しないでください。

研究の現状と引き継ぎ（エージェント向け）は [docs/RESEARCH_STATUS.md](docs/RESEARCH_STATUS.md) にまとめています。

## 現在の実装状況

Workflow C の実験的な Density Flow と Joint Density + Tissue-Structure Flow は、同一の計算実装を NumPy/SciPy（CPU）または CuPy/cupyx（CUDA）で実行できます。Joint FlowもStage A/Bの両方で既存Density Flow backendを再利用します。CPU が既定で、CUDA は Python API の `device="cuda"` または `device="auto"` から選択します。Streamlit UIでは既存のローカルCPU実行を維持し、明示的なGPUボタンからSlurmへ非同期投入できます。CUDAをgateway上のStreamlitプロセスで直接実行することはありません。

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

### Registration Explorer (Research) と研究Workflow C

研究モード専用の画面です。異なる切片の対応する組織領域を探し、候補を比べ、位置合わせを検証・修正し、その結果を保存します。Production Workflow C/D の既定の処理は変わりません。

**処理の流れ**（共通エンジン `src/registration_core`。Slurmの研究Runner・Explorer・Workflow Cの研究モードは、どれもこのエンジンを使います）

1. **C1 対応領域の探索**：両側の画像が位置合わせに使える品質なら、画像（組織・核の構造map）で先に探索します。核点群による探索も必ず別に行い、両方の候補を残します。画像が使えない、または画像がない場合は、核点群だけで探索します。
2. **C2 精密化**：候補ごとに、重なり領域の核だけでICPを行います。ICPが候補を別の領域へ動かした場合は記録します。
3. **C3 検証**：事前に固定した規則で判定します。探索が一意（UNIQUE）で、≤3 µm ≥ 0.25 かつ偶然一致の3倍以上の場合に採択です。注釈を使った実行は自動採択しません。
4. **C4 記録**：`artifact.json`（`cellreg-registration-artifact/1`）に、候補・判定・入力のfingerprint・版を保存します。

**起動と停止**（研究室サーバー上。localhostのみで待ち受けます）

```bash
analysis/e0x-difficulty-and-web-integration-20261020/dev_server.sh start 8511    # CELLREG_RESEARCH_MODE=1 で起動
analysis/e0x-difficulty-and-web-integration-20261020/dev_server.sh status 8511
analysis/e0x-difficulty-and-web-integration-20261020/dev_server.sh stop 8511
```

コードを変えたら、`stop` → `start` で反映します（file watcherは無効です）。ブラウザからは、SSHのポート転送（`ssh -N -L 8511:127.0.0.1:8511 <user>@<gateway>`）を通して `http://localhost:8511` を開き、サイドバーの「Workflow」で **Registration Explorer (Research)** を選びます。

**画面の使い方**

| タブ | できること |
|---|---|
| 対応領域の探索 | Fixed全体に、全候補の対応領域を探索方法ごとの色で表示します。候補の表（回転・倍率・平行移動・推定重なり・スコア・ICPの移動量・核配置・レビュー）と、どの段階（C1/C2/C3）で止まったかを確認できます。表の行や地図上の候補をクリックすると、Viewerがその候補に切り替わります |
| Viewer（重ね合わせ・手動調整） | H&E・蛍光の重ね合わせ、Before/After、Zoom/Pan、核と推定対応、FOV・組織・候補領域を表示します。手動調整（回転・倍率・平行移動。自動推定との比較つき）もここで行います。左の『クリック操作』で、対応点（Fixed → Moving の順にクリック）やROI（ドラッグ）を指定し、注釈として保存できます。表示中の配置は、『誤りの候補（除外）』や『手動の初期位置』として保存できます |
| 評価・注釈 | 候補ごとに『正しい / 間違い / 判断できない』と問題の種類（領域・Rotation・Scale・Translation・核配置・画像構造・表示座標系・その他）、コメントを保存します |
| Warp Preview | 逆写像による画像の重ね合わせです。手動調整中の変換でも確認できます（PREVIEW ONLY） |
| 新規実行 | 研究Workflow CをSlurmに投入します。選べる処理は、自動（画像優先＋核）、核のみ、従来法A2、注釈を使った探索です。旧Runner（X0–X3）は折りたたみの中にあります |
| 実験記録 | セッション内やサンプル全体の実行・変換・評価・注釈・操作履歴を表示し、JSON / CSV（zip）でエクスポートします |
| Artifact | 保存されたartifactを表示します。採択済みで完全自動の結果だけは、Workflow Dが読み込める形式（Workflow C result artifact、affineのみ）で書き出せます |

**保存**：画面上部で実験セッションを選ぶか、新しく作ります。評価・注釈・手動変換・実行は、`runs/research-store/research.sqlite`（SQLite、追記のみ、schema v1）に記録されます。ブラウザを再読込しても、アプリを再起動しても残ります。保存した注釈は、『新規実行』で明示的に選んだときだけ探索に使われます（assistedとして記録）。

**Workflow C（Production画面）との接続**：研究モードでは、STEP 2に「研究モード：Global位置合わせを『研究Workflow C』の採択済み結果にする」の欄が出ます（既定はOFF）。チェックして採択済みの結果を選ぶと、global affineがその結果に置き換わります。STEP 3以降は従来どおりです。

| 環境変数 | 役割（既定） |
| --- | --- |
| `CELLREG_RESEARCH_MODE` | `1` のとき研究機能を表示する（未設定なら非表示） |
| `CELLREG_RESEARCH_STORE` | 研究ストアのSQLite（`runs/research-store/research.sqlite`） |
| `CELLREG_EXPLORER_MANIFEST` | サンプル登録のJSON（`analysis/e0x-difficulty-and-web-integration-20261020/viewer_sample_manifest.json`） |
| `CELLREG_EXPLORER_RUNS_DIR` | UIから実行した研究ジョブの保存先（`runs/registration-explorer`） |
| `CELLREG_EXPLORER_MAX_ACTIVE` | 同時に動かせる研究ジョブの上限（2） |
| `CELLREG_PYTHON` / `CELLREG_E0X_LAB_DIR` | Slurmジョブで使うPythonと、旧Runnerの研究コードのディレクトリ |
| `CELLREG_REGION_CELL_MATCHING` | `1` のとき、Explorer に研究用タブ「核対応（研究）」を表示する（未設定なら非表示）。`analysis/region-cell-matching-20261113` の保存済みの結果を表示するだけで、計算も保存もしない |

**コマンドラインから実行する場合**：`python -c "from src.registration_core import run_workflow, WorkflowConfig; ..."` と書くか、`analysis/workflow-c-unification-and-registration-review-20261022/run_core_checks.py` を参考にしてください。設計は `analysis/workflow-c-unification-and-registration-review-20261022/to_be_architecture.md` にあります。

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

## Workflow C GPU Slurm worker

Streamlitプロセスはgateway上に残し、Workflow CのGPU計算だけを1 registration = 1 GPUのSlurm jobとして投入できます。CLIに加えてWorkflow C画面からの明示的な非同期submit、status refresh、直近run/Run ID復元、artifact download、既存Workflow Dへのhandoffに対応しています。

```bash
python scripts/submit_workflow_c.py \
  --config examples/workflow_c_synthetic_gpu.json \
  --runs-dir runs
```

投入時に表示されるrun directoryの `status.json` を確認します。状態履歴は `queued`、`running`、`completed` または `failed` として保存されます。

```bash
cat runs/<run_id>/status.json
cat runs/<run_id>/slurm_job.json
```

run directoryは次の構造です。

```text
runs/<run_id>/
  config.json
  input/
  status.json
  slurm_job.json
  stdout.log
  stderr.log
  result/
    workflow_c_registration_result.zip
```

実入力configの `inputs` にはpre-normalized `.npy` の `fixed_points` / `moving_points` を指定します。Joint Flowではさらに `.npy` の `he_image` / `tissue_mask` とJSONの `metadata` が必要です。submitterがこれらをrun directoryへコピーし、workerはjob開始時に `$SLURM_TMPDIR`、未設定なら `/tmp/cellreg_${SLURM_JOB_ID}` へコピーします。計算後は検証済みの最終artifactだけをshared NFSへatomic copyし、copy検証成功後にscratchを削除します。

workerはSlurm allocationとCUDA GPUを必須とし、直接実行やGPU未割当時は明示的に失敗します。batch resourceは `mib-dbia`、1 node、1 task、1 GPU、4 CPU、16 GB、30分です。

## Workflow C Joint Flow GPU parameter sweep

CLIから決定的なgrid searchを作成し、Slurm job array（既定の同時実行数2、1試行1GPU）で実行できます。基準試行を必ず含め、入力を一度だけ共有領域へ保存し、完全な試行設定・チェックサム・結果CSVを残します。安全判定を通った試行だけをランキングします。

```bash
python scripts/create_workflow_c_sweep.py --sweep-config examples/workflow_c_sweep_synthetic.json
python scripts/submit_workflow_c_sweep.py --sweep-dir sweeps/<sweep_id>
python scripts/summarize_workflow_c_sweep.py --sweep-dir sweeps/<sweep_id>
```

探索対象の8パラメータ、100試行の上限、baseline、scratch・失敗時の扱い、CSV指標は [Workflow C sweeps](docs/WORKFLOW_C_SWEEPS.md) を参照してください。数値アルゴリズム・安全閾値は変更せず、Streamlit sweep UIはこの段階には含めません。

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
- CUDA対応はWorkflow CのDensity FlowおよびJoint Flow Stage A/B反復グリッド計算が対象です。Workflow Dのraster deformationは未実装で、C→D handoff後もWorkflow Dは既存CPU経路で実行します。
- fine center-snap warp は Jacobian min が 0 以下の場合、局所的な fold-over の可能性があります。
- fine snap を強くしすぎると局所変形が破綻する可能性があるため、`Jacobian min` と overlay QC を確認してください。
