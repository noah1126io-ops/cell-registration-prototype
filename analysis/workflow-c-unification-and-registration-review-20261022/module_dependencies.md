# モジュールの依存関係（2026-10-22）

## 1. 新しく作ったモジュール

| モジュール | 役割 | 依存先 |
|---|---|---|
| `src/registration_core/__init__.py` | 公開API（`run_workflow`、`SideInput`、`WorkflowConfig`、`Constraints`、`similarity_from_landmarks`、`write_artifact`） | `workflow` |
| `src/registration_core/a2.py` | A2（`method_a2`、`refine`、`detector`、`masked_ncc` など）。`init_lib.py` から関数本体をそのまま移設 | `src/pointset_registration.py`（`icp_similarity`、`affine_icp`、`SimilarityTransform`） |
| `src/registration_core/search.py` | pose探索（`PoseSearch`、`Canvas`、`SearchConfig`）。`oir/search.py` からそのまま移設。`masked_ncc` だけを `a2` からimport | `a2` |
| `src/registration_core/pipeline.py` | `cropped_icp`、`evaluate`、`build_layers`、`run_method`。`oir/pipeline.py` からそのまま移設 | `a2`、`search`、`features`、`frames`、`metrics`、`src/pointset_registration.py` |
| `src/registration_core/features.py` | 構造map（tissue・核・edt）と品質判定。`oir/features.py` からそのまま移設 | `frames`、`tifffile`、`skimage` |
| `src/registration_core/frames.py`、`metrics.py` | `Frame`、`Affine2D`、`json_safe`、近接・mirror null・画像の整合 | numpy / scipy |
| `src/registration_core/workflow.py` | **C1–C4のorchestration**（新規）。画像の可否、探索の切り替え、除外・ROI・初期位置、判定、artifact | 上記すべて |
| `src/registration_core/workflow_d_bridge.py` | 採択済み結果 → Workflow C result artifact（新規） | `src/workflow_c_result.py`、`workflow` |
| `src/research_store.py` | 研究ストア（SQLite、追記のみ、schema v1）、エクスポート、注釈 → `Constraints` | `registration_core`（`constraints_from_annotations` のみ） |
| `src/workflow_c_research.py` | Workflow Cの研究モード：採択済み結果をglobal affineに（新規） | `registration_core.workflow_d_bridge`、`src/pointset_registration.AffineICPResult` |
| `src/registration_explorer/research_ui.py` | セッション、候補マップ、段階の診断、評価・注釈、engine実行、実験記録 | `research_store`、`jobs`、`results`、`catalog` |

## 2. 変更したモジュール

| モジュール | 変更 | 既存の動作への影響 |
|---|---|---|
| `app.py` | 研究モードのときだけ、STEP 2に研究global affineの選択欄（既定OFF）を追加。2か所のaffine計算で、選ばれた場合だけ置き換え | 研究モードOFF、またはチェックなしの場合は従来どおり（同じ関数・同じ引数） |
| `src/registration_explorer/page.py` | タブ構成（対応領域の探索、評価・注釈、実験記録）、日本語の名称、双方向viewer、ストアへの保存 | 研究モード専用ページのみ |
| `src/registration_explorer/viewer.py` | `boot(S, API)` に整理。双方向component（`interactive`）を追加。`poly_layer(pick=...)` | 一方向の `render` は同じ出力（`API=null`） |
| `src/registration_explorer/results.py` | 表示名、`SAFETY_TEXT`、engine artifactの読み込み（`record_from_engine_artifact`） | 既存artifactの読み込みは同じ |
| `src/registration_explorer/scene.py` | 候補polygonの色・pick id（tupleの長さを拡張）、boundsの計算 | 2要素tupleも従来どおり使える |
| `src/registration_explorer/jobs.py` | `WORKFLOWS`、`ENGINE`、engine runのcache key・検証・Slurm資源（12 GB、1.5 h） | 旧Runnerのcache keyは同じ式 |
| `src/registration_explorer/catalog.py` | `side_input`（manifest → `SideInput`） | 追加のみ |
| `scripts/run_registration_explorer_job.py` | `engine: registration_core` のconfigを `run_engine` へ振り分け | 旧configは従来どおり |

## 3. 変えていないもの

- `src/pointset_registration.py`、`src/density_flow.py`、`src/joint_flow.py`、`src/workflow_c_*.py`、`src/workflow_d.py`、Workflow A/B/C/D の既定動作。
- `analysis/` 配下の既存の研究成果（freezeのハッシュで確認、`REPORT.md` §7）。
- 採択規則と閾値。

## 4. 研究コードの移設とparity

| 対象 | 比較 | 結果 |
|---|---|---|
| A2 | `registration_core.a2.method_a2` と `init_lib.method_a2`（031-6_2） | M・tの差 0、detectorの判定も同じ（`cache/parity_a2.json`） |
| E0x | `registration_core.pipeline.run_method("E0x")` と `oir.pipeline.run_method`、および保存済みの10-19の結果 | 差 0（`cache/parity_e0x.json`） |
| 研究Workflow C（核のみ） | 10-19のE0x、10-20のRunner X3 | 差 0、NN median 1.9078 µm、≤3 µm 0.5995（`cache/wf_points_62.json`） |
| A2（画像なしの旧ペア） | 保存済みのR2 | 差 0（`cache/wf_a2_serial.json`） |
