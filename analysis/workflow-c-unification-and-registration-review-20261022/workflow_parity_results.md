# Parity と回帰の結果（2026-10-22）

すべてSlurm `mib-dbia`（CPU 4コア）で実行しました。ファイルは `cache/` にあります。「差」は変換行列Mと並進tの各成分の最大絶対差です。

## 1. 研究コードの移設（analysis → `src/registration_core`）

| 確認 | 比較 | 差（M / t） | 判定 | ファイル |
|---|---|---|---|---|
| A2 | `registration_core.a2.method_a2` と `init_lib.method_a2`（031-6_2、HE moving → IF fixed） | **0 / 0** | detectorの判定も同じ（FAIL / FAIL） | `parity_a2.json` |
| E0x | `registration_core.pipeline.run_method("E0x")` と `oir.pipeline.run_method`（031-6_2、HE fixed） | **0 / 0** | verdictも同じ（UNIQUE） | `parity_e0x.json` |
| E0x | 新しいエンジンと、保存済みの2026-10-19の結果（`real_031-6_2_E0x_HEfix.json`） | **0 / 0** | | `parity_e0x.json` |

## 2. 研究Workflow C（C1–C4）と既存の結果

| 確認 | 比較 | 差 | 結果 | ファイル |
|---|---|---|---|---|
| 031-6_2、核のみ（画像を使わない） | 2026-10-19のE0x、2026-10-20の旧Runner（X3） | **0 / 0**（どちらとも） | ACCEPTED。NN median **1.9078 µm**、≤3 µm **0.5995**、null 0.049（IF frame） | `wf_points_62.json` |
| 031-6_2、自動（画像優先、HE fixed） | 画像探索の探索poseと、2026-10-19のE2tnの保存値 | **0 / 0**（探索pose） | ACCEPTED（画像・核の両探索がUNIQUEで同じ領域）。最終はNN median 1.9078 µm、≤3 µm 0.5995 | `wf_auto_62_HE.json` |
| 031-6_2、自動（IF fixed） | 2026-10-19のE0x IFfix（1.886 µm） | — | ACCEPTED。NN median 1.8855 µm、≤3 µm 0.5985 | `wf_auto_62_IF.json` |
| 画像なしの旧ペア（serial_section_pair_20260904）、A2 | 保存済みのA2（R2） | **0 / 0** | 画像も物理範囲もないため、A2のdetectorで判定 | `wf_a2_serial.json` |
| 画像の読み込み失敗（031-6_2のH&E画像パスを存在しないファイルに） | — | — | `IMAGE_READ_FAILED` を記録して核探索に移行。ACCEPTED、1.9078 µm | `wf_imgfail.json` |

## 3. 実行経路の違い：Explorer / Slurm Runner と直接呼び出し

UIの『新規実行』と同じ経路（`jobs.submit` → sbatch → `run_registration_explorer_job.py` → `run_workflow`）と、`run_workflow` の直接呼び出しを比べました。研究ストアのセッション「検証 2026-10-22」に登録しています（`parity_runner_vs_direct.json`）。

| sample | Runnerのrun | 全候補の探索pose・最終変換の差 | 判定（Runner / 直接） | 選ばれた候補 | NN median（Runner / 直接） | 入力fingerprint |
|---|---|---|---|---|---|---|
| 031-6_2 | 20261022-155543-fd1a6e | **0 / 0** | ACCEPTED / ACCEPTED | image#0 / image#0 | 1.9078 / 1.9078 µm | 同じ |
| 031-8_6 | 20261022-155543-dfafb7 | **0 / 0** | NOT_ACCEPTED / NOT_ACCEPTED | image#0 / image#0 | 13.58 / 13.58 µm | 同じ |

## 4. 表示の座標（既知の変換で定量確認）

| 確認 | 方法 | 結果 |
|---|---|---|
| 核の表示位置 | 表示中の変換 × 提供者のGeoJSONから読んだ蛍光核の座標と、viewerに描かれた点の座標（ブラウザの中で読む、Test S） | 最大差 < 0.02 µm（表示用に0.01 µmへ丸めているため） |
| 画像と核の整合 | 核のlevel-0画素の中心を、Moving画像layerの画素 → 表示座標の行列で写した位置と、同じ核の描画位置 | pytest：差 < 1e-6 µm。ブラウザ（Test S）：画素の半分未満 |
| 対応点のクリック | Fixedの核と、写したMovingの核をクリックし、Python側で求めたMoving座標を、元の核座標（GeoJSON）と比べる（Test L） | 表示の画素1つ分以内（`ui_test_results.md`） |
| 合成データ（既知の相似変換、部分重なり） | 400 µmのMovingを30°回転して2,100 µm中に配置し、点だけで研究Workflow Cを実行（pytest） | 採択。四隅の誤差の中央値 < 2 µm |
| Workflow D用の書き出し | 既知の変換で書き出したartifactを読み込み、H&E px → 蛍光 µm で写した点（pytest） | 1e-9以内で一致。fine stageは `off`、`fine_applied=False` |
| Workflow Cのglobal affine（研究結果から） | 031-6_2の採択済み結果からProductionの形（HE px、反転なし）に変換し、研究の変換と比べる（pytest） | 1e-6 µm以内で一致。別の点を入れると拒否（ValueError） |

## 5. テスト一式

- `pytest tests`（`--gres=gpu:1` のノード）：**262 passed、1 skipped**。既存の248件に、新しい14件を加えたものです（`test_registration_core.py` 9件、`test_research_store.py` 5件）。
- 新しいテストでの確認内容：
  - 規則の判定（fallback、CONFLICT、C1/C2/C3、assistedは自動採択しない）
  - 除外後のverdictの再計算
  - 画像の有無と読み込み失敗の区別
  - 合成データでのE2E
  - Workflow D用の書き出し（拒否と変換）
  - ストアの追記のみ（triggerで拒否）
  - 再起動後の読み込み、サンプル間の分離
  - artifactの差し替えの禁止
  - 注釈 → 制約（方向の扱いを含む）
  - エクスポート
