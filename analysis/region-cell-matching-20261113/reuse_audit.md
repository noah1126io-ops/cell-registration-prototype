# 既存の研究資産の監査と再利用計画（2026-11-13、実装の前）

- 対象：`feature/gpu-slurm`、HEAD `dd68fca`。作業前の時点で未コミットの変更なし、`origin` と同期済み。

## 1. C1 の Top-k 候補を取り出す最小の方法
- エンジンは、すべての探索（image / points / a2）について上位 `n_candidates`（既定 6）件を残している。それぞれに C2（ICP）の結果と C3 の記録を付けて、artifact の `candidates` に保存している。
  - 1 件の中身：`id`、`source`、`rank`、`T_search`、`T_final`、`flags`、`validation.fixed_frame`
- そのため、**エンジンの変更は不要**。
  - 保存済みの artifact と、11-11 のベンチマークの `result.json`（`gates.<gate>.candidates`）は、そのまま読める。
  - 新しく計算するときは、`workflow.localize` と `workflow.refine_and_validate` を呼べばよい。
- 実装：`src/registration_research/candidate_adapter.py`（`from_artifact`、`from_benchmark_result`、`run_c1_topk`）。
- 11-11 の結果からわかる制約：8_6 系の正例では、核探索（A）と旧特徴（B）の上位 6 件の中に、正しい領域が 1 つもない（`localization_comparison.csv` の `topk_hit_rank` が空）。occ+cleft（C）は、すでに 1 位が正しい。実データで「正解が 2 位以下にある」のは、P6_2_IFhalfL の核探索（2 位）の 1 件だけ。**実データには救済の機会がほとんどない**ので、救済できるかどうかは合成データ（SYN-C1）で調べる。

## 2. 再利用できる Graph / UOT / B2 の関数

| 関数 | 場所 | 役割 |
|---|---|---|
| `build_fixed_context(fixed, moving)` | `analysis/contextual-uot-cost-ablation-20261008/run_cost_ablation.py` | 地域 NCC、fingerprint、30 候補、graph support、cost の成分 |
| `method_runs(context)` | `analysis/graph-matching-adoption-separation-20261014/experiment_lib.py` | B0（Position+Graph）、B1（相対 cost の Graph-UOT）、B3（Full）、Position の UOT |
| `result_frame(...)` | 同上 | 核ごとの表（状態、特徴量、正解があれば正誤） |
| `apply_separated_adoption(...)` と `LogisticCalibrator` | 同上 | B2。モデルと閾値は `diagnostic_summary.json` に保存された係数から作る（**再学習しない**） |
| `graph_support_scores`、`mutual_knn_graph` | `analysis/contextual-graph-uot-20261008/graph_consistency.py` | （`build_fixed_context` の中で呼ばれる） |
| `_run_plan`、`_extract` | 同 `contextual_uot.py` | Sinkhorn（POT 0.9.7、`/mnt/.../analysis/cpd-uot-comparison-20260909/vendor`）と、行ごとの診断 |

- 読み込みは `src/registration_research/legacy.py` で行う。各研究ディレクトリの `sys.path` の慣習をそのまま使い、import 先が想定どおりのファイルであることを確かめる。
- 注意：POT は、リポジトリの外（ホーム下の analysis/vendor）にある。

## 3. UOT conservative の境界（`analysis/uot-adoption-validation-20260911/validation_lib.py` の `run_uot`）
1. 輸送計画：sqeuclidean / (2·10²)、ε = 0.1、KL 0.5、translation-invariant Sinkhorn
2. 行の質量の比 `confidence = row_mass / source_mass`。0.25 未満の行は捨てる。
3. アンカー = 重心（`plan @ fixed / row_mass`）。変位に `min(confidence, 1)` を掛ける。
4. 変形場 = `fit_linear_rbf(anchors, displacements, moving, grid)`（linear、smoothing 10、近傍 40）
5. `common_audit(...)`：硬い安全条件、Jacobian の分位、点の改善の確認。通らなければ変位 0。

- 1〜3（対応）と 4（変形場）は関数として分かれている。そのため、**外部のアンカーを `fit_linear_rbf` に渡し、同じ `common_audit` で検査できる**（Phase 2 の B′ / C / E）。
- D は `run_uot` を変更せずに呼ぶ。

## 4. RBF に外部の対応アンカーを渡せるか
- `fit_linear_rbf(anchors, anchor_displacements, moving, grid_x, grid_y, smoothing, neighbors)`：**渡せる**。任意のアンカーを受け取る。
- 既存のアプリ（app.py STEP の fine warp）の `src/pointset_registration.py::matched_nuclei_rbf_fine_warp`：アンカーを渡せない。内部で `mutual_nearest_pairs` を使っている。そのため、方式 B ではそのまま使う。外部のアンカーと比べるときは、`mutual_nearest_pairs` と `fit_linear_rbf` を組み合わせて B′ とする。

## 5. 入力と出力の座標系

| 方式 | 入力 | 出力 |
|---|---|---|
| registration_core（C1 / C2） | 各モダリティの frame-local world（µm） | `fixed = M @ moving + t`（moving µm から fixed µm へ） |
| Graph / UOT / B2 | 2 つの点群が**同じ座標系にあること**（Affine を当てた後の moving と fixed）。単位は µm（cost の尺度 10 µm、graph の許容 5 µm で固定） | moving の各核に対する fixed の index と状態 |
| UOT conservative / `fit_linear_rbf` | Affine を当てた後の moving と fixed（µm）、格子 | 格子上の変位（moving から fixed）と変形後の moving |
| `matched_nuclei_rbf_fine_warp` | 同上 | 独自の格子（`FineWarpResult.bounds` と `grid_spacing`）。比較するときは `sample_displacement_field` で共通の格子に写す |

- このアダプタでは、候補ごとに Moving を `T_final` で Fixed の µm 座標に写してから、Graph / UOT に渡す。

## 6. 既存のテストと、再現できる基準の結果

| 対象 | テスト | 基準の結果 |
|---|---|---|
| 本体 | `pytest tests`（270 成功、1 スキップ。GPU のテスト 2 件は `--gres` が必要） | 11-11 の `test_results.md` |
| Graph / B2 | `analysis/graph-matching-adoption-separation-20261014/tests`（10 件） | `real_exploratory_per_nucleus.csv`（実データの B0 / B1 / B3 / B2） |
| contextual | `analysis/contextual-graph-uot-20261008/tests` | `correspondence_results.csv` |
| UOT conservative | `verify_outputs.py` | `synthetic_metrics.csv`（50 シナリオ） |
| C1 / C2 | 11-11 の `runs/*/result.json` | 6_2 と 8_6 の最終の変換が 11-10 と差 0 |

- 今回の基準の確認（tests_rcm）では、次の 3 点を調べる。
  1. 実データの B0 / B1 / B3 / B2 を再実行して、10-14 の値と一致すること。
  2. SYN-FIELD の D を再実行して、09-11 の値と一致すること。
  3. 再利用したファイルの sha256 が変わっていないこと。

## 7. 新しく書くもの（`src/registration_research/`）

| ファイル | 内容 |
|---|---|
| `legacy.py` | 凍結済みのモジュールを読み込む。B2 の凍結済みモデル |
| `candidate_adapter.py` | Top-k 候補 |
| `cell_correspondence.py` | 重なりの抽出、B0 / B1 / B3 / B2 の実行、decoy、候補ごとの要約、正解との比較 |
| `candidate_reranking.py` | 事前に決めた再順位付けと、支持の判定 |
| `correspondence_anchors.py` | アンカーの作成（相互最近傍、B2、B2 に重みを付けたもの） |
| `deformation_experiment.py` | A / B / B′ / C / D / E の変形場と、共通の安全検査 |
| `evaluation.py` | EPE、TRE、Jacobian などの集計 |

- アルゴリズムを複製しない。新しいのは、組み合わせ方と、候補を選ぶ規則と、評価だけ。
