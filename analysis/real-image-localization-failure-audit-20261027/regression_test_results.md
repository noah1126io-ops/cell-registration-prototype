# 回帰テストの結果 — 2026-10-27

## pytest

`CUDA_VISIBLE_DEVICES=0 python -B -m pytest -q tests`（cell-rg環境）：**266 passed、1 skipped**（既存262件はすべて通過、新規4件）。

- GPU parityの2件は、`CUDA_VISIBLE_DEVICES` が未設定だと失敗します（ログインノードで直接実行した場合）。コードとは関係ありません（2026-10-22から同じ）。

新規テスト：

| テスト | 内容 |
|---|---|
| `test_reference_fit_rejects_outlier_and_reports_heldout` | 9組中1組の誤クリックを除外する。倍率・回転を復元する。leave-one-out / held-out < 10 µm |
| `test_reference_compare_flags_180_degree_flip` | 180°反転した候補を、回転差180°・距離 > 500 µm と判定する |
| `test_occupancy_ignores_nuclei_and_maps_background_to_zero` | 明るい核があっても、組織 ≈ 1、間隙・スライド背景 ≈ 0 になる |
| `test_config_key_unchanged_for_old_runs_and_split_by_image_feature` | 10-27以前のrunのキャッシュキーは不変。画像特徴量が違えば別のキーになる |

## 特徴マップの互換性（`run_engine_checks.py --task feature_parity`、`cache/feature_parity.json`）

- 4画像すべてで、従来の `tissue / nuclear / edt / tissue_bin / envelope` は変更前と**差 0** でした。
- エンジンの `occ` と解析用の `variants.py` の差は0、`cleft` の差は1.2e-7（float32の丸め）でした。

## 実データの回帰（`before_after_comparison.csv`）

| サンプル | 変更前（10-22、tissue+nuclear） | 変更後（occ+cleft） |
|---|---|---|
| 031-6_2 | ACCEPTED。画像UNIQUE 0.504、核UNIQUE 0.477、≤3 µm 0.7745 | **ACCEPTED。最終変換は完全に同じ（M・t の差 0）**。画像UNIQUE 0.480、≤3 µm 0.7745 |
| 031-8_6 | NOT_ACCEPTED。画像AMBIGUOUS 0.006（1位は参照から1245 µm） | NOT_ACCEPTED。**画像UNIQUE 0.303（1位は参照から55 µm）**。C3（核の近接率）で停止 |

## 合成既知変換（`synthetic_validation.csv`、粗探索の段階）

| | 従来 | occ+cleft |
|---|---|---|
| Test 1：同一画像（6条件） | 6/6 | 6/6 |
| Test 2：部分重なり（4条件） | 2/4 | 4/4 |
| Test X：031-6_2 モダリティ間（3条件） | 3/3 | 3/3 |
| Test X：031-8_6 モダリティ間（3条件、手動参照に対して40 µm以内） | 0/3（1.2〜1.4 mm） | 0/3（52〜56 µm、3条件とも同じ位置） |

## 既定動作への影響

| 対象 | 変化 |
|---|---|
| Production（Workflow A〜D、研究モードOFF） | なし（`registration_core` を使わない） |
| 研究エンジンの既定 `WorkflowConfig.image_feature` | `tissue+nuclear` → `occ+cleft`。`CORE_VERSION` を 1.1 に上げた。過去のartifactには使った値が `config.image_feature` として残っている |
| Explorerの新規実行 | 画像特徴量を選べる（既定はocc+cleft）。キャッシュキーに含める（過去のrunのキーは不変） |
| 採択規則・Safety Gate・Workflow D | 変更なし |
| 研究ストア | スキーマは変更なし。参照の指定は `note` 注釈（追記のみ）、対応点のメタデータは `payload` / `detail` に保存 |
