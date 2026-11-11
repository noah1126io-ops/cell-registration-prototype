# テスト結果（2026-11-11）

本番コード（`src/`, `app.py`）は今回変更していない。変更点は `.gitignore` の追記と、本ディレクトリの解析コードだけ。

## 既存テストスイート（`pytest tests`、Slurm）

| 実行 | ノード | 結果 |
|---|---|---|
| job 899839 | mib-dbia-g01（GPU ノード、`--gres` なし） | 268 成功、1 スキップ、**2 失敗** |
| job 899853 | mib-dbia-u01（`--gres` なし） | 268 成功、1 スキップ、**2 失敗**（同じ 2 件） |
| srun 899865（GPU パリティの 2 ファイルのみ） | mib-dbia-g01、`--gres=gpu:1` | **3 成功**（スキップされていた 1 件も成功） |

- 失敗した 2 件：
  - `test_density_flow_gpu_parity.py::test_density_flow_cpu_cuda_float64_synthetic_parity`
  - `test_joint_flow_gpu_parity.py::test_joint_flow_cpu_cuda_float64_stage_and_final_parity`
- 原因は `assert provenance["cuda_visible_devices"] is not None`。
  - GPU は検出されるが、Slurm で GPU を割り当てていないため、環境変数 `CUDA_VISIBLE_DEVICES` が設定されない。
  - **環境に依存する問題で、コードの退行ではない**。GPU を割り当てると成功する。
- 合計すると、11-10 と同じ 270 成功に相当する（268 + GPU 割り当てで成功した 2 件）。
- 記録：
  - `logs/pytest_899839_on_gpu_node_without_gres.out`
  - `logs/pytest_899853.out`
  - `logs/pytest_gpu_parity_with_gres.out`
- 改善案：GPU テストのスキップ条件に「`CUDA_VISIBLE_DEVICES` が未設定」を加える（今回は変更していない）。

## この研究のテスト（`test_adaptive_validation.py`）
- 15 件すべて成功（Slurm と、ゲートウェイでの軽い実行の両方）。
- 内容：
  - 派生ケースへの参照姿勢の引き継ぎ（HE の切り出し、IF の切り出し、IF の鏡像）
  - probe 距離
  - 切片関係の根拠：UNKNOWN を SAME にしないこと、6_2 型の CONFLICT の検出、8_6 型の確定
  - 反復性の指標（周期的な構造と単一の構造の区別）
  - プランナー：全手段を残すこと、TISSUE_LEVEL を自動採択しないこと、UNKNOWN で細胞レベルの自動採択をしないこと、SAME で核ゲートを使うこと、DIFFERENT で組織ゲートを使うこと、構造が弱いとき主な手段が核になること、小視野の flag、安全違反の検出

## エンジン固定と再現性の確認
- 特徴 map は今回、engine の `extract` で抽出し直した。10-27 のキャッシュとの最大の差は **0.0**（`cache/feature_parity_vs_20261027.json`）。
- `P8_6_orig` と `P6_2_orig` の方式 D の最終変換は、11-10 の engine artifact と**完全に一致した**（差 0.0）。
  - 判定も同じ：8_6 は TISSUE_LEVEL（ランドマーク中央値 21.0 µm、最大 46.7 µm、象限のずれ 9 / 3 / 10 / 4 µm、回転の差 ≤1°）、6_2 は ACCEPTED。
- `baseline_manifest.json` に記録したもの：git HEAD `f66ecbe`、未コミット差分の sha256、対象コードの各ファイルの sha256、設定・規則、入力画像と核の sha256。
