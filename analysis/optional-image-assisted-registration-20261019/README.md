# optional-image-assisted-registration-20261019

画像を任意に使う連続切片位置合わせ（P0 Point-only / P1 Single-image / P2 Dual-image）の研究用prototypeと評価です。

- 結論と質問への回答：`REPORT.md`
- 入力監査：`input_audit.md`
- 設計（frame、artifact、fallback、Workflow C/Dとの境界）：`architecture.md`

Production Workflow C/D・UI・A2/ICP・Graph/UOT/B2・safety gateは変更していません。既存のモジュールはimportして呼ぶだけです。

## 構成

| パス | 内容 |
|---|---|
| `oir/frames.py` | `Frame`（pixel ↔ world）、`Affine2D`（forward / inverse / compose）、`TransformArtifact` |
| `oir/features.py` | modalityごとの構造表現（tissue / nuclear / edt）と品質判定（`usable`） |
| `oir/search.py` | 共通の探索エンジン `PoseSearch`（points / image / cross の各層、robust z、UNIQUE / AMBIGUOUS / LOW_PEAK） |
| `oir/pipeline.py` | E0〜E3のランナー、cropped ICP、trust region、評価、採択、fallback |
| `oir/synthetic.py` | 半合成のpartial-overlapケースの生成 |
| `oir/metrics.py` | 点群・画像の評価指標、逆写像によるwarp |
| `run_audit.py` | 入力監査（`sample_manifest.json`、`coordinate_consistency.csv`、監査の図） |
| `run_real.py` | 実サンプルの1設定を実行（`--sample --config`）。特徴量の作成は `--features <sample>`、P0互換性の確認は `--p0compat` |
| `run_synthetic.py` | 半合成の1ケースを5手法で実行（`--case`） |
| `aggregate.py` | `cache/` からCSV、`diagnostic_summary.json`、`artifacts/` を作成 |
| `fig_features.py`、`fig_results.py` | 図とプレビュー |
| `test_oir.py` | 単体テスト（9件） |
| `real.sbatch`、`syn.sbatch`、`p0.sbatch` | Slurmのジョブスクリプト |
| `cache/` | 中間結果（特徴量、各runのJSON、`real_tasks.txt`） |
| `freeze/` | 作業前後のSHA-256、job ID |

## 再実行

すべて計算ノードで実行します（partition `mib-dbia`、環境 `cell-rg`）。

```bash
cd analysis/optional-image-assisted-registration-20261019
PY=/mnt/mib-dbia08_1/users/noa.ganeko/miniconda3/envs/cell-rg/bin/python

# 0. 単体テスト
srun -p mib-dbia -c 4 --mem=16G -t 00:20:00 $PY -B -m pytest -q test_oir.py

# 1. 入力監査と特徴量
srun -p mib-dbia -c 4 --mem=32G -t 00:30:00 $PY -B run_audit.py
for s in 031-6_2 031-8_6; do srun -p mib-dbia -c 4 --mem=32G -t 00:20:00 $PY -B run_real.py --features $s; done
srun -p mib-dbia -c 4 --mem=32G -t 00:20:00 $PY -B fig_features.py

# 2. 実サンプル（21設定×2、cache/real_tasks.txt）、P0互換性、半合成（32件）
sbatch --array=0-41 real.sbatch
sbatch p0.sbatch
sbatch --array=0-31 syn.sbatch

# 3. 集計と図（保存済みのcache/から、すべての表と図を再現できる）
srun -p mib-dbia -c 4 --mem=32G -t 00:30:00 $PY -B aggregate.py
srun -p mib-dbia -c 4 --mem=48G -t 00:40:00 $PY -B fig_results.py
```

所要時間の目安（4 CPU）：実サンプル1設定あたり1.5〜16分、半合成1ケースあたり42〜50分。

## 入力データ

- `projects/031-6_2/`、`projects/031-8_6/`（OME-TIFF 2つ、GeoJSON 3つずつ）
- H&E側の核：StarDistの検出結果（analysis `he-stardist-seqif-registration-20261019`）

詳細は `input_audit.md` を参照してください。

## 注意

- `previews/` は研究用の確認画像です。Production Workflow Dの出力ではありません。
- 031-8_6のH&Eは別の切片（031-7）です。結果を解釈するときは、この点を併記してください。
