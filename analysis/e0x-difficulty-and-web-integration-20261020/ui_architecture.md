# Registration Explorer：UI設計（2026-10-20）

既存のStreamlitアプリ（`app.py`）に、研究モード専用のページを1つ追加しました。新しいアプリは作っていません。

## 1. 有効化とProductionとの分離

- 環境変数 `CELLREG_RESEARCH_MODE=1` を設定したときだけ、サイドバーの「Workflow」に **Registration Explorer (Research)** が出ます。既定では無効で、既存のWorkflow A〜Dの選択肢と動作は変わりません。
- `app.py` の変更は、`main()` で選択肢を1つ足し、選ばれたときにページ関数を呼ぶ部分だけです（import 1行、約10行）。
- ページはWorkflow C/Dのsession stateやartifactに触りません。**Workflow Dへの受け渡しボタンはありません。**
- 研究アルゴリズム（A2、ICP、E0x）は、新規実行のときだけSlurmジョブの中で呼びます。すべて無変更です。

## 2. モジュール構成（`src/registration_explorer/`）

| ファイル | 役割 |
|---|---|
| `__init__.py` | feature flag（`CELLREG_RESEARCH_MODE`）、ページ名 |
| `catalog.py` | サンプル登録（`viewer_sample_manifest.json`）を読み、modalityごとのframe（spacing、shape、範囲、原点UNKNOWN）と核座標を扱う。画像の利用可否と理由を判定 |
| `results.py` | Result Adapter：保存済みの研究結果を共通の `ResultRecord` に変換。safety表示（UNIQUE / Accepted、AMBIGUOUS、LOW_PEAK、FAIL、IMAGE_UNUSABLE、POINT_ONLY_FALLBACK、UNVERIFIED）を決める |
| `imaging.py` | Optional Image Loader：OME-TIFFのpyramid、level選択、**タイル単位のROI読み込み**、表示用の変換、**逆写像によるWarp Preview** |
| `scene.py` | 表示frame（Fixed側）に画像・点群・FOV・組織輪郭・候補領域・残留ベクトルをまとめる。UIでの再計算による診断 |
| `viewer.py` | 対話型viewer（Canvas2DのHTMLを自前で生成し、外部スクリプトなし）。Zoom・Pan・透明度・チェッカーボード・layerのON/OFFはブラウザ側で処理し、Streamlitの再実行を起こさない |
| `jobs.py` | Slurm上の研究ジョブ：投入・状態・ログ・キャンセル・結果キャッシュ |
| `manual.py` | 手動調整（2026-10-22追加）：中心まわりの回転・拡大縮小と平行移動の合成、MANUAL_UNVERIFIEDとしての保存と一覧 |
| `page.py` | Streamlitのページ本体 |

その他：`scripts/run_registration_explorer_job.py`（計算ノードで動く研究用Runner）、`slurm/registration_explorer_cpu.sbatch`（CPU 4コア、6 GB、45分）、`tests/test_registration_explorer.py`。

## 3. データの流れ

```text
viewer_sample_manifest.json ──► catalog.SampleInfo（frames + nuclei + 画像パス、UNKNOWNはUNKNOWNのまま）
                                     │
result_sources ──► results.load_records ──► ResultRecord（moving→fixedの変換、verdict、accepted、
   oir-20261019 artifacts + mode_comparison.csv + run cache        status、fallback、保存済み指標、候補）
   e0x-ablation-20261020 cache（候補ごとのICPと評価）
   runs/registration-explorer/*/result.json（UIから実行した結果）
   stored-transform（過去analysisの変換。verdictなし → UNVERIFIED）
                                     │
page: Sample / Fixed・Moving / 方式 / 結果 / 候補 ──► scene.build_scene ──► viewer.render（iframe）
                                     └► Warp Preview（imaging.warp_preview：逆写像）
```

**再計算はしません。** 保存済みの指標をそのまま表示します。UIで計算する診断（NN、≤r、mirror null）は母集団を明記した参考値で、採択には使いません。

## 4. 座標の扱い

- **表示frame = 選んだFixed側のframe-local world**です。BeforeとAfterは同じframeで描きます。Beforeは「変換なし（両frameの(0,0)を重ねただけ、原点はUNKNOWN）」とバナーで示します。
- 変換は moving world → fixed world（点の順写像）です。結果が逆方向で計算されていれば逆行列を使い、「逆方向で計算」と表示します。計算方向と出力frameは別に扱います。
- 画像の配置：level-k画像の pixel (u, v) → world = corner + (u, v)·um_k。cornerは −0.5 × level0 spacingです（pixel中心が整数index、核座標と同じ規約）。Moving画像は、これにさらに変換をかけてcanvasに描きます。
- **Warp Preview**：target gridの各pixel中心を逆写像でsource画像のindexへ戻し、bilinearまたはnearestでsampleします。範囲外は0で、無効として扱います（外挿なし）。補間・範囲・方向・source levelを記録して表示します。viewer上で、逆写像した画像と順写像した核が重なるかを確認できます。
- pixel spacingがUNKNOWNのframe（過去のPoint-onlyペアのH&E）は、画像も物理的なFOVもUNKNOWNとして扱い、E0x（物理単位のscale格子が前提）は実行できないようにしています。

## 5. 大きな画像への対応

| 方針 | 実装 |
|---|---|
| level 0全体を読まない | `read_level_cached` は4096²画素を超えるlevelの全体読み込みを拒否する。概観はpyramidのlevel 3〜4（約0.8〜3 µm/px） |
| ROI | `read_region` が、ROIに触れる512 pxのタイルだけを復号する（level 0も可）。最大約1600²画素 |
| キャッシュ | levelの配列とJPEGはプロセス内のLRUで保持。結果一覧は20秒のTTL |
| 核マーカー | Canvas2Dで画面内の点だけを描く（4,800点程度で問題なし） |

## 6. 安全性の表示

| 状態 | 表示 |
|---|---|
| accepted=true | 緑「UNIQUE / Accepted（研究artifactで採択）」。採択規則（事前固定）を併記 |
| AMBIGUOUS / LOW_PEAK / FAIL / UNVERIFIED | 黄・赤「PREVIEW ONLY — Workflow Dへ渡しません」 |
| 候補を選んだとき | 「候補プレビュー — 候補の選択は採択ではありません」 |
| 画像を使う手法で画像が使えない | 「IMAGE_UNUSABLE → Point-only（E0x）の結果を表示」 |
| 保存済みのPOINT_ONLY_FALLBACK | fallback元を表示し、fallbackの変換との切り替えを可能にする |

UIには閾値を変える入力欄がありません。

## 7. 新規実行

- 入力検証：核ファイルがあること。E0xはspacingと撮像範囲が既知であること。
- 結果キャッシュ：同じ設定（sample、方式、Fixed）の完了済み結果があれば、投入ボタンを無効にして既存の結果を案内します。
- 同時実行の上限：explorerジョブは同時に2件まで（`CELLREG_EXPLORER_MAX_ACTIVE`）。投入前に確認のチェックが必要です。
- 状態表示：`st.fragment(run_every=10s)` で、ページ全体を止めずに更新します。
- キャンセル：自分のrunディレクトリに記録したjob IDだけを対象にします。squeueで所有者とジョブ名（`cellreg-explorer`）を確かめてから `scancel` します。
- run directory：`runs/registration-explorer/<run_id>/`（config.json、status.json、slurm_job.json、stdout/stderr、result.json）。状態の管理は、既存のWorkflow Cの `update_status` / `write_json_atomic` を使います。
