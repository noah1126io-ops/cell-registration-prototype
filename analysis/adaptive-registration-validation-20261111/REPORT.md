# 異なる切片の Registration の外部検証と、Adaptive Registration の設計（2026-11-11）

- 対象：branch `feature/gpu-slurm`、HEAD `f66ecbe` と未コミットの差分（registration_core 1.2）。
- エンジン、既定値、閾値は変更していない（`baseline_manifest.json`）。
- 計算はすべて Slurm（mib-dbia、1 ジョブ 4 CPU）で実行した。

## 結論（要約）
1. **031-8_6 に似た、未使用の実画像ペアは 0 件。** 既知の研究データ領域のうち、次を確認した。
   - 自分のプロジェクト領域と、`groups`、`nagasaki` の浅い範囲（他のユーザーのホームは対象外）。
   - 画像のある HE / IF の組は、開発に使った 031-6_2 と 031-8_6 の 2 組だけだった。
   - したがって**独立した外部検証はできていない。**
2. 代わりに、2 組から**派生ケース 16 件**を作った（**独立ではないストレステスト**として明記する）。
   - 正例 8 件：部分 FOV、小視野、部分重なり
   - 負例 6 件：鏡像、真の領域を除いた HE
   - 推定負例 2 件：異なるサンプルどうしの組
   - 各ケースで方式 A〜D と、2 種類の検証ゲートを評価した（56 ジョブ、欠損なし）。
3. **occ+cleft（方式 C）は、8_6 系の正例 4 件すべてで top-1 が正しい領域だった**（15〜26 µm）。
   - 核探索（A）と旧特徴（B）は、4 件すべてで約 870〜1250 µm 離れた位置を選んだ。
   - 6_2 系では、A、B、C ともほぼ正しかった。ただし、IF を半分にしたケースで A だけが、誤った位置を UNIQUE（margin 0.88）で選んだ。
4. **本番の判定（D）では、誤採択は 0 件**（負例 8 件中 0 件）。正例は 8 件中 6 件を採択した。
   - 採択できなかった 2 件（偽陰性）：8_6 の小視野（C1 で AMBIGUOUS）と部分重なり（C3 の象限チェックで 2/4）。
5. **重大な発見：TISSUE_LEVEL の誤採択が起きる。**
   - 6_2 由来の鏡像 IF（物理的に正しい姿勢が存在しない）で、画像探索が UNIQUE（margin 0.215）になり、象限一致も 3/4 で合格した。
   - 本番では、6_2 はファイル名の推測で同一切片のゲートになるため、止まった。**来歴どおり「異なる切片」として扱うと、TISSUE_LEVEL になる。**
   - **4/4 や 3/4 の一致は、正解の証明ではない。**
6. **6_2 の切片関係が食い違っている。**
   - manifest のユーザー訂正（2026-10-20）は DIFFERENT。engine 1.2 のファイル名の推測は SAME。
   - 6_2 の ACCEPTED は、ファイル名の推測に依存している。
   - 11-10 の報告で 6_2 を「同一切片」と書いたのは、この食い違いを見落とした誤りだった。
7. 設計を作成した：Data Profiler、Strategy Planner、設定 schema（研究モードの試作とテスト 15 件）、5 経路の検証、v1.0 の優先順位。
   - 最優先の次の作業：**切片の関係を来歴から渡し、UNKNOWN / CONFLICT では細胞レベルの自動採択を保留する**（本番の動作が変わるので、承認が必要）。

## Phase A：エンジンの固定（`baseline_manifest.json`）
- 記録したもの：
  - git HEAD、未コミット差分の sha256（`cache/baseline_tracked.diff`）、untracked の対象コード各ファイルの sha256
  - `CORE_VERSION` "registration_core 1.2 (2026-11-10)"
  - WorkflowConfig と SearchConfig の既定値、特徴抽出の既定値、ICP と trust region の設定
  - C3 の規則：`ACCEPT` と `QUADRANT_RULE`（3/4、30 µm、2°、±4°、±100 µm）
  - 入力の画像 4 枚と核ファイル 4 個の sha256
- 「正しい領域」の判定は、**実行前に** probe 距離 ≤100 µm と決めた（`common.CORRECT_UM`、`cases.json`）。
- 再現性：
  - 特徴 map は 10-27 のキャッシュと完全に一致した。
  - 8_6 と 6_2 の元のペアの最終変換は、11-10 と**差 0.0**。
- 再確認した報告値（8_6）：
  - TISSUE_LEVEL、`accepted=False`
  - 4 象限のずれは 9 / 3 / 10 / 4 µm、回転の差は ≤1°
  - ランドマーク誤差の中央値 21.0 µm（最大 46.7 µm）

## Phase B：データの調査（`dataset_manifest.csv`）

| ペア | 画像 | 切片の関係 | 来歴の信頼度 | 開発に使ったか |
|---|---|---|---|---|
| 031-8_6 | 両方 | DIFFERENT（HE 031-7 / IF 031-8） | 中 | 使った（occ+cleft と象限の規則を作った） |
| 031-6_2 | 両方 | **DIFFERENT（ユーザー訂正）と SAME（ファイル名の推測）が食い違う** | 低 | 使った |
| serial_section_pair_20260904 | HE のクロップ画像のみ（IF 画像なし）。HE の spacing は不明 | UNKNOWN | 低 | 使った（点群の研究） |

- 組織の種類はメタデータにない。染色（Dystrophin1 / Desmoplakin）から横紋筋または心筋と推測されるが、確認されていない。
- 不足しているデータの仕様と、追加データが来たときの実行手順：`v1_roadmap_recommendations.md`、`data_profiler_spec.md` の §5。

## Phase C：既存の Registration の比較

**C1：対応領域の探索**（`localization_comparison.csv`、図 `figures/fig1_localization_top1.png`）

| ケース | A 核 | B tissue+nuclear | C occ+cleft |
|---|---|---|---|
| P8_6_orig | AMB、1255 µm | AMB、1202 µm | **UNIQUE、23 µm** |
| P8_6_IFhalfL | AMB、1232 | AMB、932 | **UNIQUE、21** |
| P8_6_IF500 | AMB、1022 | AMB、866 | AMB、15（top-1 は正しい） |
| P8_6_HEpartial60 | AMB、1035 | AMB、1202 | **UNIQUE、26** |
| P6_2_orig | UNIQUE、2 | UNIQUE、2 | UNIQUE、2 |
| P6_2_IFhalfL | **UNIQUE だが 859 µm（正解は 2 位）** | UNIQUE、4 | UNIQUE、3 |
| P6_2_IF500 | UNIQUE、2 | UNIQUE、2 | UNIQUE、4 |
| P6_2_HEpartial60 | UNIQUE、2 | UNIQUE、2 | UNIQUE、5 |

- top-k の発見率：C は正例 8 件すべてで top-1 が正しい。A は 8 件中 3 件が top-1、1 件が top-2 で、8_6 系の 4 件は top-6 にも入らなかった。B は 8 件中 4 件（6_2 系だけ）。
- 負例の C1 では、UNIQUE になったのが 2 件あった。C の N6_2_IFmirror と、B の N8_6_IFmirror。それ以外は AMBIGUOUS か LOW_PEAK。

**C2：精密化と C3：検証**（`external_validation_results.csv`、`validation_rows.csv`）
- ICP の移動量（中央値の probe 距離）は、6_2 系で 3.5〜8.5 µm、8_6 系で 5〜13 µm だった。
  - 8_6 の部分重なりのケースは 34 µm 動き、trust region を外れた（`ICP_LEFT_TRUST_REGION`）。
  - 隣接切片のゲートでは ICP は診断用で、探索で得た姿勢を維持する（8_6_orig は 12.7 µm で、11-10 と同じ）。
- 核の近接率：
  - 8_6 系は、正しい位置でも 0.037〜0.052 で、null は 0.022〜0.036。区別できない。
  - 6_2 系の正しい位置は 0.77〜0.87、null は 0.03〜0.08。

**判定の混同行列**（`validation_confusion_matrix.csv`）

| 方式 / ゲート | 正例：TP / FN / 誤った領域で採択 | 負例：FP / TN | 推定負例：FP / TN |
|---|---|---|---|
| D 本番 | 6 / 2 / 0 | 0 / 6 | 0 / 2 |
| D 同一切片のゲートで統一 | 4 / 4 / 0 | 0 / 6 | 0 / 2 |
| D 隣接切片のゲートで統一 | 4 / 4 / 0 | **1** / 5 | 0 / 2 |
| B（旧特徴）本番 | 4 / 4 / 0 | 0 / 6 | 0 / 2 |

**計算時間**（D、4 CPU、16 件。図 `figures/fig3_runtime.png`）

| 段階 | 中央値（最小〜最大） |
|---|---|
| C1 画像探索 | 821 s（235〜907） |
| C1 核探索 | 439 s（134〜471） |
| C2/C3 同一切片のゲート | 3 s |
| C2/C3 隣接切片のゲート（象限 1 回を含む） | 69 s（14〜112） |
| 全候補の象限の監査（研究用の追加） | 253 s |
| 合計 | 1572 s |

- CPU 時間は wall 時間とほぼ同じ（numpy は実質 1 スレッド）。
- 最大メモリ（RSS）は 415〜610 MB。
- GPU：engine は CPU だけで動くため、該当しない。
- Slurm の待ち時間：中央値 1694 s。大半は、自分で設定した同時実行の上限（%8）による待ちで、共有キューの混雑ではない。

## Phase D：誤採択の検証
- 負例は、すべて新しく作った（8_6 の過去の誤った候補だけには頼っていない）。
  - 鏡像（探索は proper のみなので、正しい姿勢が存在しない）
  - 真の領域を除いた HE の帯（同じ組織だが、対応しない領域）
  - それと小視野 IF の組み合わせ（全面で重なりうるが、対応しない）
  - 異なるサンプルどうしの組（XY 位置が異なる。対応がないことは未確認）
- 結果と象限チェックの詳しい監査：`quadrant_validation_audit.md`
  - 誤った候補 129 個のうち、象限 ≥3/4 になったのは 1 個（鏡像）。
  - 部分重なりと小視野では、正しい候補が棄却された。
  - 象限ごとの全域探索（より独立な方式）は、誤りを止めたが、8_6 の正しい候補も 0/4 で通さなかった。

## Phase E：失敗の分類
`failure_taxonomy.md` を参照。

## Phase F：設計
- `data_profiler_spec.md`
- `adaptive_registration_architecture.md`（図 `figures/fig7_adaptive_architecture.svg`）
- `adaptive_strategy_rules.md`
- `v1_roadmap_recommendations.md`
- 試作：`data_profiler.py`、`adaptive_strategy.py`、`adaptive_strategy_schema.json`。全ケースへの適用結果は `profiles.csv` と `planner_output.csv`。
- いずれも研究モード専用で、本番には組み込んでいない。

## §14 の質問への回答
1. **031-8_6 に似た未使用データの数**：0 件。画像付きで使えるのは、開発に使った 2 組だけだった。
2. **共通する特徴**：比べられる独立データがない。8_6 の特徴は次のとおり。
   - 組織がほぼ全面にある（外縁の情報 0.15）。
   - 間隙がはっきりしている（cleft 0.28）。
   - IF の非核チャンネルが核と相関しない（−0.20）。
   - 反復が少ない（0.065）。
   - 核は対応しない（隣接切片）。
3. **occ+cleft は未使用データで正しい領域を見つけたか**：未使用データでは**検証していない**。派生ケースでは、正例 8 件すべてで top-1 が正しかった。ただし、小視野の 1 件は AMBIGUOUS だった。
4. **核点群探索と比べて有利だった条件**：
   - 異なる切片（8_6 系）では、C だけが成功した。
   - 同一切片の可能性がある 6_2 では、ほぼ同等だった。視野が半分のケースでは、C が A より安定していた（A は誤った位置が UNIQUE になった）。
5. **失敗の原因**：
   - 小視野で一意に決まらない（C1）。
   - 部分重なりで象限が視野外になり、誤って棄却される（C3）。
   - 鏡像のような、大きなスケールの構造に引っぱられて誤った候補が UNIQUE になる（C の弱点）。
6. **象限チェックは正しい候補を通し、誤った候補を止めたか**：
   - 全視野で重なる正しい候補は通した（4/4、3/4）。
   - 部分重なりと小視野では、正しい候補を止めた。
   - 誤った候補 129 個のうち 128 個を止めたが、鏡像の 1 個を通した。
7. **TISSUE_LEVEL の誤採択の危険**：
   - 主探索と同じ特徴を使う局所検査なので、視野全体に共通する大きな構造が候補と偶然一致すると、合格してしまう（実例 1 件）。
   - 切片の関係が来歴どおり DIFFERENT と設定されたサンプルで、起きうる。
   - 自動で採択しない方針は維持すべき。
8. **戦略選択に有用な情報**：
   - 切片の関係の**来歴**（最も重要。ただし今は使われていない）
   - IF の非核チャンネルの妥当性（occ と核の相関）
   - FOV の比、外縁の情報、cleft の強さ
   - 核の密度は、今回は予測力がなかった。
9. **ルールベースでどこまで実現できるか**：
   - 実現できる：入力の有無と座標による手段の決定、切片の関係によるゲートの選択、危険の flag、自動採択を保留する条件。
   - 実現できない：「どちらの手段が当たるか」の事前予測。今回も、6_2 の視野半分のケースで A が外れることは予測できなかった。そのため、**全手段を並行して実行し、事後に一致を見る**方式が必要。
10. **v1.0 で必須の機能と、後回しにできる機能**：`v1_roadmap_recommendations.md`。
    - 必須（A1〜A9）：来歴からの切片の関係、Profiler、安全な戦略選択、判定理由の説明、手動補助、実行履歴、互換性のテスト、IF チャンネルの設定化。
    - 後回し：機械学習、閾値の最適化、非剛体の自動制御。
11. **構造上の問題**：`adaptive_registration_architecture.md` の最後の節。主なもの：
    - 切片の関係がファイル名だけで決まる。
    - IF チャンネルの意味がハードコードされている。
    - 特徴抽出に差し替え口がない。
    - 象限チェックが 2 値。
12. **次に最優先で実装すべきこと**：A1 と A8。切片の関係を来歴から渡し、UNKNOWN / CONFLICT では細胞レベルの自動採択を保留し、その回帰テストを追加する。
    - 本番の動作が変わる（6_2 は ACCEPTED から保留になる）ので、**6_2 の来歴の確認と、ユーザーの承認を先に**行う。

## §15 完了条件への回答
- **組織画像を優先すべきスライド**：異なる切片で、IF に核と相関しない組織チャンネルがあり、間隙構造がはっきりしているもの（8_6 型）。
- **核点群が有効なスライド**：同一切片が来歴で確定していて、核の密度が十分なもの。核ゲートを必ず使う。
- **複数の手段を併用すべき条件**：常に併用する。両方が一致したときだけ、強い独立の証拠として扱う。
- **自動採択を保留すべき条件**：切片の関係が UNKNOWN または CONFLICT、TISSUE_LEVEL、FOV の比 ≥10、部分重なり、手段どうしの CONFLICT、注釈を使った実行（ASSISTED）。

## 制約の遵守
- 守ったこと：
  - 元画像と研究 DB は読むだけで、書き込んでいない。
  - 本番のコード、既定値、Safety Gate は変更していない。
  - 過去の成果物は上書きしていない。
  - 重い処理は Slurm で実行した。開発サーバーとトンネルには触れていない。
- 例外：
  - ゲートウェイで、軽い処理（ケース定義の作成、プロファイルの確認、pytest 15 件で約 2 秒）だけ実行した。
  - 依頼文では「commit / push しない」とされていたが、作業中にユーザーから「GitHub も毎回更新」との指示があった。そのため、作業の区切りで commit と push を行う。`.gitignore` に、キャッシュ、バイナリ、5 MB 以上のファイルの除外を追加した。

## ファイル一覧

| ファイル | 内容 |
|---|---|
| `baseline_manifest.json` | エンジン固定時の記録 |
| `dataset_manifest.csv` | 実データ 3 組と派生ケース 16 件 |
| `cases.json` | 派生ケースの定義と参照姿勢（実行前に作成） |
| `external_validation_results.csv` | ケース × 方式（D と B） × ゲートの判定、時間、メモリ、Slurm |
| `localization_comparison.csv` | A、B、C の C1 比較 |
| `validation_confusion_matrix.csv` | 方式とゲートごとの混同行列 |
| `validation_rows.csv` | 全ゲートの判定の詳細 |
| `quadrant_audit_all_candidates.csv`、`quadrant_global_audit.csv` | 象限チェックの監査 |
| `profiles.csv` / `.json`、`planner_output.csv` | Profiler とプランナーの出力 |
| `failure_taxonomy.md`、`quadrant_validation_audit.md` | 分類と監査 |
| `data_profiler_spec.md`、`adaptive_registration_architecture.md`、`adaptive_strategy_rules.md`、`adaptive_strategy_schema.json`、`v1_roadmap_recommendations.md` | 設計 |
| `test_results.md` | テスト |
| `figures/fig1`〜`fig4`、`fig7` | 図（図 5 と図 6 は、図 1 と図 2 に統合した） |
| `prep.py`、`benchmark_runner.py`、`quadrant_global.py`、`collect.py`、`data_profiler.py`、`adaptive_strategy.py`、`*.sbatch` | 再現用のコード |
