# A2 Affine Global Initialization：独立Synthetic外部検証

- **実施日**：2026-10-15
- **対象**：branch `feature/gpu-slurm` @ `f66ecbe`。analysis専用。
- **凍結対象（無変更）**：A2・ICP・検出器・閾値・Graph/UOT/B2・Production・UI・safety gate。
  - tracked 81ファイル、既存analysis 508ファイルのSHA-256を、実行前後で照合して一致を確認した（`freeze/`）。
- **再現確認**：Real A2を再実行し、前回のAffineと一致した（RMS差 0.0 µm）。B2 matched も 1,324 で一致した。
- **実行環境**：Slurm `mib-dbia`（CPUのみ）。commit / push はしていない。

## 結果の要約

### Generator別の結果

| Generator | Cases | A2 Affine Success | Detector False Accept | True Accept | Downstream Top-1（中央値） | Matched Precision（B2, pooled） |
|---|---:|---:|---:|---:|---:|---:|
| Clustered (G1) | 28 | 20/28 (71%) | **0** / 8 wrong | 18/20 | 0.975（PASSのみ）/ 0.959（全） | 0.986（PASSのみ）/ 0.978（全） |
| Inhomogeneous (G2) | 28 | 18/28 (64%) | **0** / 10 wrong | 18/18 | 0.964 / 0.934 | 0.987 / 0.980 |
| Structured Tissue (G3) | 28 | 21/28 (75%) | **0** / 7 wrong | 20/21 | 0.966 / 0.935 | 0.982 / 0.976 |
| Repeated / Ambiguous (G4) | 28 | 19/28 (68%) | **0** / 9 wrong | 16/19 | 0.981 / 0.969 | 0.989 / 0.968 |
| **計** | **112** | **78/112 (70%; 95% CI 61–77%)** | **0 / 34**（rate 95%上限 10.2%） | **72/78 (92%; CI 84–96%)** | 0.971 / 0.948 | 0.986 / 0.976 |

表の定義：

- **成功**：common physical region上のRMS ≤5 µm、かつchiralityが正しいこと。
  - 評価点：moving画像の41×41格子のうち、真のAffineでfixed FOV内に入る点。単位はµm。
- **下流**：production同等のsubset規則を通した、Separated Matching / Adoption（B2）の結果。
  - 「全」は検出器がFAIL/AMBIGUOUSのcaseも含む研究診断値。運用上、それらのcaseは停止扱い（§4）。

### Controlの結果

- Negative / adversarial control 24件（独立generator）で、PASSによる誤採択は **0件**（A0 0/24、A2 0/24、A3 0/4）。
- 共通核のない control・誤chirality control（A2、12件）のPASSも0件。Wilson 95%上限は24%。

### 前回（開発時generator）との比較

- 前回：A2 42/45（93%）。
- 今回の独立generator：78/112（70%）。
- この差は、今回の計画に**前回含まれていなかった失敗領域**を意図的に入れたことによる（§2）。
  - moving FOVがfixedを包含するcase、fixed FOVが小さいcase、強い変形。
- 前回と同種の条件（transform、密度、共有率・検出漏れ、非対称crop）に限ると **69/72（96%）**。

---

## 1. 実施内容

### 独立generator（`indep_gen.py`、詳細は `generator_description.md`）

- `init_lib.make_pair` のコードやテンプレートは使っていない。前回のseedも使っていない（ASTを使ったtestで確認）。
- 4族を新規実装した。
  - **G1**：Thomas点過程
  - **G2**：log-Gaussian場 × Voronoi段差 × 勾配
  - **G3**：上皮様の帯・腺管様の環・裂隙・無核域
  - **G4**：格子motifの繰り返し。密度の自己相関が格子ベクトル上で0.30〜0.50になる。
- 次の要因はそれぞれ独立に生成した。
  - 共通核率（q_shared）
  - 検出漏れ
  - 局所的な核の消失
  - 重心の測定誤差（0.6 µm）
  - **affine成分を除去した**滑らかな非剛体変形
  - 撮像範囲
- persistent IDはpairに保持するが、アルゴリズムには渡していない。
- 2Dの部分共有モデルである。実組織の再現は主張しない。

### 計画（実行前に固定。`experiment_protocol.md`, `design.py`、hashは `freeze/plan_sha256_before_runs.txt`）

| 区分 | 内容 |
|---|---|
| Main | 4族 × 14条件 × 2 replicate = 112 pair。A0/A2は全件、A3はreplicate 0（56件） |
| Control | 24件（NC1〜NC6） |
| Downstream | 344 job：main × {A0, A2, TRUE（真のAffineを入れた参照）} と、NC1/NC2のA2 |

### 凍結の扱い

- A2・検出器・閾値・Graph/UOT/B2は、前回のモジュールをimportして呼び出しただけで変更していない（`freeze/frozen_components.json`）。
- 検出器の使い方は前回と同一にした。
  - Synthetic：basin_marginのみ。
  - Real：basin_margin＋checkerboard held-out。

### 実行中の修正（すべて記録）

1. **A2を一度も実行する前**：generatorのseed key生成にbugがあった。文字列が4 byteに切り詰められ、`nuclei` と `nuclei-b` が衝突していた。hash方式に修正し、回帰testを追加した。
2. **実行後（集計コードのみ）**：
   - 下流の状態表にcontrol行が混入していたので除外した。
   - 失敗分類（初期化 / refine）を追加した。
   - **アルゴリズム・閾値・計画・成功定義は変更していない。**
3. **事後に追加した解析**（REPORT中では「事後」と明記）：
   - moving側被覆率による層別
   - `diagnose_failures.py`

## 2. Q1. Does A2 generalize?

**条件付きで汎化する。**

前回と同種のストレスに対しては、4族すべてでほぼ完全に汎化した（69/72）。

| 因子 | A2 | A0 | A3（rep0） |
|---|---:|---:|---:|
| Transform（offset 0/100/150、rot 0/3/6°、反転なし、1.30 µm/px＋非対称crop＋中密度） | 31/32 | 8/32 | 7/16 |
| 共通核率（低q 0.30 / 検出漏れ50%） | 14/16 | 0/16 | 7/8 |
| 密度（高密度＋強不均一 0.0042） | 8/8 | 0/8 | 3/4 |
| 基準 | 8/8 | 0/8 | 4/4 |
| 重なり（C09包含 / C10小fixed / C11非対称partial） | 8/24（C11のみ8/8） | 0/24 | 3/12 |
| 変形（6 µm＋局所消失 / 12 µm） | 9/16 | 1/16 | 6/8 |
| 探索範囲外（offset 320 µm） | 0/8 | 0/8 | 0/4 |

- **成功例の精度**：RMSの中央値0.52 µm（95%点 3.2 µm）、translationの中央値0.26 µm。
- **A0**：offset 0の条件（C02）以外では、ほぼ全滅だった（9/112）。
- **A3**：同じ56 caseでA2 40/56に対し、A3 30/56。特にoffset 150・回転6°（0/4）と複合scale（0/4）に弱い。

## 3. Q2. Where does A2 fail?

A2の失敗は34件。事後分類の内訳は次の通り。

| 失敗の型 | 件数 | 条件 |
|---|---:|---|
| **Refine drift**：NCCの初期値は真値付近だったが、ICPが別basinへ移動 | 15 | C09 7、C10 7、C13 1 |
| 初期化：真値付近の仮説なし | 9 | C05 4（範囲外）、G4×C07 2、C09/C10 各1、G2×C04 1 |
| 誤chirality | 4 | C05（範囲外、180°） |
| Imprecise：真のbasin内だが、変形によるbias（RMS 5〜12 µm） | 6 | C13 5、C12 1 |

### (1) 視野の重なり：最大の失敗要因

- **moving側被覆率**（moving FOVのうちfixed FOVに入る面積比、事後指標）で明確に分かれる（[affine_error_vs_overlap.png](affine_error_vs_overlap.png)）。
  - 探索範囲内・強変形を除いた場合：
    - 被覆率 <0.7：**0/16**
    - 被覆率 ≥0.7：**67/69**
- 一方、fixed側の被覆率が0.57しかないC11（movingがfixedより小さい）は8/8で成功した。
- C09/C10の内訳（`cache/diagnose_containment.csv`）：
  - A2のNCC初期値は**正しかった**（scale 1.10、translation誤差1〜3 µm、真のパラメータでのNCC 0.68〜0.84）。
  - 続くproduction ICP（trim 0.8/0.7）が、fixed外にあるmoving点まで最近傍に吸着させた。その結果、scaleが **−22〜−53%** に縮んだ（FOV面積比0.55/0.28にほぼ一致）。
  - このとき、縮んだ解のtrimmed objectiveは真の解より**小さい**（例：7.9 µm 対 9.4 µm）。
  - 前回の partial_overlap_350 と同じ機構である。
- FOV輪郭が原因という仮説は否定された。輪郭内に一様乱数を置いた点群では、NCCは0.06〜0.18にとどまった。

### (2) 探索範囲（±200 µm）

- 起点は点群重心である。そのため、FOV中心のoffsetではなく**重心のずれ**が範囲を決める。
- C05（offset 320）は0/8だった。
- G2×C04（offset 150）では、密度勾配によって重心のずれが257 µmになり、失敗した。

### (3) 強い局所変形（RMS 12 µm）

- ICPは真のbasinに到達した（objectiveは真値から始めた場合と同一）。
- しかし、非affine変形をtrimmed ICPがaffineで吸収するため、5〜12 µmのbiasが残る。2/8のみ成功。
- これは「誤った解」ではなく、成功閾値5 µmに対する精度不足である。

### (4) 繰り返し構造

- G4単独では失敗しなかった。
  - main 19/28。G4の失敗は他族と同じ条件（C05、C09、C10、C13）。
  - NC3（jitter 1.2 µmのほぼ同一の繰り返し）は4/4で正解した。
- **G4 × 低共通核率（C07）だけが0/2**で、格子1周期分（約130 µm）ずれた解を選んだ。
  - 真の解のobjectiveは5.0、選ばれた解は7.5で、真の解の方が良い。それでも真の解の仮説は上位6仮説に入らなかった（初期化の失敗）。
  - 検出器はAMBIGUOUS（inlier/null 2.1〜2.2）を返し、PASSにはしなかった。

### 失敗しなかった条件

- 回転6°、反転の有無、scale 1.30 µm/px、検出漏れ50%、高密度・強不均一。

## 4. Q3. Is the failure detector reliable?

**誤採択（PASS）は観測されなかった。判定は安全側に寄っている。**

### Confusion matrix

[detector_confusion_matrix.png](detector_confusion_matrix.png)、[detector_confusion_matrix.csv](detector_confusion_matrix.csv)

| 母集団 | True accept | False reject | **False accept** | Correct reject |
|---|---:|---:|---:|---:|
| A2 main（112） | 72 | 6 | **0 / 34** | 34 |
| A2 main＋truth control（NC3/4/6）（124） | 76 | 6 | **0 / 42** | 42 |
| A0 main（112） | 9 | 0 | 0 / 103 | 103 |
| A3 main rep0（56） | 26 | 4 | 0 / 30 | 26 |

- A2 mainの各率（95% Wilson CI）：
  - True acceptance rate：92.3%（84–96%）
  - False rejection rate：7.7%（3–16%）
  - FAIL/AMBIGUOUS率：35.7%
  - PASS時のAffine正解率：72/72
- **False acceptance rate**：0/34。95%上限は10.2%（rule-of-threeでは8.8%）。main＋truth controlでは0/42、上限8.4%。
  - 有限の実験数による結果であり、**偽陽性率0%を保証するものではない**。
  - 族ごとの母数は7〜10件と少ない。族別の上限は28〜35%。

### 危うい事例（[false_accept_examples.png](false_accept_examples.png)）

誤採択は0件だったため、図には「最もPASSに近かった誤ったAffine」を示した。

- いずれもAMBIGUOUSで止まっている。
- 主にinlier3（≤3 µm一致率）がPASS条件の0.25を下回ったことで止まった（例：G2×C13 RMS 7.3 µm、inlier3 0.19、null比4.7、block 0.70）。

### 最も重要なcontrol：NC2（組織パターンは一致、共通核なし）

- G3/G4では、A2が**組織スケールでは正しいAffine**（RMS 4.1 / 0.7 µm）を回復した。
- 検出器はAMBIGUOUSを返した（inlier3 0.16/0.18、null比 3.0/3.6、block 0.36/0.32）。
- ただし、G4ではPASS条件のうち null比 ≥3 を満たしており、G3も2.96で閾値をわずかに下回っただけである。止めたのは主に inlier3 と block の2条件で、**余裕は大きくない**。
- 下流を計算すると、B2は **184件・293件のmatchをすべて誤って採択**した（precision 0）。
- したがって、検出器のgateは安全上必須である。

### 厳しすぎる判定（False reject 6件）

- いずれも変形RMS 6〜12 µmのcase（C12 4件、C13 2件）。
- inlier3が0.17〜0.21で、PASS条件の0.25を下回った。
- このうち4件は、下流も成功していた（top-1 0.91〜0.97）。
- 閾値3 µmは変形に対して保守的である。安全側ではあるが、強い変形下では採択率が落ちる。

## 5. Q4. Does correct Affine recover cell correspondence?

**回復する。** 下流（B2）の状態を3つに分けた（[graph_accuracy_vs_affine_error.png](graph_accuracy_vs_affine_error.png)）。

- **Graph success**の定義（事前固定）：top-1 ≥0.80、かつ B2 precision ≥0.90。

| | A0 | A2 | TRUE（参照） |
|---|---:|---:|---:|
| 1. 正しいAffine＋Graph成功 | 9 | **76** | 107 |
| 2. 正しいAffine＋Graph失敗 | 0 | 2 | 5 |
| 3. 誤ったAffine＋Graph失敗 | 103 | 32 | — |
| 4. 誤ったAffine＋Graph成功 | 0 | 2 | — |

- **状態2**は、すべてC13（変形12 µm）。真のAffineを入れてもtop-1は0.66〜0.79だった。これは**Graph Matching固有の限界**である。
- **状態4**は、RMS 6.6 / 7.4 µmのimprecise例。5 µm閾値の近くでも、Graphは許容できている。
- **全体（112 case、研究診断値）の中央値**

| 指標 | A0 | A2 | TRUE |
|---|---:|---:|---:|
| Candidate Recall@30 | 0.08 | 1.00 | 1.00 |
| Top-1 | 0.00 | 0.95 | 0.97 |
| B2 precision | 0.00 | 0.98 | 0.99 |
| Recall | 0.00 | 0.60 | 0.86 |
| F1 | 0.00 | 0.74 | 0.92 |
| Unmatched F1 | 0.60 | 0.87 | 0.90 |
| Coverage | 0.01 | 0.42 | 0.46 |
| False match rate | 1.00 | 0.02 | 0.01 |
| Duplicate destination | — | 1 | 1 |

- **A2がPASSしたcase（72件）**
  - top-1：中央値0.971、最小0.855
  - B2 precision：中央値0.986、最小0.946
  - recall 0.88、F1 0.93、unmatched F1 0.90、coverage 0.59
- **運用上の採択**（FAIL/AMBIGUOUSは停止）
  - A2：112件中40件を停止。採択されたmatchは72,041件で、precision 0.986。
  - A2のAffineが誤っていたcaseで計算した診断用の下流は、B2 matched 3,902件・うち誤り814件だった。これらはすべて検出器で停止している（PASS 0件）。

## 6. Q5. Does the synthetic–real gap remain?

**残る。ただし方向は「Realの方が容易」である。**

- 比較対象：A2で補正したReal（PASS、held-out 0.796、inlier/null 17.6）と、A2がPASSした独立Synthetic 72件。
- 詳細：[synthetic_real_distribution.csv](synthetic_real_distribution.csv)、[synthetic_real_domain_gap.png](synthetic_real_domain_gap.png)

| 指標 | Real（A2） | Synthetic PASSの中央値［5–95%］ | Realの位置 |
|---|---:|---|---|
| 核密度（/µm²） | 0.00199 | 0.00206［0.00169–0.00408］ | 中央付近 |
| kNN中央値（µm） | 11.2 | 10.0［8.0–11.4］ | 93 percentile |
| 候補距離中央値（µm） | 1.33 | 2.66［1.31–7.31］ | 8 percentile（無変形のC02相当） |
| NCC peak / margin | 0.70 / 0.16 | 0.70 / 0.07 | 51 / 92 percentile |
| inlier3 | 0.80 | 0.61［0.34–0.73］ | **全例より高い** |
| Graph support中央値 | 0.82 | 0.74［0.41–0.78］ | **全例より高い** |
| UOT row mass | 0.51 | 0.49［0.45–0.51］ | 86 percentile |
| B2 confidence | 0.96 | 0.88［0.38–0.94］ | **全例より高い** |
| B2 coverage | 0.76 | 0.59［0.22–0.70］ | **全例より高い** |

- 解釈：Realは、独立Synthetic（変形RMS 2 µm、共有率0.65〜0.70）よりも**切片間の残差が小さく**、対応のある核の割合も高いように見える。
- **注意**：Realには真の細胞IDがない。Graph supportやB2 confidenceの分布が一致・超過していても、Realの正答率が高いとはみなせない。
- 逆に言うと、今回のSyntheticの失敗領域（低いmoving被覆率、12 µm変形）がReal pairで起きうるかは、このReal 1 pairからは判断できない。

## 7. Q6. 次に改善すべき点（最大3件。採否は別の検証データで判断する）

1. **重なりを考慮したrefine（最優先）**
   - 失敗の主因は、NCCの初期値ではなく、ICPのscale崩壊である。
   - 候補となる対策：
     - NCCで得たscaleを拘束、または事前分布としてICPに与える
     - correspondenceを推定重なり領域（fixed hull ∩ 推定moving FOV）に限定する
     - objectiveに重なり面積の項を加える
   - これらを、**新しい独立検証セット**（別のseed key、別の被覆率分布）で評価する。今回のC09/C10を見て調整し、同じデータで採否を判断することはしない。
2. **探索範囲の起点**
   - 点群重心ではなく、FOV中心・pixel scaleを使う。あるいは、FOVサイズに応じて探索範囲を広げる。
   - 重心ずれが200 µmを超えるcase（C05、G2×C04）は現在すべて停止しており、安全だが回収はできない。
3. **変形に応じた判定とGraph許容幅の検討、およびReal側の外部情報**
   - 変形6〜12 µmでのfalse reject（6件）と、Graph状態2（5件）への対策を検討する。
   - あわせて、独立landmarkやHE由来の指標を持つReal pairを追加する。
   - Syntheticの方が厳しいという方向の乖離があるため、閾値の再校正にはReal側の真値が必要である。

## 8. 最終研究判断

- **主結論：C. LOW-OVERLAP FAILURE DOMINATES**
  - A2の失敗34件のうち24件は、重なり不足（moving側被覆率 <0.7：16件）と探索範囲外（8件）だった。
  - 前者の機構は、正しいNCC初期値からのICP scale崩壊である。
  - それ以外の条件では、4族すべてで69/72成功した。
- **副結論1（A. EXTERNAL VALIDATION SUPPORTED：検出器と下流に限定）**
  - 凍結検出器は、独立generatorで誤ったAffineをPASSしなかった（0/42、95%上限8.4%）。
  - 共通核のないcontrolもPASSしなかった（0/12）。
  - PASSしたcaseの下流は、真のIDでtop-1 ≥0.855、precision ≥0.946だった。
  - Affine自体の成功率（70%）は「汎化した」とは言えないので、Aを主結論にはしない。
- **副結論2（E：限定的）**
  - 強い変形（12 µm）では、真のAffineでもGraph Matchingが失敗した（5/8）。これは独立したGraph側の限界である。
- **棄却した結論**
  - D（構造・繰り返しの曖昧性）：G4単独では失敗しなかった。
  - B（検出器の弱さ）：False acceptが0件だった。

## 9. 限界

- **統計的独立性**
  - 条件は設計されたものであり、母集団からの無作為標本ではない。全体の成功率は計画に依存する。
  - generator間の汎化については、実効的な独立単位は**4族のみ**である。
  - 各条件・族は2 replicateで、族×条件の単位では推定が粗い。
- **生成モデル**
  - 2Dの部分共有モデルで、核の楕円体・切片厚・染色などの画像情報はない。
  - 繰り返し構造はG4の1方式のみ。
- **事後解析**
  - moving側被覆率と失敗分類は事後指標である。被覆率0.7という境界は、今回のデータから読み取った値である。
- **Real**
  - 1 pairのみ。真値がないため、gapは分布の比較にとどまる。
- **A3**
  - 補助として56件のみで実行した。

## 成果物・再現

### ファイル構成（`analysis/affine-a2-independent-validation-20261015/`）

| 種類 | ファイル |
|---|---|
| 文書 | `REPORT.md`、`experiment_protocol.md`、`generator_description.md` |
| CSV | `affine_accuracy.csv`、`detector_confusion_matrix.csv`、`negative_control_metrics.csv`、`downstream_correspondence_metrics.csv`、`synthetic_real_distribution.csv`、`runtime_metrics.csv` |
| JSON | `diagnostic_summary.json` |
| 図 | `synthetic_generator_examples.png`、`affine_success_by_generator.png`、`affine_error_vs_overlap.png`、`affine_error_vs_shared_fraction.png`、`detector_confusion_matrix.png`、`false_accept_examples.png`、`graph_accuracy_vs_affine_error.png`、`synthetic_real_domain_gap.png` |
| コード | `indep_gen.py`、`design.py`、`run_validation.py`、`aggregate.py`、`diagnose_failures.py`、`freeze_manifest.py`、`test_validation.py`、`case.sbatch` |
| 中間結果 | `cache/`（per-case JSON、表） |
| 凍結記録 | `freeze/`（hash、job ID、pip freeze） |
| ログ | `logs/` |

### Runtime（4 CPU）

| 方法 | 中央値 | 最大 | traced memory |
|---|---:|---:|---:|
| A2 | 42 s | 111 s | 約65 MB |
| A0 | 6 s | — | — |
| A3 | 111 s | — | — |
| Real A2 | 89 s | — | — |

下流は1 jobあたり約1.7分。GPUは使っていない。

### 再現手順

1. `python -B freeze_manifest.py`
2. `sbatch -J main --array=0-111 case.sbatch main`
3. `sbatch -J ctl --array=0-23 case.sbatch control`
4. `sbatch -J real case.sbatch real`
5. `sbatch -J ds --array=0-343 case.sbatch downstream`
6. `sbatch -J realds case.sbatch real_downstream`
7. `python -B aggregate.py`
8. `python -B diagnose_failures.py`

Tests：`pytest test_validation.py`（14 passed）。
