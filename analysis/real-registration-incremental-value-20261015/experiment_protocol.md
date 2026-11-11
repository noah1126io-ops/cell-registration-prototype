# Experiment protocol：Real registrationの段階別寄与（2026-10-15、比較・残差解析の実行前に固定）

この文書は、Real上でstage間の比較指標・残差診断・Synthetic Testを1件も計算する前に作成した。
例外として、`stages.py` の動作確認のためReal A2を1回実行した（R2が既存A2と完全一致、Real上はR1a=R1bであることのみ確認）。結果を見た後に変更した場合はREPORTに明記する。

## 1. 凍結対象（変更禁止・importのみ）

- A2（`init_lib.method_a2`）、production ICP（`src/pointset_registration`）、`init_lib.detector`
- Graph/UOT/B2（`experiment_lib`、B2は保存済み `per_nucleus_results.csv` から決定的に再fit）
- Regional NCC（監査で凍結コードと一致を確認済みの `audit_core.regional_ncc`、半径18 µm）
- 独立generator `indep_gen.py`（2026-10-15）
- 実行前後に tracked 81ファイル・既存analysis 1,479ファイルのSHA-256を照合する（`freeze/`）。

`stages.py` は `init_lib._finish` を **実行時に一時的にwrap** して、ICPに渡される仮説リストを記録するだけである（fileは変更しない）。R2は既存A2と一致することを毎回確認する。

## 2. Stage定義

| Stage | 定義 | ICP情報 |
|---|---|---|
| R0 | 保存Affine（centroid初期化＋ICP、`saved_effective`） | — |
| R1a | A2のfine NCC順位1位の仮説（similarity：chirality, θ, scale, 並進） | 使わない |
| R1b | A2が最終的に採用したbasinの、ICP前の仮説 | **仮説選択にICP objectiveを使う** |
| R2s（補助） | R1bにsimilarity ICP（trim 0.8）だけを適用 | 使う |
| R2 | A2最終出力（R1b → similarity ICP → affine ICP trim 0.7） | 使う |
| R3（Syntheticのみ） | 真のglobal Affine | — |

R1a=R1bの場合は1つの結果として扱い、その旨を明記する。

## 3. Real：評価点集合（ROIを方式に合わせて変えない）

- **ALL**：全fixed 1,979核・全moving 1,842核（ROIなし）。
- **COMMON（主）**：全stage（R0, R1a, R1b, R2s, R2）で共通に有効な核だけ。
  - moving：全stageの写像先が、fixed凸包を15 µm内側に縮めた領域に入る核。
  - fixed：全stageのHE視野（moving画像矩形の写像）を10 µm内側に縮めた領域に入る核。
- **STAGE-ROI**：各stageのproduction subset規則（`subset_rule`）。下流が実際に見る集合。

## 4. Real：指標

### 点群

- moving→fixed / fixed→moving NN距離（median, p90, p95）。symmetric NN = 両方向の距離をpoolしたもの。
- 1/2/3/5/10 µm以内の近接率（moving→fixed）、mutual NN率、mutual NN∧≤3 µm率
- mirror-x / mirror-y / 180°回転のnull（同じ評価集合）
- 局所密度相関：σ = 5, 10, 20 µmのGaussian密度mapのPearson相関（共通mask内）
- Regional NCC（凍結、±18 µm）：best NCC median、境界到達率、offset median
- 200 µm（および100 µm）blockの局所correlogram：search ±72 µm、kernel 3 µm。peak比（peak/背景）とrefined peak offsetの大きさ（= **local residual**、peak比 ≥5のblockのmedian）
- ICP trimmed objective（**ICP自身の目的関数＝ICPに有利な指標**として別記）

### 変換

- 行列、並進、極分解の回転、scale（√|det|）、異方性、shear、det
- R1→R2の座標変位：moving全核と画像格子での median / p95 / max。**変位量であり誤差ではない。**

### HE画像（moving点と独立）

- HE白色（無組織候補）領域へのfixed核の投影率（白色面積比と比較。白色＝真の無組織とは限らない）
- HEのhematoxylin様強度（色分解）とfixed核密度の相関（footprint内、σ 10 µm）
- 白色境界の符号付き距離別のfixed核密度（境界・裂隙付近の整合）
- 並進scan（±30 µm、2 µm刻み）：白色投影率最小／密度相関最大となるoffset＝**moving核を使わない独立な残留offset推定**
- HE視野（画像footprint）と点群ROI（fixed凸包）の重なり

撮像範囲metadataの有無は入力・runディレクトリで再確認し、同一とも異なるとも仮定しない。

## 5. ICPの追加価値（paired）

- 上記全指標についてR1a/R1b→R2s→R2の差
- 100 µm blockごとの≤3 µm率とlocal residualの改善／悪化割合
- **Held-out**（主）：fitに使った核で評価するとICPが有利になるため、学習核と評価核を分ける。
  - checkerboard 100 px（2色）、x半分、y半分の計6 fold。
  - 学習側だけでA2（計装版）を実行し、R1a/R1b/R2sを得る。
  - 評価は反対側の核で、≤1/2/3 µm率とNN median。
- Graph support・B2の差（§7）

## 6. 非剛体の必要性（新しい変形場は生成・採用しない）

### 残差source（いずれも真の細胞変位ではない）

- P：R2下のmutual NN pair（≤4 µm）のベクトル u = fixed − moving
- C：window 50/100/200 µmの局所correlogram peak（search ±15 µm、kernel 2 µm、moving ≥8核、peak比 ≥5）
- G：R2下流のB2 matched pairのベクトル

### 統計

- 残差の大きさの分布
- ベクトル自己相関 ρ(r) = mean(u_i·u_j) / mean(|u|²)（平均除去後）。lag bin 0–20, 20–50, 50–100, 100–200 µm
- 方向一貫性：半径 r（20/50/100 µm）内の単位ベクトル平均長
- 窓平均残差：20/50/100/200 µm窓
- 局所密度・Graph support／B2 confidence・ROI境界距離・HE白色境界距離との関係

### Held-out予測

- 100 µm / 200 µmのcheckerboard blockで、片方の色をfit、反対色をheld-outとする。
- fit側Pベクトルから、Gaussian kernel回帰で残差を予測する。bandwidthは {20, 50, 100, 200} µm から、**fit側内部の2-fold block CVだけで**選ぶ。
- held-outの核の予測値と、その核自身のPベクトルを比較する。
  - skill = 1 − SSE(予測) / SSE(fit側平均)
  - held-out window平均ベクトルとの相関
- 点の移動・補正は行わない。予測値は検定統計量であり、変形場として保存・採用しない。

### Null / 校正

- N1 位置shuffle：fit側ベクトルを位置間でpermute（200回）
- N2 無関係pattern：R2姿勢でmovingをmirror-xし、偶然のNN pairで同じ解析を行う
- N3 Real-template semi-synthetic：
  - Real fixed核から擬似movingを作る。共有率・pair残差のばらつきはRealのR2に合わせ、非共有核も加える。
  - 既知のGRF変位（RMS 0, 0.5, 1, 2, 4 µm、相関長110–200 µm、affine成分除去）を加える。各20 rep。
  - 振幅0はnull、それ以外は検出力の校正に使う。
- 独立generatorのSynthetic（§8 T1–T4）で、TRUEとR2の両条件について同じ診断を行い、真の変位場との一致を確認する。

### 判定規則（事前固定）

**「空間的に一貫した残差がある」** と判定する条件（すべて満たすこと）：

1. Realのheld-out skillが、両方のcheckerboard色で、N1とN3（振幅0）の95パーセンタイルを超える。
2. 1.を満たすscaleが少なくとも1つある。
3. 推定される一貫成分のRMSが0.5 µm以上。

上記を満たさない場合は「random（検出限界以下）」とし、N3で得た検出限界の振幅を併記する。

1.を満たすが一貫成分のRMSが0.5 µm未満の場合は、「検出可能だが小さい」とする。

## 7. 下流

- R0, R1a, R1b, R2s, R2を、同じ凍結Graph/UOT/B2に渡す。subsetは各stageのproduction規則による。
- R0については、保存subset配列（26/209/1519の再現）も併記する。
- 指標：
  - candidate距離
  - Graph support
  - top-1の一致（stage間、R2のtop-1との一致）
  - UOT row mass
  - B2 matched / ambiguous / unmatched、duplicate destination
  - matchedの100 µm block別分布：R1→R2の増分の集中度。Gini、増分の上位10% blockが占める割合、ICP変位量との相関
- Realではtop-1 accuracyを計算しない。

## 8. Synthetic（独立generator）

### Test（新規、未使用seed）

以下の7条件 × 4族 × 2 rep = **56 pair**。条件keyは `IV_*` で、既存112件とはseedが別になる。

| 条件 | 内容 |
|---|---|
| T1_affine_only | 変形0 |
| T2_nonrigid_mild | 変形RMS 3 µm |
| T3_nonrigid_moderate | 変形RMS 6 µm |
| T4_nonrigid_strong | 変形RMS 12 µm |
| T5_fov_contained | moving 1230 px、offset 40 |
| T6_fov_smallfixed | fixed 520 µm、offset 60 |
| T7_shared_low | q_shared 0.30 |

その他の設定はC01_baseと同じ。

### 評価

- 各pairについて、R1a, R1b, R2s, R2, R3で以下を評価する。
  - 真のAffineに対するcommon-region RMS
  - 共有核のtrue-pair残差（median、≤3 µm率）
- 下流（R1a, R1b, R2, R3）：Candidate Recall@30、Graph top-1、matched precision / recall / F1、unmatched F1
- 層別：global-only（T1）、軽度（T2）、中等度（T3）、強度（T4）、視野差（T5–T6）、低共有（T7）

### Dev（記述のみ）

- 既存112件で計装版A2を再実行し、R1a / R1b / R2を得る。
- 既存のA2結果と一致することを確認する。
- 開発で使用済みのデータなので、未見Testとしては扱わない。

## 9. 報告区分

- **Measured**：実際に計算した値
- **Supported inference**：独立性のある複数の診断が支持する解釈
- **Unverified**：真のIDや独立landmarkなしには検証できない事項

Realは1 pairである。真の細胞IDはない。
