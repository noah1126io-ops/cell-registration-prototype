# Experiment protocol：A2独立Synthetic外部検証（2026-10-15、実行前に固定）

この文書・`design.py`・`indep_gen.py`・`freeze/frozen_components.json` は、独立generator上でA0/A2/A3を1件も実行する前に作成した。結果を見た後に計画・閾値・成功定義を変更した場合は、REPORTに変更として明記する。

## 1. 凍結対象（変更禁止）

`freeze/frozen_components.json` に、次のSHA-256と設定値を保存した。

- A2本体（`init_lib.method_a2`）
  - Gaussian rasterization（σ 8 µm、4 µm px）
  - 凸包mask付きPadfield masked NCC（overlap 30%未満は除外）
  - 探索範囲：translation ±200 µm、回転は粗4°→細1°、scaleは{RMS比, NN比}×{0.97, 1, 1.03}
  - 上位6仮説のrefine
  - production ICP：`icp_similarity` trim 0.8 → `affine_icp` trim 0.7
- A0 / A3（baseline・補助）
- 失敗検出器 `init_lib.detector` と `DETECT` 閾値
- Graph/UOT/B2：`experiment_lib`。B2は保存済み `per_nucleus_results.csv` から決定的に再fitする（前回と同一）。

検出器の使い方は前回と同じにする。

- **Synthetic**：`basin_margin` のみ渡し、held-outは使わない（前回のsynthetic 180評価と同一）。
- **Real**：`basin_margin`（単一basinならinf）と、checkerboard held-out平均を渡す（前回の `aggregate.py` と同一）。

追加の確認：

- 実行前後で、tracked file 81件と既存analysis 508ファイルのSHA-256を照合する（`freeze/*_before.txt` / `*_after.txt`）。
- Real A2を再実行し、前回のA2 Affineと一致することを確認する（凍結の再現確認）。

## 2. 生成モデル

`indep_gen.py` に新規実装した（`generator_description.md`）。G1 Clustered、G2 Inhomogeneous、G3 Structured Tissue、G4 Repeated/Ambiguousの4族。

- seed：SHA-256で派生したkey（`SEED_ROOT=20261015`、purpose・family・condition・replicate）を使う。前回のseedや潜在テンプレートは使わない。
- 各caseは固有の潜在組織を持つ。
- 共通細胞ID・真のAffineはpairに保持するが、アルゴリズムには渡さない。

## 3. ケース計画（`design.py`）

- **Main**：4族 × 14条件 × 2 replicate = **112 pair**。
  - A0とA2は全pairで実行する。
  - A3はreplicate 0（56 pair）で実行する。
- **Control**：**24 pair**（NC1〜NC6、§5）。
  - A0とA2で実行する。NC3のみA3も実行する。
- **Downstream**：344 job。
  - 内訳：main 112 pair × {A0, A2, TRUE}、およびNC1/NC2のA2。
  - TRUEは真のglobal Affineを入れた**診断用の参照**であり、方法ではない。
- **Real**：A2を1回、checkerboard held-out用に2回、さらに下流を1回実行する。

| 条件 | 内容 | 主な因子 |
|---|---|---|
| C01_base | offset 60 µm, rot 2°, reflect, 1.10 µm/px, q_shared 0.65, det 0.93/0.90, 非affine変形RMS 2 µm, 密度0.0020/µm²（Real相当）, fixed 1000 µm / moving 900 px | 基準 |
| C02 | offset 0, rot 0, 反転なし, 変形なし | transform / chirality / no deform |
| C03 | offset 100, rot 3° | transform |
| C04 | offset 150, rot 6° | transform |
| C05 | offset 320（探索範囲外） | out-of-range |
| C06 | offset 120, rot 5°, 1.30 µm/px, moving 760×520 px, 密度0.0030 | 複合（scale・非対称crop・中密度） |
| C07 | q_shared 0.30 | 共通核率 低 |
| C08 | q_shared 0.85, det 0.50/0.50（観測上の共有率はC07と同程度） | 検出漏れ |
| C09 | moving 1230 px（1353 µm）がfixedを包含、offset 40 | 重なり中（範囲内） |
| C10 | fixed 520 µm（movingに包含）、offset 60 | 重なり低（範囲内） |
| C11 | moving 900×600 px, offset 160 | 非対称crop・部分重なり |
| C12 | 変形RMS 6 µm + 局所核集団消失3 patch | 変形 中 |
| C13 | 変形RMS 12 µm | 変形 強 |
| C14 | 密度0.0042 + 不均一強化（hetero 1.8） | 高密度・強不均一 |

## 4. 評価指標と成功定義（事前固定）

### Affine

- **Affine success**：common physical region上のRMS変換誤差 ≤5 µm、かつchiralityが正しいこと。
- common physical regionの定義：
  - moving画像を41×41格子（pixel座標）でサンプルする。
  - そのうち、真のAffineによるworld座標（µm）がfixed FOV矩形の内側に入る点を評価点とする。
  - 誤差は |A_est(p) − A_true(p)|（µm）。
- 前回定義（moving全視野21×21格子のRMS）も併記する。
- その他の記録項目：
  - translation（moving中心）・rotation・scale・chiralityの誤差
  - ICP objective、真の解から始めたICPのobjective
  - NCC peak、2位peak（15 µm以上離れた別仮説）、NCC peak margin
  - 空間重なり（物理FOV多角形の面積比）、centroid shift
  - runtime、検出器出力
- **AffineとDetectorは別の指標**として扱う。

### Detector

- confusion matrixは、真のAffine（success/failure）× Detector（PASS / AMBIGUOUS・FAIL）で集計する。
- **False accept**：誤ったAffineがPASSになったもの。最重要の安全指標とする。
- 率にはWilson 95%信頼区間を付ける。
- 0件であっても「0%を保証」とは書かない。0/nの場合はrule-of-threeによる上限も併記する。

### 下流

production同等のsubset規則のもとで、次を報告する。

- Candidate Recall@30
- top-1 accuracy（subset内の共通核）
- B2のmatched precision / recall / F1、unmatched F1、coverage、false match rate、duplicate destination

**Graph success**：top-1 accuracy ≥0.80 かつ B2 matched precision ≥0.90。

報告では次の3状態を分ける。

1. 正しいAffine＋Graph成功
2. 正しいAffine＋Graph失敗
3. 誤ったAffine＋Graph失敗
4. （残余）誤ったAffine＋Graph成功

**運用上の採択**：DetectorがFAIL/AMBIGUOUSのcaseは停止とし、matched 0として扱う。研究診断用に計算した下流結果とは混同しない。

### Domain gap

Real A2補正後と、独立Synthetic（A2 subset）で次を比較する。

- 密度、kNN距離
- 候補距離
- NCC peak・margin
- Graph support
- 残差の整合性（inlier3、block consistency、ICP objective）
- UOT row mass
- B2 confidence

Realには真の細胞IDがないため、Graph supportなどの分布が一致しても、正答率が一致するとはみなさない。

## 5. Negative / adversarial controls

| Control | 族 | 内容 | PASSの意味 |
|---|---|---|---|
| NC1_unrelated | G1–G4 | 独立な組織同士 | 常にfalse accept |
| NC2_density_matched | G1–G4 | 同じテンプレート（組織スケールのpattern一致）から独立に核を抽出。共通核なし | 常にfalse accept（組織スケールのAffineが正しくても細胞対応は存在しない） |
| NC3_repeated_near_exact | G4×4 | 格子コピーのjitter 1.2 µm・保持0.92（ほぼ同一の繰り返し） | 誤ったAffineならfalse accept |
| NC4_tiny_overlap_off900 | G1–G4 | 重なり約10% | 誤ったAffineならfalse accept |
| NC5_wrong_chirality | G1–G4 | 誤ったchiralityのみ探索 | 常にfalse accept |
| NC6_oor_off400 | G1–G4 | 探索範囲外offset 400 µm | 誤ったAffineならfalse accept |

## 6. 独立性の規則

- 結果を見た後に、A2のパラメータ・Detectorの閾値・generator・計画を変えない。
- 失敗の原因診断は行ってよい。ただし、改善案の採否はこのデータでは判断しない。
- 成功例だけを抽出しない。族別・条件別の結果をすべて残す。
- 同じ族・条件のcaseは生成器を共有している。generator間の汎化について、実効的な独立単位は4族しかない。
- 実装上のbugによる再実行は許可する（評価コードのみ）。行った場合はREPORTに記録する。
