# Real Affine / Coordinate / Candidate Discovery 独立監査（Claude）

実施日: 2026-10-14 ／ 対象: `serial_section_pair_20260904`（run `20260904-161247-698fe6c9`、sweep inputと全ファイルhash一致）／ branch `feature/gpu-slurm` @ `f66ecbe`。
analysis専用。Workflow C/D・UI・safety gate・Graph/UOT・閾値・既存analysisは変更していない（tracked 81ファイル・既存analysis 228ファイルのSHA-256を前後で照合）。deformation fieldは生成していない。以下の「診断用再fit」はglobal Affine 1本の診断計算であり、production提案ではない。

| Audit | Status | Evidence | Severity |
|---|---|---|---|
| Coordinate system | **PASS**（内部経路）／metadata UNKNOWN | 再構成誤差0 µm、往復1.7e-13 px、subset exact再現、chirality確認。spacing/crop/CRSは未保存 | LOW（規約）／MEDIUM（provenance） |
| Affine direction | **PASS** | HE px→蛍光world、二重適用なし | LOW |
| Affine alignment | **FAIL** | 保存Affineは共有核解から **≈(−7.6, −49.9) µm＋1.24°** ずれた局所解。≤3 µm一致 7.2%（null 4.8–5.0%）→ 再fit 73.8%（held-out 71–79%）。ICP目的関数 7.54→1.04 µm | **CRITICAL** |
| ROI overlap | **FAIL** | HE視野と蛍光視野が約50 µmずれ、centroid初期化を誤誘導。"tissue mask"はHE画像footprint（白色無組織域7.7%を含む） | HIGH |
| Regional NCC | **FAIL（設定）** | ±18 µm窓 < 残差50 µm。境界到達58.4%はmirror null（61.2%）と同水準＝窓内に真のpeakなし。±72で2.5%、再fit後±18で4.8% | HIGH（Bの帰結） |
| Candidate discovery | **PASS（K=30）**／K=10でFAIL | 共有核proxy partnerはfinal-30に94.4%（spatial rank中央値16、距離50 µm）。K=10では21% | MEDIUM |
| Candidate pruning / ranking | **FAIL** | proxy partnerがtop-1になるのは0.08%。position costが残差0を仮定するため | HIGH（Bの帰結） |

PASS/FAIL基準: PASS=実装・保存artifact・独立チェックが整合。FAIL=再現可能で、null・held-out・画像の独立証拠で裏付けられた不整合。真のTREは独立landmarkがないため算出していない。

---

## 核心となる観測

**Realの2切片は多数の核を共有しているが、保存Affineはそれらを約50 µm離れた位置に置いている。**

1. **局所displacement correlogram**（fixed−moving差分ベクトルの2D分布）: 200 µm格子の **25/25 blocks** で鋭いpeak（3 µm kernel、背景比中央値12.2、最小7.9）。同じ処理のmirror/180° nullは最大4.7。peakにはblockあたりmoving核の約67%が乗る。offsetは滑らかで、(dx,dy)≈(−2…−20, −38…−62) µm。線形モデル（回転1.22°、RMS misfit 0.48 µm）でよく近似できる。→ 非剛体の局所変形ではなく、**global Affineの平行移動＋回転の誤差**。
2. **保存Affineの近接統計はchanceに近い**: ≤3 µm 7.2%（mirror null 4.8–5.0%、Poisson 5.3%）、径方向cross-g(0–2 µm)=1.42。correlogramで初期化して**production `affine_icp`** でrefineすると、≤3 µm 73.8%、NN中央値 1.36 µm、g(0–2)=23.9。
3. **Held-out**: 片半分の核とblockだけで再fitし、反対側半分で評価すると ≤3 µm 71.4–78.6%（保存 6.3–7.6%、null 4.1–5.8%）。4分割すべてで再現。
4. **点照合と独立な画像チェック**: HE画像の白色無組織域（裂隙・管腔、footprintの7.7%）に落ちる蛍光核は、保存Affineで6.4%（≒面積比、回避なし）、再fit後0.26%（HE核自身は0.05%）。蛍光核を平行移動しながらこの割合を最小化すると、offset (−15, −51) µmで最小になり、点correlogram (−7.6, −49.9) と独立に一致する。
5. **なぜICPが外したか**: 4 flip×12回転の48通りの標準初期化はいずれも共有核basin（目的関数1.04）に到達しない（最良7.54）。保存Affineからの平行移動摂動289通りでは、真の解から概ね±30 µm以内の26通りだけが同じbasinに収束した。初期化はcentroid一致で、HE視野と蛍光視野が約50 µmずれているため、centroid一致が真の位置から約50 µm外れる。NN間隔約11 µmのICPはこの距離から脱出できない。

---

## Q1. Is the coordinate system correct?

**内部規約はPASS。** 単位・方向・軸・反転・ID/indexの誤りは見つからなかった（[coordinate_audit.md](coordinate_audit.md)）。chiralityは、現行の反射を含む向きでのみ共有核peakが出ることで確認した。一方、raw spacing・crop・downsampling・CRSは未保存（UNKNOWN）。これらは誤りの証拠ではないが、視野ずれを事前に検出する手段がない。

## Q2. Is the Affine transform correctly implemented and applied?

**実装・適用はPASS、推定結果はFAIL。** 式・方向・適用順は正しく、二重適用もない。ただし推定値はICPの局所解である。

| | 保存Affine | 診断用再fit | 差（world） |
|---|---|---|---|
| post-flip matrix | [[−1.0964, 0.0135],[−0.0081, −1.1004]] | [[−1.1063, 0.0346],[−0.0349, −1.1070]] | 回転 +1.24°、scale +0.5〜1.1% |
| translation | (1002.8, 1013.5) | (989.5, 978.2) | 移動中央値 (−7.6, −49.9) µm、範囲 37.7–64.9 µm |
| ICP trimmed objective | 7.54 µm | **1.04 µm** | |

再fitは同じ反射・同じICPコードを使い、初期値だけを変えた。目的関数がずっと低いので、production推定器自身の基準でも保存解は劣る。

## Q3. Is there systematic residual misalignment?

**はい。大きく系統的である。** 全域で約50 µm、空間的に滑らかで、Affineで説明可能（回転1.2°＋並進）。局所非剛体成分は、線形モデルからのRMS 0.5 µm以下ではほぼ検出されない。

Codex監査の局所NN vector（中央値 (0.19, 0.41) µm）は、隣の無関係な核を指すためこのずれを捉えられない（[affine_residual_vectors.png](affine_residual_vectors.png) 中央）。NN vectorは真の移動ではなく、今回もそれが示された。独立landmarkがないためTREは主張しない。ただし、held-out点照合とHE画像による独立チェックの2系統が一致している。

## Q4. Why do 58.4% of regional offsets hit the search boundary?

**探索窓（±18 µm）が残差（約50 µm）より小さく、窓内に真のpeakが存在しないため。** 窓内のNCC曲面はchanceの形をしている。

| 条件 | 核境界率 | 独立anchor境界率 | correlogramと6 µm以内一致 | 隣接anchor offset差中央値 |
|---|---:|---:|---:|---:|
| 保存 ±18 | 58.4% | 58.3% (193/331) | 0% | 19.0 µm |
| 保存 ±36 | 54.0% | 54.1% | 10.9% | 28.5 µm |
| 保存 ±54 | 33.4% | 32.6% | 67.7% | 3.0 µm |
| 保存 ±72（追加診断） | **2.5%** | 2.1% | **79.2%** | 3.0 µm |
| 再fit後 ±18 | **4.8%** | 4.9% | — | 3.0 µm |
| mirror null ±18 | 61.2% | 61.2% | — | 21.6 µm |

- ±18の境界率は無関係パターン（null 61%）と同じ。境界セルの面積比28%より高いのは、平滑なNCC曲面のchance最大値が縁に寄るため。
- 窓を広げると、NCCは**独立に推定した点correlogram offsetへ収束**し、隣接anchor間でも一致する（3 µm＝1格子）。Codex監査が「遠方density peakへの吸引」と解釈した±54のoffset中央値 (−6, −45) µmは、真のずれである可能性が高い。±54で残る境界到達（y−方向）は、左側の真のdyが−58〜−62 µmと窓外にあるためである。
- 再fit後は元の±18でも境界率4.8%となり、synthetic正解Affine条件（Codex 7.4%）と同水準。**NCC自体は機能する。** 問題は前段のAffine。
- 核単位と独立anchor単位の率はほぼ同じ（331 anchorを共有するため、核単位の値は独立標本数ではない）。

## Q5. Are potentially useful candidates being missed or pruned?

真の対応がないため、correlogramから導いた **proxy partner**（moving+局所offsetから3 µm以内の相互最近傍）で評価した。存在率73.1%（mirror null 7.5%）。再fitから独立に求めたpartnerとの一致は99.8%。

- **探索（discovery）**: proxy partnerはspatial-30に96.5%、regional-30に83.1%、fingerprint-30に81.8%、union（中央値37）に95.4%、**final-30に94.4%** 含まれる。K=30では大部分が候補内にある。K=10ではspatial 14%、augmented 21%しか含まれない（K=50で100%）。
- **選択（ranking/pruning）**: しかしtop-1になるのは **0.08%**。position cost（約50 µm → 約5.6）がpartnerを罰し、Graphのcompatibility項 `exp(−0.35·base_cost)` も近傍の遠い候補を罰する。そのため正しい候補群が候補内にあっても、整合する部分グラフとして浮上しない。
- ±54 NCC offsetをregional sourceに入れると、K=10のcoverageは88%に上がる。しかしGraph supportは0.015に崩れる（position項は残差0を仮定、region項は−50 µmを仮定し、両者が矛盾するため）。**窓拡大だけで改善とは言えない**というCodexの指摘は正しいが、原因は偽peakではなくcost項間の矛盾である。
- 再fit後は、全K（10–100）でpartner coverage 100%、top-1=partner 99.6%、top-1 Graph support 0.807。

→ **Eは独立原因ではなくBの帰結。** 現行の30候補は、正しいAffine上では十分である。

## Q6. What best explains the low Real Graph support?

**B（Affine misalignment）が主因。** 凍結pipeline（Codex実験と同一コード、B2は保存結果から再fitし、Real 26/209/1519を完全再現）を条件だけ変えて実行した:

| 条件 | Graph support | top-1距離 | 最小cost | UOT row mass (B1) | B2 conf | B2 matched / amb. / unmatched |
|---|---:|---:|---:|---:|---:|---|
| Real 保存Affine | 0.227 | 10.6 µm | 0.804 | 0.466 | 0.172 | 26 / 209 / 1519 |
| Real mirror null（無関係） | 0.178 | 11.3 µm | 0.851 | 0.452 | 0.136 | 8 / 155 / 1591 |
| **Real 再fit（診断）** | **0.805** | **1.38 µm** | **0.154** | 0.505 | **0.957** | **1267 / 135 / 352** |
| Real-like synthetic Test（既存） | 0.718 | 4.98 µm | 0.290 | 0.467 | 0.804 | — |
| synthetic 正解Affine（5 seeds） | 0.62–0.77 | 3.0–7.4 µm | — | — | — | top-1 accuracy 0.87–0.99 |
| synthetic ＋Real実測残差を注入 | 0.11–0.16 | 11.8–12.5 µm | — | — | — | top-1 accuracy 0.01–0.03、NCC境界49–84% |

- 保存AffineのRealは、統計的に**無関係パターンと区別できない**。既存negative control（unrelated pattern: Graph 0.202、距離11.0、B2 conf 0.168）ともほぼ一致する。
- 実測残差をsyntheticに注入すると、Realのsignature（Graph約0.15、距離約12 µm、NCC境界率高）が再現される。Affineを直すとRealはsyntheticと同等以上になる。
- matched数1267は**正解数ではない**（真値なし、B2はsynthetic校正）。主張できるのは「domain gapの大部分がAffine誤差で説明される」まで。
- 空間的には、保存Affine下のGraph supportは全域で一様に低い（局所NN距離との相関 ρ=−0.64 はnaive p値で、空間自己相関を補正していない）。再fit後に低いまま残るのは下端帯（y<100 µm、partner率12%）と左端列（HE裂隙付近）で、新しいROI上の非重複域に対応する。
- **F（生物学的・切片差）** は残る約26%の非共有核や裂隙の説明にはなり得るが、Graph supportが低い主因ではない。

## Q7. What should be corrected or investigated next?

今回は修正していない。推奨順:

1. **Affine初期化の修正を設計・検証する（最優先、production変更は別依頼）。** centroid一致に頼らない粗探索（点correlogram/displacement-histogram peak、またはNCC ±72程度の並進探索）で初期化し、既存ICPでrefineする。採否は、synthetic（既知の視野ずれ・crop差を含める）とRealのheld-out検証で事前に決めておく。
2. **入力provenanceの保存**: HE/蛍光それぞれの取得範囲（stage座標・crop）、spacing、downsampling、CRS。今回の約50 µmずれは視野の違いが原因であり、metadataがあれば事前に検出できた。
3. **"tissue mask"の意味を修正する**（HE白色域を組織外とする実マスク）。subset・ROI・edge判定が現在は画像footprint基準になっている。
4. **少数の独立landmark**（両画像で同定できる核・血管）でTREを測り、再fitの精度を確定する。
5. Affineを修正するまで、Real上でGraph/UOT/B2/NCC窓/Kを調整しないこと。現状のReal指標はAffine誤差を測っているだけである。
6. その後、Real-like generatorに「視野ずれ・crop差」と「切片間で核を共有しない割合（約26%）」を加える。3D serial-section generator（§8構想）の必要性は、この残差を見てから判断するのが妥当（今回は実装していない）。

## 原因分類

| 分類 | 判定 | 区分 |
|---|---|---|
| A. Coordinate system error | **否定** | 観測: 規約整合、chirality確認 |
| **B. Affine misalignment** | **主因（確定的）** | 観測: correlogram 25/25 blocks、held-out、HE画像、ICP目的関数 |
| **C. ROI/overlap** | **寄与（Bの原因）** | 観測: 視野が約50 µmずれ、mask=footprint。推論: centroid初期化を誤誘導した |
| D. Regional NCC limitation | **副次（Bの帰結）** | 観測: 再fit後±18で境界率4.8% |
| E. Candidate discovery/pruning | **副次（Bの帰結）** | 観測: final-30に94.4%含まれるがtop-1は0.08%。再fit後99.6% |
| F. Biological/domain | **小さい残余** | 推論: 約26%の非共有核と裂隙。主因ではない |
| G. Inconclusive | 該当せず（ただしTRE未測定） | 未確認: 真の対応・landmark |

## Codex監査との比較

| 点 | Codex | Claude | 差の理由 |
|---|---|---|---|
| 実装・方向・subset | PASS／exact | 同一（独立再実装で一致） | — |
| ±18/36/54 境界率 | 58.4/54.0/33.4% | 同一 | 同一データ・同一定義 |
| Affine alignment | UNKNOWN（NN proxyのみ） | **FAIL** | null比較・局所correlogram・held-out・HE画像チェックを追加 |
| ±54のoffset (−6, −45) | 遠方density peakへの吸引 | **真の残差** | 独立correlogramと79%一致、隣接anchorで一貫 |
| K/pruning | top-1安定 → 主要因でない | 安定は事実だが、proxy partnerを0.08%しか選ばない | 「安定」は正しさを意味しない |
| 主因 | D＋G | **B（＋C）**、DとEは帰結 | — |

Codexの数値はすべて独立に再現した。差は解釈と追加検証による。

## 検証・再現性

- 凍結regional NCCとvectorized版: offset完全一致、NCC差 ≤1.3e-15。凍結Graphとvectorized版: 差 5.6e-17。保存candidate hash再現。B2 refitで保存Real結果のstatusが全一致（confidence差 4.4e-16）。
- UOT solverは全条件で収束、全cost有限、NaN/Infなし。
- Analysis tests: 7 passed。Repository tests: GPU割当（`--gres`）なしでは2件のGPU parity testが `CUDA_VISIBLE_DEVICES is None` で失敗する（環境要因、tracked fileは無変更を確認）。GPU割当ありの結果は `logs/repo-tests-gpu-*.out` を参照。
- 実行: Slurm `mib-dbia`（CPU、計約12分）。`python -B run_audit.py --stage all` → `make_figures.py` → `finalize.py` で再生成できる。

## 成果物

`coordinate_audit.md`, `affine_audit.json`, `roi_overlap_metrics.json`, `ncc_window_sensitivity.csv`, `candidate_source_audit.csv`（stable ID、各source候補、proxy partner rank）, `candidate_k_sensitivity.csv`, `spatial_residual_metrics.csv`, `diagnostic_summary.json`; 図 `coordinate_transform_diagram.png`, `raw_vs_affine.png`, `roi_overlap.png`, `affine_residual_vectors.png`, `regional_ncc_boundary_map.png`, `ncc_window_comparison.png`, `candidate_distance_distribution.png`, `graph_support_map.png`, 追加 `icp_capture_and_he_gap_check.png`。中間結果は `cache/`。

## 解釈の限界

proxy partnerは真の細胞IDではない（null偶然率7.5%）。再fit Affineは診断値であり、production採用の検証はしていない。matched数・Graph support・NN一致率の上昇は対応精度の証明ではない。ただし、点照合と独立なHE画像チェックとheld-out検証が同じ解を支持している。
