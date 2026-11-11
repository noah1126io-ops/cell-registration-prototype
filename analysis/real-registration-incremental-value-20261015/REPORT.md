# 実データにおけるAffine・ICPの実効改善量と非剛体補正の必要性（2026-10-15）

対象：`serial_section_pair_20260904`（run `20260904-161247-698fe6c9`）。branch `feature/gpu-slurm` @ `f66ecbe`。analysis専用。

- A2・ICP・検出器・Graph/UOT/B2・Workflow C/D・UI・safety gateは変更していない。
  - 実行前後で、tracked 81ファイルと既存analysis 1,479ファイルのSHA-256が一致した（`freeze/`）。
- 新しい非剛体変形場は生成・採用していない。commit / pushはしていない。
- 計算はSlurm CPU（`mib-dbia`）。GPUは使っていない。

**指標の区別**：本文の「近接率」「NN距離」「Graph support」「B2 matched」は、すべて**幾何学的一致**の指標である。Realには真の細胞IDがないため、細胞対応の正答率は1つも計算していない。

## 結果の要約

| Stage | ≤3 µm近接率 | Symmetric NN median | Local residual | Graph support | B2 matched |
|---|---:|---:|---:|---:|---:|
| R0 Stored Affine | 0.065 | 10.44 µm | 51.1 µm | 0.228 | 27 |
| R1a NCC-only | 0.447 | 3.38 µm | 2.72 µm | 0.815 | 1,217 |
| R1b Selected NCC hypothesis | 0.447（R1aと同一） | 3.38 µm | 2.72 µm | 0.815 | 1,217 |
| R2 NCC + ICP | **0.785** | **1.32 µm** | **0.22 µm** | 0.821 | 1,324 |
| （補助）R2s NCC + similarity ICPのみ | 0.783 | 1.32 µm | 0.25 µm | 0.821 | 1,320 |

**列の定義**
- **≤3 µm近接率・NN**：COMMON集合（全stageで有効な共通の核：moving 1,621・fixed 1,724）で計算した。mirror nullの≤3 µm率は0.049〜0.062。
- **Local residual**：200 µm blockの局所correlogram peak offsetのmedian（25 blockすべてでpeak比 ≥5）。
- **Graph support**：Position+Graphのtop-1 support median。
- **B2 matched**：各stageのproduction subset規則で凍結B2を実行した。
  - R0の保存subset配列では26 / 209 / 1,519で、既存結果と完全に一致した。
  - R2は既存のA2結果（1,324 / 132 / 295）と完全に一致した。

**Real上ではR1a = R1b。** NCC 1位の仮説が、そのままICP選択でも勝った。そのためRealでは「ICP情報を使った仮説選択」の影響はない。

**Held-out**：学習核と評価核を分けた2つのcheckerboard foldの平均（fitした核で評価するとICPに有利になるため）。

| | ≤1 µm | ≤2 µm | ≤3 µm | NN median |
|---|---:|---:|---:|---:|
| R1a | 0.05 | 0.19 | 0.385 | 3.8 µm |
| R2 | 0.40 | 0.70 | 0.796 | 1.28 µm |

## 研究判断（Q6）

**主結論：B. ICP adds meaningful improvement（Real）**

**副結論：**

- **C. ICP refinement needs safeguards**：Realでは問題は出なかったが、Syntheticで再現性のある破綻条件がある。
- **E. Cell-level uncertainty dominates**：Affine後に残る約1 µmの残差は、空間的に一貫しないrandomな残差である。
- **F. More ground truth required**：細胞IDの正しさは未検証。

**棄却：**

- **D. Non-rigid refinement likely needed**：0.5 µm RMS以上・100〜200 µmスケールの一貫残差は検出されなかった。検出力はSyntheticで確認済み。
- **A. A2 alone sufficient**：held-outの≤3 µm率が0.385対0.796で、精度不足。

---

## Q1. How much does A2 improve Real alignment?（R0 → R2）

保存Affineは約50 µmずれた局所解だった。A2はこれを、共有核が一致する解へ移した。R0→R2の座標変位はmedian 51 µm（p95 63 µm）。

| 指標 | R0 | R2 | 種類 |
|---|---:|---:|---|
| ≤3 µm近接率（COMMON） | 0.065（null 0.05） | 0.785 | 点群 |
| Symmetric NN median / p95 | 10.4 / 29.2 µm | 1.32 / 16.1 µm | 点群 |
| Mutual NN ∧ ≤3 µm | 0.065 | 0.784 | 点群 |
| 密度map相関 σ5 / σ20 | 0.05 / 0.33 | 0.74 / 0.81 | 点群 |
| 局所correlogram residual（200 µm） | 51.1 µm | 0.22 µm | 点群（NN非依存） |
| Regional NCC（凍結±18）best / 境界到達 | 0.36 / 57.8% | 0.79 / 2.5% | 点群 |
| HE白色域へのfixed核投影（白色面積7.4%） | 6.4% | **0.22%** | HE画像（moving点と独立） |
| HE白色境界の内側0–5 µmの核密度（組織比） | 0.95 | 0.03 | HE画像 |
| fixed核位置のhematoxylin（z） | 0.06 | 2.43 | HE画像 |
| Graph support / B2 matched | 0.228 / 27 | 0.821 / 1,324 | 下流 |

- R0のHE白色投影率（6.4%）は面積比（7.4%）とほぼ同じで、組織を避けていない。R2では白色域の約1/33に下がる。
  - このHE指標はmoving点を使っていない。そのため、点照合とは独立にA2の位置を支持する。
- 【限界】白色＝真の無組織とは限らない（ガラス・裂隙・管腔・薄い染色を含む）。

## Q2. How much does ICP add after A2?（R1a → R2s → R2、paired）

### 変換の違い

**変位量であり、誤差ではない。**

| | R1a（NCC） | R2s | R2 | R1a→R2 |
|---|---:|---:|---:|---:|
| scale（√\|det\|） | 1.1015 | 1.1071 | 1.1072 | **+0.52%**（縮小ではない） |
| 回転（chirality補正後） | 2.00° | 1.80° | 1.80° | −0.20° |
| 異方性 / shear | 1 / 0 | 1 / 0 | 1.0007 / −0.0002 | ほぼ0 |
| 画像中心の位置 | (505.5, 462.3) | (506.7, 463.8) | (506.7, 463.8) | (1.2, 1.5) µm |
| moving核の変位 median / p95 / max | | | | 2.70 / 5.18 / 5.95 µm |
| R2s→R2（affine ICP段）の変位 median / p95 / max | | | | **0.13 / 0.21 / 0.25 µm** |

**ICPの改善はほぼすべてsimilarity段による。** 後段のaffine ICPは点を約0.13 µm動かすだけで、どの指標もほとんど変わらない。

- R2s→R2：≤3 µm 0.783→0.785、held-out 0.797→0.796、B2 1,320→1,324。
- 100 µm blockで±0.05を超えて変化したのは、改善2・悪化2のみ。

NCCの粗さ（4 µm raster、回転1°刻み、scale ±3%刻み）による2〜5 µmの誤差を、ICPが補正している。

### 指標別の改善・悪化

詳細は `icp_incremental_effect.csv`。

| 区分 | 指標 | R1a → R2 | 判定 |
|---|---|---|---|
| in-sample（**ICPに有利**） | ≤1 / 2 / 3 / 5 / 10 µm | 0.065→0.394 / 0.244→0.692 / 0.447→0.785 / 0.757→0.837 / 0.897→0.899 | 改善（10 µmでは差なし） |
| in-sample | Symmetric NN median / p90 / p95 | 3.38→1.32 / 11.1→11.0 / 16.3→16.1 µm | medianのみ改善（tailは非共有核で決まる） |
| in-sample | ICP objective（**ICP自身の目的関数**） | 2.53→0.98 µm | 改善（ただしICPに有利な指標のため証拠としない） |
| **held-out**（2 fold） | ≤3 µm | 0.580→0.798 / 0.190→0.794 | **改善** |
| 点群（NN非依存） | 局所correlogram residual 200 / 100 µm | 2.72→0.22 / 2.93→0.37 µm | 改善 |
| 点群 | peak at ≤3 µmのblock割合 | 0.52→1.00 | 改善 |
| 点群 | 密度map相関 σ5 / σ10 / σ20 | 0.681→0.740 / 0.746→0.758 / 0.810→0.806 | 細かいscaleのみ改善、σ20では差なし |
| 点群 | Regional NCC best / 境界到達率 | 0.786→0.786 / 3.7%→2.5% | ほぼ不変（±18窓より誤差が小さいため） |
| **HE画像**（独立） | 白色投影率 | 2.0%→0.22% | 改善 |
| HE画像 | 白色内縁0–5 µmの核密度 | 0.30→0.03 | 改善 |
| HE画像 | hematoxylin最大となるoffset | 2.2 µm→0 | 改善 |
| HE画像 | hematoxylin–fixed密度相関（σ10） | 0.145→0.150 | ほぼ不変 |
| 下流 | Graph support median | 0.815→0.821 | 小さい |
| 下流 | 候補距離median | 3.44→1.33 µm | 改善 |
| 下流 | UOT row mass | 0.506→0.505 | 不変 |
| 下流 | B2 matched / ambiguous / unmatched | 1,217/217/317 → 1,324/132/295 | 採択+107 |
| 局所region（100 µm block、95個） | ≤3 µm率 | 改善75・悪化3・不変17 | 大半で改善 |

- **悪化**：Realでは、ICPで系統的に悪化した指標はない。null指標は全stageで約0.05のまま。
- **Graph supportの変化は小さい**（0.815→0.821）。R1aの2〜3 µmの誤差は、Graphの相対配置の許容範囲内にあるため。
- **B2採択の増加（+107）はICP変位量と連動している。**
  - 増分は、ICP変位の大きいblockに集中している（Spearman 0.61）。R1aの局所近接率が低かったblock（−0.64）とも対応する。
  - 増加blockは47、減少8、不変40。Giniは0.64で、上位10%のblockが正の増分の36%を占める。組織全体に一様に広がってはいない（`accepted_matches_spatial_map.png`）。
  - R1aとR2の両方でmatchedになった1,193核では、99.8%が同じpartnerだった。ICPはpartnerを入れ替えていない。ICPの効果は、距離由来のconfidenceを閾値の上へ押し上げることである。**これは正しさの証明ではない。**

## Q3. Is ICP necessary?

**Realでは必要（主にsimilarity段）。**

- ICPがないと、held-outの≤3 µm率は0.385で、ICPありの約半分になる（2 foldとも）。
- B2の採択は107核少なくなる。
- 後段のaffine ICPは、Realではほぼ寄与しない（0.13 µm）。

**Synthetic Test**（未使用seedの新規56 pair、7条件×4族×2 rep、`synthetic_truth_comparison.csv`）。改善・悪化の判定は、common-region RMSが±0.5 µmを超えて変化したかで行った。

| 条件 | R1a RMS（median） | R2s | R2 | ICPで改善 / 悪化 | affine段で悪化（R2 > R2s） | 下流 B2 F1：R1a / R2 / R3（真Affine） |
|---|---:|---:|---:|---|---:|---|
| Global affineのみ | 3.05 µm | 0.13 | 0.07 | **8 / 0** | 0/8 | 0.892 / 0.960 / 0.960 |
| 軽度非剛体（3 µm） | 1.75 | 0.39 | 0.67 | 5 / 0 | 1/8 | 0.869 / 0.889 / 0.890 |
| 中等度（6 µm） | 3.96 | 1.19 | 2.86 | 6 / 1 | **8/8** | 0.674 / 0.737 / 0.703 |
| 強度（12 µm） | 4.01 | 5.12 | 6.71 | 1 / **6** | **8/8** | 0.375 / 0.455 / 0.392 |
| 大きい視野差（T5包含 / T6小fixed） | 3.10 | 104 | **106** | 2 / **14**（16中） | 11/16 | 0.826 / **0.000** / 0.906 |
| 低共有率（q 0.30） | 3.00 | 0.65 | 0.59 | **8 / 0** | 0/8 | 0.540 / 0.681 / 0.685 |

**ICPが不要な条件**：該当なし。

- 視野差以外のすべての条件で、NCCだけでは2〜4 µmの誤差が残る。
- 下流のB2 F1は、ICPを入れると同等以上になる。

**ICPが必要な条件**：global Affine・低共有率・軽度変形。Realはこの型に該当する。

**ICPが逆効果になる条件**：

- **大きい視野差**：NCC初期値は正しかった（R1a 3.1 µm、10/16がAffine成功）。しかしICPが13/16 caseで縮小方向に崩れた（scale −13〜−53%、変位72〜223 µm）。下流はすべて失敗した。
  - 2026-10-15外部検証と同じ機序。
  - Realでは、ICPはscaleを**+0.52%**変え、変位は最大6 µmだった。この崩れは起きていない。
- **中〜強度の変形**：affine ICP段が、真のglobal Affineからさらに離れる（8/8）。
  - ただし、変形のうち共通領域でaffine的な成分を吸収した結果である。下流のB2 F1は、R2の方が真のAffineより高い（0.737 対 0.703、0.455 対 0.392）。「真のAffineからの距離」と「局所対応のしやすさ」は一致しない。

**R1bの注意（DevとTest）**：R1aとR1bが異なるcaseでは、R1b（ICP objectiveで選んだ仮説のICP前Affine）の方が真値から遠い。

- Devでは、≤5 µmだったのがR1a 80/112、R1b 60/112。
- ICP objectiveによる選択は「ICP後に最も良くなる仮説」を選ぶ。最も正確な初期値を選ぶわけではない。
- R1bは「ICPなしの結果」ではない。

**Safeguardの候補**（事後の記述であり、採否は未検証）：

- TestでICPが崩れた13 caseでは、R1a→R2のscale変化が−12.8〜−53%、画像格子の変位medianが72〜223 µmだった。
- Realは+0.52%・2.7 µm。
- 「NCC解とICP解の差（特にscaleの縮小）」を監視する手がかりになり得る。閾値は、新しい独立データで事前に決める必要がある。

## Q4. Does Real show evidence of non-rigid deformation?

**検出可能な水準（RMS 0.5 µm以上、相関長約100〜200 µm）では、空間的に一貫した残差はない。**

事前固定の判定規則（`experiment_protocol.md` §6）の4条件は、いずれも満たさなかった。

| 診断 | Real R2 | 対照 / 校正 |
|---|---|---|
| Proxy残差（mutual NN ≤4 µm、1,310対）の大きさ | median 1.03 µm、p90 2.29 µm | 一貫成分のない擬似データ（N3、振幅0）で1.03 µm |
| ベクトル自己相関 ρ（0–20 / 20–50 / 50–100 / 100–200 µm） | 0.054 / 0.052 / 0.020 / 0.008 | N3振幅0のp95：0.035。振幅0.5 µmで0.14、1 µmで0.37 |
| 同（参考） | | R1a（affine誤差＝陽性対照）：0.57 / 0.57 / 0.52 / 0.44 |
| **Held-out skill**（100 µm checker、2色） | **0.002 / −0.036** | permutation p95：0.002 / −0.041。N3 null p95：−0.005。**振幅0.5 µmの5パーセンタイル：0.064** |
| Held-out skill（200 µm checker） | −0.031 / −0.031 | N3 null p95：−0.011 |
| 窓平均残差（50 / 100 / 200 µm）と、iidのnoise floor | 0.57 / 0.41 / 0.25 µm（floor 0.63 / 0.34 / 0.17） | noise floorと同程度 |
| 局所correlogram窓（NN非依存、±15 µm）の残差median | 50 µm：0.67、100 µm：0.43、200 µm：0.24 µm | 隣接窓の相関 ρ ≤0.06（100 / 200 µm） |
| B2 matched対のベクトル（source G） | ρ(0–20)=0.068、skill −0.022 | proxyと同じ結論 |
| HE画像の3×3領域ごとの最適offset（moving点を使わない） | hematoxylin最大のoffsetは9領域すべてで\|offset\| ≤1.4 µm | R1aでは1〜5 µmの系統的なずれを検出した（陽性対照） |

上表のHE指標について：白色投影率最小のoffsetは、白色域の少ない領域では平坦（plateau）になり不安定である。例として2領域で約(−11, −12) µmを示した。そのため、hematoxylin指標を主に用いた。

**検出力の確認**

- **Real-template semi-synthetic（N3）**：既知の滑らかな変位を加えると、0.5 µm RMSでheld-out検定が20/20で検出した（検出下限 = 0.5 µm）。Real R2のskillはこの下限より下にある。
- **独立generator（Test T1–T4、真の変形場あり）**：
  - 3 µmの変形は、真のAffine（R3）・A2（R2）のどちらでも8/8検出した。proxyベクトルと真の場の相関は0.74–0.78。
  - 変形なしでは、R3で誤検出0/8、R2で1/8だった。
  - 12 µmでは、proxyの4 µm打ち切りのためskillが低下する（検出6/8・5/8）。
  - Realのproxy残差はmedian 1 µmと小さく、NN非依存のcorrelogram窓でも同じ結論になる。Realではこの飽和は問題にならない。

**ただし、ごく弱い短距離相関が残る。**

- ρ(0–20 µm) = 0.054 で、null p95の0.035を上回る。
- 共分散から換算すると約0.33 µm RMS（null p95は0.23 µm）。
- held-outでは再現しない。
- 原因候補：近接核の重心差、密な領域でのNN切替、0.5 µm未満の局所変形。区別はできない。

**C（領域依存）は大きさだけに見られ、方向の一貫性ではない。** 次の領域で、残差の大きさがやや大きい（`spatial_residual_vectors.png`、`diagnostic_summary.json` relations）。

| 領域 | median | 2 µm超の割合 |
|---|---:|---:|
| HE白色境界の外側0–5 µm | 1.27 µm | 23% |
| 同 15 µm以上離れた場所 | 0.95〜1.0 µm | 10% |
| 高密度（kNN6距離が短い四分位） | 1.15 µm | — |
| 低密度 | 0.93 µm | — |
| ROI境界から50 µm以内 | 1.2 µm | — |
| それ以外 | 1.0 µm | — |

裂隙の縁・密集部・視野縁では、核同定と重心の不確かさが大きくなる。この観察と整合する。

## Q5. Are residuals caused by incorrect identity matches?

### Measured（実際に計算した値）

- R2のproxy pair（mutual NN ≤4 µm）は、COMMON内moving核の81%にある。
- 無関係なpattern（R2姿勢でmovingをmirror-x）でも、偶然のproxy pairが約9%の核にできる。
- B2 matched 1,324のうち94%がproxy pairを持ち、そのうち99.5%は同じfixed核を指す。
- 残差はB2 confidenceが低いほど大きい：conf ≥0.95ではmedian 0.86 µm、<0.9では約2.0 µm。
  - ただしconfidenceは距離を使って計算されるため、独立の証拠ではない。
- Graph supportとの関係は弱い（Spearman −0.14）。

### Supported inference（複数の診断が支持する解釈）

- R2後の残差は、空間的に一貫しない約1 µmのばらつきである。大きさは、共有核の切片間重心差（各軸 σ ≈ 0.9 µm）のモデルと一致する。
- 誤対応の上限は約1割と考えられる（null pairの頻度）。誤対応は、境界・密集部の大きめの残差に混ざっている可能性が高い。
- Syntheticでは、同じproxy定義の真ID率が以下のとおりだった。Realの「残差約1 µm、一貫成分なし」はaffine-only〜軽度変形に近い。
  - Affineのみ：0.999
  - 軽度変形：0.96
  - 中等度：0.84
  - 強度：0.58

### Unverified（真の対応IDや独立ランドマークなしには検証できない事項）

- 個々のpairが同じ細胞かどうか
- B2 matched 1,324件の正答率
- 2 µmを超える残差のうち、どれが誤対応でどれが真の局所ずれか

撮像範囲metadataは再確認したが存在しない（`input/metadata.json` はwarp後のcanvasだけ）。HE視野と蛍光視野は同じとも異なるとも仮定していない。HE footprint中心とfixed重心の差（R2で37 µm）は観察値として記録した。

## Q6. What should we develop next?

- **主結論 B**：Real上では、A2が位置の大きな誤り（約50 µm）を直し、ICPのsimilarity段が精度（約2.7 µm → 約1 µm）を出している。両方が必要である。
- **副結論 C**：ICP（特に視野差のある条件）にはsafeguardが必要。
  - 候補：NCC解に対するscale変化の監視、またはscale拘束。
  - Realは単一pairで、その条件に当たらなかっただけである。撮像範囲metadataがないため、事前には分からない。
  - 採否は新しい独立データで判定すること（今回のTest 56件を見て候補を挙げたので、このTestを採否の検証に再利用しない）。
- **副結論 E**：Affine後のReal残差は、約1 µmの細胞単位の不確かさが支配的である。非剛体refinementを入れても、検出可能な一貫成分はない。過適合の恐れの方が大きい。
- **副結論 F**：少数の独立landmark（両画像で同定できる核・血管）と、Real pairの追加が必要である。B2採択の正しさと、0.5 µm未満の局所変形の有無を決めるため。
- **D（非剛体）は棄却**：0.5 µm RMS以上の一貫成分は、校正済みの検出力で否定された。
  - 未検討の範囲：0.5 µm未満の変形、50 µmより短い相関長の変形、COMMON外（視野縁）の変形。
- **A（A2単独で十分）も棄却**：held-outで約半分の精度にとどまり、B2採択も減る。affine ICP段（R2s→R2）の省略だけは、Realでは差がない（0.13 µm）。

---

## 方法の要点

### Stage

- `stages.py`：凍結 `init_lib._finish` を実行時だけwrapし、ICPに渡る仮説リストを記録する（fileは変更しない）。
  - R2は既存A2と完全一致した（Real、Dev 112件ともmax差0.0）。
  - test：`test_study.py`、10 passed。

### 評価集合

| 集合 | 内容 |
|---|---|
| ALL | ROIなし |
| COMMON | 全stageで有効な共通核（主） |
| STAGE-ROI | production subset |

主要値は3集合で同じ傾向だった（`real_stage_comparison.csv`）。

### Held-out

- ICPに有利にならないよう、checkerboard 100 pxの2 foldを主とした（事前固定）。
- 半画像fold（x / y）では、A2がNCC段で失敗した。≤3 µm率は0.04〜0.10でnull水準である。
  - movingの半分だけだと点群重心が探索範囲外にずれる（既知のcrop stressで、2026-10-14と同じ）。
  - ICPの評価には使っていない。

### 非剛体の診断

- 残差source：proxy pair、correlogram窓、B2 matchedの3種類
- 統計：自己相関、方向一貫性、多スケール窓平均、held-out予測
- null：N1 permutation、N2 mirror、N3 semi-synthetic
- 校正：Real templateにGRF 0〜4 µmを加え各20 rep、さらに独立generator T1–T4
- held-out予測は検定統計量であり、点の補正・変形場の保存はしていない。

### Synthetic

- **Test**：新規seed（`IV_*`）の56 pair。
- **Dev**：既存112件。記述のみで、未見扱いしていない。
- 下流：R1a / R1b / R2 / 真Affineを凍結Graph/UOT/B2に渡した。R1a = R1bの場合は再計算せず複製した。

## 計画からの変更・注記

1. **`run_study.py`の修正（評価コードのみ）**：初回の `real_residual` 実行が、relations集計（kNN分位のbin端 > 1）で停止した。修正して再実行した（job 665756）。残差の数値は初回logと同一。plan hashが変わったのはこの1か所。
2. **図の修正**：`aggregate.py` の図のerror bar形状を修正した（集計のみ）。
3. **事後の記述**：Q3のsafeguard候補（scale変化・変位）は事後の記述統計であり、事前の判定規則ではない。
4. **R0のHE translation scan**：探索範囲（±30 µm）が、R0の約50 µmのずれより狭い。R0のHE offset値（26 µm）は真のずれを表さない。R0の評価には、白色投影率（面積比と同等）を使った。
5. **Regional NCC**：各stageのproduction subset（STAGE-ROI）で計算した。
6. **SyntheticのRMSの解釈**：変形条件では、fitしたAffineが共通領域の変形のaffine的成分を吸収する。そのためR2 vs R3のRMSは「純粋な誤差」ではない（Q3参照）。

## 限界

- Realは1 pairで、真の対応ID・独立landmarkはない。
- Real上の差（例：held-out ≤3 µm 0.385 → 0.796）は、このpair・この核検出の上での値である。
- Syntheticの実効独立単位は4族しかない。
- 変形の検出下限0.5 µmは、相関長110–200 µmのGRFに対する値である。

## 成果物

### CSV / JSON

- `real_stage_comparison.csv`
- `affine_parameter_comparison.csv`（変換parameter、変位、held-out foldのfit結果）
- `icp_incremental_effect.csv`
- `spatial_region_residuals.csv`（100 µm block）
- `downstream_graph_comparison.csv`
- `synthetic_truth_comparison.csv`（Test＋Dev）
- `diagnostic_summary.json`（判定規則と結果、校正表、関係表を含む）

### 図

- `real_registration_stage_overlay.png`
- `nn_distance_by_stage.png`
- `icp_before_after_residual.png`
- `spatial_residual_vectors.png`
- `residual_spatial_autocorrelation.png`
- `graph_support_by_stage.png`
- `accepted_matches_spatial_map.png`
- `synthetic_affine_vs_nonrigid_error.png`

### コード

- `stages.py`, `metrics.py`, `nonrigid.py`, `run_study.py`, `aggregate.py`, `test_study.py`, `study.sbatch`
- 事前固定：`experiment_protocol.md`
- 中間結果：`cache/`
- log：`logs/`
- 凍結・hash・job ID：`freeze/`

### 再現手順

1. `sbatch study.sbatch <mode>` を各modeで実行する（`freeze/job_ids.txt` を参照）。
2. `python -B aggregate.py`
