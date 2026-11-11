# Affine Global Initialization：研究・比較実験

実施日: 2026-10-14 ／ branch `feature/gpu-slurm` @ `f66ecbe` ／ analysis専用。Production（Workflow C/D、UI、safety gate、ICP本体、Graph/UOT/B2、閾値）と既存analysis成果物は無変更（tracked 81ファイル・既存analysis 280ファイルのSHA-256を前後照合）。commit/pushなし、非剛体deformation fieldの生成なし、3D generatorの実装なし。

## 最終比較

| Method | Synthetic affine success | Wrong-basin rate | Real held-out alignment (≤3 µm) | Runtime (median / max) | Negative-control false positives |
|---|---:|---:|---:|---:|---:|
| Existing Centroid + ICP (A0) | 5/45 (11%) | 89% | 0.054（null 0.049） | 4.4 s / 5.8 s | 0/7 |
| Correlogram + ICP (A1) | 33/45 (73%) | 27% | 0.053（null 0.053）**Real失敗** | 127 s / 168 s | 0/7 |
| **Density correlation + ICP (A2)** | **42/45 (93%)** | **7%** | **0.796**（null 0.059） | 53 s / 98 s | 0/7 |
| Multi-start + ICP (A3) | 39/45 (87%) | 13% | **0.796**（null 0.059） | 125 s / 160 s | 0/7 |

- 指標の定義:
  - success: moving視野全体でのRMS変換誤差 ≤5 µmかつchiralityが正しい。
  - wrong basin: それ以外。中間（imprecise）は0件。
  - Real held-out: 100 px checkerboardの片色でfitし、もう片色の核を評価した2分割の平均（下記）。
  - runtime: 4 CPU、1 pairあたりの全探索＋refine時間。
- 失敗の残り方: A2の3件はすべて`partial_overlap_350`（下記の限界）。A3の6件は`partial_overlap_350`×3と回転6°×3。

**推奨: A2（point-density masked NCC）を主候補、A3を独立なcross-checkとし、事前規定の失敗検出器でgateする。** Production統合は別依頼。

---

## 1. 前回値の再確認

| 値 | 前回監査 | 今回 |
|---|---|---|
| 保存Affineの≤3 µm一致 | 7.2% | 7.18%（重なり域の全1,842核基準では7.2%） |
| 正しいbasinの≤3 µm一致 | 73.8%（subset） | 79.6%（重なり域） |
| ICP objective | 7.54 → 1.04 µm | 7.537 → 1.042 µm（A2/A3） |
| Graph support | 0.227 → 0.805 | 0.228 → 0.821（各methodのAffineにproduction同等のsubset規則を適用） |
| NCC ±18 境界到達率 | 58.4% → 4.8% | 57.8% → 2.5%（同上） |

A0（production `estimate_affine_with_y_flip`）の再推定は保存Affineとbit一致した（RMS差 0.0 µm）。前回の診断用再fitとの差は、A2/A3でRMS 0.0 µmだった。これは前回の再fitを入力に使った結果ではない。同じ初期化非依存のbasinに、同じproduction ICPが到達したため一致した。

## 2. 方法（初期化以外はすべて共通）

すべての候補を同じproduction refine（`icp_similarity` trim 0.8 → `affine_icp` trim 0.7、各80 iter）に通した。最終Affineは同じ2×2+並進モデルで、評価・subset規則も同一。Realのoffset (−7.6, −49.9) µmはどの方法にも与えていない。探索範囲は全方法共通の±200 µm（Synthetic設計の0〜150 µmを覆うよう設定し、Realに合わせた調整はしていない）。

| | 仮説生成 | 選択 |
|---|---|---|
| A0 | centroid一致＋RMS scale、4 flip × 30°刻み12回転（production） | production規則 |
| A1 | 2 chirality × 回転（粗4° → 細1°）× scale（RMS比・NN間隔比、細±3%）ごとに、pairwise変位ヒストグラムを作る。1 µm bin。σ15 µmの局所背景に対するz-scoreで局所密度・部分重なりの偏りを補正し、上位peakを複数保持 | 上位6仮説をrefineし、ICP objective最小を採用 |
| A2 | 同じ回転・scale格子で、点をGaussian（σ8 µm、4 µm px）にrasterizeし密度を正規化。両点群の凸包maskでPadfield masked NCCをFFTで計算（重なり<30%は除外＝空領域artifact対策）。peakは複数保持 | 同上 |
| A3 | 2 chirality × 15°回転 × scale × 並進格子（±200 µm、25 µm刻み）の最大27,744 startを、短いICP（4 iter、部分標本）で採点。上位12の異なる仮説をfull refine | 同上 |

## 3. Synthetic（真のAffineが既知）

自前の生成モデル（`init_lib.make_pair`）で検証した。結果はこの生成モデルに依存する。

- 組織: 2400 µm四方の潜在組織。滑らかな密度変動とGaussian bump、細長い無核の裂隙を含む。hard-core 4.5 µm。
- 切片間の核共有: 共有60%、各切片固有20%ずつ（Realと同程度の約70%がpartnerを持つ）。
- 検出・位置: 検出率 fixed 95% / moving 90%。滑らかな変形 3 µm、centroid差 0.8 µm。
- 視野: fixedは1010 µm視野。movingはHE pixel枠（901 px、1.1 µm/px、y反射＋回転）でcropし、視野中心をランダム方向へoffsetした。

15条件×3 seed（45 pair）:

| 条件 | A0 | A1 | A2 | A3 |
|---|---:|---:|---:|---:|
| offset 0 µm | 3/3 | 3/3 | 3/3 | 3/3 |
| offset 25 | 2/3 | 3/3 | 3/3 | 3/3 |
| offset 50 / 75 / 100 | 0/9 | 8/9 | 9/9 | 9/9 |
| offset 150（大） | 0/3 | 3/3 | 3/3 | 3/3 |
| 回転 3° / 6°（offset 50） | 0/6 | 4/6 | 6/6 | 3/6 |
| 回転90°・反射なし | 0/3 | 1/3 | 3/3 | 3/3 |
| 非対称crop（901×640 px） | 0/3 | 2/3 | 3/3 | 3/3 |
| 異なる撮像視野（640×640 px、offset 80） | 0/3 | 3/3 | 3/3 | 3/3 |
| 部分重なり（offset 350 µm） | 0/3 | 0/3 | 0/3 | 0/3 |
| 検出漏れ40% | 0/3 | 2/3 | 3/3 | 3/3 |
| 強い密度不均一＋裂隙 | 0/3 | 2/3 | 3/3 | 3/3 |
| 非共有核の増加（共有30%） | 0/3 | 2/3 | 3/3 | 3/3 |

成功例の変換RMS誤差の中央値は0.28〜0.37 µm（回転・scale誤差はほぼ0）。詳細は [synthetic_affine_accuracy.csv](synthetic_affine_accuracy.csv)、[synthetic_transform_error.png](synthetic_transform_error.png)。

**誤った局所解の性質**（[icp_convergence_basins.png](icp_convergence_basins.png)）:
- 部分重なり以外のwrong basin 49件では、到達objectiveが6.7〜9.3 µm、真の解から始めたICPのobjectiveが中央値2.2 µmだった。つまり**目的関数は正しい解を好んでいるが、初期値がbasin外にある**（初期化の失敗）。
- A0の捕獲範囲は約25 µmで、前回Realで測った約±30 µmと一致する。
- A0は180°回転の取り違えも頻発した（例: 平行移動誤差50 µmに加えて回転180°）。

**部分重なり（offset 350 µm）はすべての方法で失敗した。** 2つの限界が重なっている:
1. 探索範囲 ±200 µm を超えている。
2. 重なりが約65%しかなく、trim 0.7のICP objectiveでは**真の解の方が悪いobjective**（約10 µm）になる。

したがって、重なりの小さいpairはobjective最小では選べない。検出器は全件をFAILにしたので、誤採択はしていない。

## 4. Real（`serial_section_pair_20260904`）

| | A0 | A1 | A2 | A3 |
|---|---:|---:|---:|---:|
| ICP objective (µm) | 7.537 | 8.400 | **1.042** | **1.042** |
| ≤3 µm一致（重なり域、mirror null） | 0.072（0.054） | 0.056（0.054） | **0.796**（0.045） | **0.796**（0.045） |
| Checkerboard held-out（最小、null） | 0.045（0.049） | 0.051（0.053） | **0.794**（0.059） | **0.794**（0.059） |
| Held-out fitとfull fitのRMS差 | 16.4 µm | 758 µm | **0.08 µm** | **0.08 µm** |
| Local block consistency | 0/25 | 0/25 | **25/25** | **25/25** |
| HE白色域に落ちる蛍光核（面積比7.7%） | 6.4% | 6.3% | **0.26%** | **0.26%** |
| NCC ±18 境界到達率 | 57.8% | 65.1% | **2.5%** | **2.5%** |
| 保存Affineとの差 RMS | 0 | 481 µm | 52.0 µm | 52.0 µm |
| 検出器 | FAIL | FAIL | **PASS** | **PASS** |

A2の選択済み仮説: reflect、回転0°、translation offset (−8, −48) µm（粗探索）→ refine後に前回の再fitと同一。A2の粗探索では、正しいchirality・回転の仮説がNCC 0.46で、2位（0.38、隣の4°）と異なる向き（≤0.16）を大きく引き離した（[coarse_search_peaks.png](coarse_search_peaks.png)）。

**A1はRealで失敗した。**
- 粗段階（回転4°刻み、σ4 µm）では、細胞レベルのpeakが回転ずれでぼける。
- さらに、球近傍探索が作る円形サポートの縁で局所背景が過小評価され、zが膨らむ（z-mapの縁リング）。その結果、ノイズpeak（z約12）が正解を上回った。
- これはA1の**実装上の欠陥**であり、correlogramという考え方自体の否定ではない。結果を見た後にRealで修正・再調整はしていない。修正版を評価する場合は、Syntheticで事前に検証すべき。

**独立性に関する注意:**
- A1〜A3は仮説をICP objectiveで選んでいる。同じReal pair上のobjective・≤3 µm一致・held-out一致は互いに相関した指標であり、独立な外部検証ではない。
- 最も独立に近いのはHE画像の白色域チェック（点の照合に使っていない画像情報）。ただしこれも前回監査で用いた同じpairの指標である。
- 真のTREは独立landmarkがないため未測定。

Half-field splitはcrop stressとして別に記録した（[real_affine_comparison.csv](real_affine_comparison.csv) の `crop_stress_*`）。movingの片半分だけでfitすると、centroid一致が約250 µmずれて探索範囲±200 µmの外に出るため、全方法が失敗した（A2/A3でも≤3 µm一致は約0.06）。**探索範囲を超える視野差には、現行設定ではどの方法も対応できない。**

## 5. Negative controlと失敗検出

検出器は**Real・controlを見る前にコードで固定**した（`init_lib.DETECT`）:
- **PASS**: 次をすべて満たす。
  - ≤3 µm一致 ≥0.25
  - かつ mirror null比 ≥3
  - かつ 200 µm blockの≥60%で、局所変位peakが鋭く（背景比≥5）原点から6 µm以内
  - かつ（held-outがある場合）held-out一致 ≥0.20
  - かつ 別basinとのobjective差 ≥10%
- **AMBIGUOUS**: null比 ≥2 だがPASS条件の一部を欠く。
- **FAIL**: それ以外。

| Control | 内容 | PASS（偽陽性） |
|---|---|---:|
| NC1a | Real fixed vs 無関係なsynthetic moving | 0/4 |
| NC1b | 共有核のない独立切片（synthetic） | 0/4 |
| NC2a | Real、誤ったchiralityのみ探索 | 0/4 |
| NC2b | Synthetic、誤ったchiralityのみ探索 | 0/4 |
| NC3 | 重なりがごく小さい（offset 900 µm） | 0/4 |
| NC4 | Real fixedの密度だけを再現した独立点（共有核なし、Affine既知） | 0/4 |
| NC5 | 一様乱数のmoving | 0/4 |

- control全28件でICPは「収束」した（objective 6.2〜9.7 µm）。しかし検出器はすべてFAILだった。**最適化が成功したように見える誤ったAffine**（例: NC4のA2は組織スケールでは正しいAffineを回復したが、共有核がないためnull比1.5）を、細胞対応の根拠なしとして拒否できている。
- Synthetic 180件では、検出器のPASSと真の成功が完全に一致した（成功119件中PASS 119、失敗61件中PASS 0）。
- ただし閾値は今回の生成モデル上での結果であり、別のデータ族で校正し直す必要がある。

**提案する失敗判定**（production統合時）: 上記検出器がFAIL/AMBIGUOUSならAffineを採用せず、既存の保存結果も「未確認」として扱う。PASSでも、A2とA3の解が一致しない場合（RMS差 >5 µm）はAMBIGUOUSとする。

## 6. 下流確認（Graph/UOT/B2は凍結）

Candidate rule・Graph cost・UOT relaxation・B2 calibration・閾値は一切変えず、各AffineとProduction同等のsubset規則だけを入れ替えた。B2は保存済み学習結果から再fitし、前回Real結果（26/209/1519）の完全再現を確認済み。

Real（Position+Graph / B2）:

| | A0 | A1 | A2 = A3 |
|---|---:|---:|---:|
| Candidate distance 中央値 | 10.50 µm | 11.45 µm | **1.33 µm** |
| Graph support 中央値 | 0.228 | 0.178 | **0.821** |
| Full-UOT matched / B2 matched | 24 / 27 | 19 / 16 | 1,241 / 1,324 |
| Duplicate destinations（B0 / B1） | 16 / 730 | 14 / 895 | 98 / 450 |
| 前回診断再fitのpartnerとtop-1が一致（参考） | 0.1% | 0.2% | 98.4% |

**Realのmatched数の増加は精度向上の証拠ではない**（真値なし、B2はsynthetic校正）。B0の重複行き先は増えた（16→98）。これはmatched数が増えたことに伴うもので、それ自体の問題かは未確認。

Synthetic（真の対応あり、seed 11）:

| 条件 | A0 top-1正解率 / B2 precision | A2 top-1正解率 / B2 precision |
|---|---|---|
| offset 0 | 0.987 / 0.987 | 0.987 / 0.987 |
| offset 50 | 0.000 / — | **0.987 / 0.987** |
| offset 100 | 0.000 / — | **0.985 / 0.987** |
| 検出漏れ40% | 0.000 / — | **0.965 / 0.993** |
| 非共有核の増加 | 0.000 / — | **0.933 / 0.974** |
| 部分重なり350 | 0.000 / — | 0.003 / —（全方法失敗） |

Affineが正しいbasinに入れば、凍結済みGraph/UOT/B2は高精度で動作する。Affineが外れると、真の候補がcandidate setにほぼ入らない（recall@30 0.03〜0.30）。**Graph Matchingの低性能は上流のAffine失敗で説明できる**という前回の結論を、真値つきで確認した。

## 7. 質問への回答

1. **最も安定して真のAffineを回復した方法**: A2（42/45、失敗は探索範囲外の部分重なりのみ）。次点はA3（39/45、6°回転に弱い。15°刻みの回転格子と短いICPでは回転誤差を回収しきれない）。
2. **視野ずれ・cropへの頑健性**: A2/A3は0〜150 µmのoffset、非対称crop、異なる撮像視野、検出漏れ40%、密度不均一、共有率30%で成功した。A0は25 µmを超えると失敗した。共通の限界は、探索範囲±200 µmと、重なり約70%未満でobjectiveが誤った解を好むこと。
3. **誤った局所解の頻度**: A0 89%、A1 27%、A3 13%、A2 7%（Synthetic）。部分重なりを除くとA2 0%、A3 7%。
4. **Realの約50 µmずれの自動回復**: A2・A3とも可能。どちらも前回の再fitと同一の解に到達した。正解情報を与えずに到達し、checkerboard held-outで0.79、HE白色域0.26%、検出器PASS。A1は失敗した。
5. **計算コスト**: A2は中央値53秒・最大約67 MB（traced）、A3は約2分・約18 MB。いずれも4 CPUでの値。Workflow C 1回の実行に対して実用的な範囲。粗探索はFFTとKD木で済み、GPUは不要。
6. **下流Graph/UOTの挙動**: Real Graph support 0.228→0.821、候補距離 10.5→1.33 µm、NCC境界 58%→2.5%。Syntheticではtop-1正解率が0→0.93〜0.99に回復した。
7. **Production候補**: A2を主、A3を独立確認、事前規定の検出器でgateする構成。統合前に必要なこと:
   - (a) 別の生成族と追加のReal pairで再検証する
   - (b) 部分重なりに対応する探索範囲とoverlap-aware objectiveを別途検討する
   - (c) 検出器閾値を再校正する
   - (d) 少数の独立landmarkでTREを測る

## 8. 限界

- Syntheticは自作の単一生成族（2D、smooth変形、切片固有核は独立配置）に依存する。3D切片過程は再現していない。
- Realは1 pairのみ。held-outはfixed側を共有しており（fixedは標的として全点使用）、完全な独立分割ではない。
- 下流Syntheticは各条件1 seed。A1の欠陥は未修正。
- Repository testsのGPU付き再実行（前回監査のjob 662472）は、GPUの空き待ちで未完了。今回はtracked fileを変更していない。

## 成果物・再現

`REPORT.md`, `method_comparison.csv`, `synthetic_affine_accuracy.csv`, `real_affine_comparison.csv`, `negative_control_metrics.csv`, `downstream_graph_metrics.csv`, `runtime_metrics.csv`, `diagnostic_summary.json`。図は `coarse_search_peaks.png`, `initialization_comparison.png`, `icp_convergence_basins.png`, `synthetic_transform_error.png`, `real_before_after_overlay.png`, `graph_support_by_initialization.png`。

再現手順（Slurm `mib-dbia`、CPU）:
1. `sbatch -J syn --array=0-44 case.sbatch synthetic`
2. `sbatch -J ctl --array=0-6 case.sbatch control`
3. `sbatch -J real --array=0 case.sbatch real`
4. `sbatch -J realchk --array=0 case.sbatch real_checker`
5. `sbatch -J ds --array=0-28 case.sbatch downstream`
6. `python -B aggregate.py`

Tests: `pytest test_init.py`（6 passed）。
