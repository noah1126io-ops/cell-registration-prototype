# UOT Adoption Validation for Workflow C

実施日: 2026-09-11 JST  
対象branch: `feature/gpu-slurm`  
対象HEAD: `f66ecbe5bfb691dd18695d34702a3fd13c3dcf7a`

## Executive summary

結論は **B. UOT PROMISING, MORE VALIDATION NEEDED**。ただし、採用候補として支持されるのは **UOT conservative** であり、UOT primaryではない。

前回UOT primary（symmetric median NN 8.3365 µm）とconservative（8.6405 µm）は同じreal dataset上で主要値を数値誤差 `1e-10` 未満で再現した。両者ともJoint（9.4134 µm）よりNN proxyが良く、common auditはPASSだった。

一方、サーバー上で条件を満たす独立real serial-section datasetは1つしか確認できなかった。このためreal-data generalizationは未検証であり、分類Aは選べない。

known-ground-truth synthetic 50 scenariosではUOT conservativeがJointより安全かつ正確だった。PASS率94% vs 82%、tissue-domain field endpoint error（EPE）中央値2.2889 vs 2.9234 µm、独立landmark TRE 2.2557 vs 3.0560 µmで、EPEは46勝3敗1同等だった。ただしunmatched点の移動は0.9901 vs 0.9405 µmでわずかに大きく、missingness増加に伴う相対優位の単調増加はなかった。

UOT primaryはsynthetic PASS率64%、attempted foldover 6/50、NNがJointより良くてもfield errorが悪いcaseが19/50だった。これは「real datasetで最も良いNN」をそのまま一般化できない直接的な証拠である。

現行architectureは **Joint維持**。Workflow C本体、Streamlit UI、Workflow D、safety thresholdは変更していない。

## Validation question

固定済みUOTを次期point-only registration methodとして採用できるかを、次の4点で判定した。

1. 前回改善の再現性
2. NN proxyへの自己適合の有無
3. Joint Flowに対する安全性・ground-truth accuracy・汎化性
4. Workflow Cをpoint-onlyへ簡略化する根拠の十分性

結果を見る前に [validation_plan.json](validation_plan.json) へ方法、synthetic条件、seed、比較規則を固定した。parameter sweep、dataset別parameter変更、結果を見た後のretuningは行っていない。

## Fixed UOT configurations

source of truthは、指定pathが対象worktree内ではなくhome直下に存在したため、以下を読み取り専用で使用した。

- `/mnt/mib-dbia08_1/users/noa.ganeko/analysis/cpd-uot-comparison-20260909/REPORT.md`
- 同 `uot_trials.csv`
- 同 `trials/uot/coarse_04/result.json`
- 同 `trials/uot/coarse_09/result.json`

| Parameter | UOT conservative | UOT primary |
|---|---:|---:|
| Source trial | coarse_09 / cost10_e0.1_m0.5 | coarse_04 / core_e0.1_m0.5 |
| Cost | squared Euclidean / `(2 × scale²)` | same |
| Cost scale | 10 µm | 20 µm |
| Entropy regularization | 0.1 | 0.1 |
| KL mass relaxation | 0.5 | 0.5 |
| Solver | translation-invariant unbalanced Sinkhorn | same |
| Solver limit / tolerance | 2000 / `1e-7` | same |
| Row-mass discard | ratio `<0.25` | same |
| Confidence | `min(row_mass/source_mass, 1)` | same |
| Field | SciPy local linear RBF | same |
| Smoothing / neighbors | 10 / 40 | 10 / 40 |

推測値は用いていない。verbatim source recordと実行値は [fixed_configurations.json](fixed_configurations.json) に保存した。

real reproductionにおけるUOT diagnostics:

| Method | Transported mass | Anchors retained | Discarded | Retained fraction | Retained confidence median (p05–p95) |
|---|---:|---:|---:|---:|---:|
| UOT conservative | 0.271274 | 982 | 860 | 53.31% | 0.3997 (0.2693–0.6189) |
| UOT primary | 0.410597 | 1,558 | 284 | 84.58% | 0.4458 (0.2812–0.6757) |

## Fixed Joint configuration

source of truthはtrial 0005 artifactと `/mnt/mib-dbia08_1/users/noa.ganeko/analysis/density-vs-joint-20260909/joint_exact_config.json`。既知値と一致した。特に記載省略されがちなStage B supportのeffective weightは `0.15 × 0.70 = 0.105` である。

| Stage | Scales (µm) | Iterations | LR | Update smoothing | Density | Support | Structure |
|---|---|---:|---:|---:|---:|---:|---:|
| A | 32, 16, 8 | 8 | 0.05 | 6 | 0.35 | 0.70 | 0 |
| B | 8, 4 | 10 | 0.05 | 3 | 1.0 | 0.105 | 0.35 |

Common: soft Jacobian weight 0.05、exploratory max/p95 displacement 35/25 µm。最終common thresholdsは変更せず、Jacobian hard 0.1–3.0、p05≥0.8、p95≤1.25、max displacement≤35 µm、p95≤30 µm、foldover=0、finite、および既存point-improvement/degradation guardsを使用した。

Density Flow baselineも前回の `density_effective_parameters.json` をそのまま固定した。

## Real datasets

home以下の保存済みWorkflow C inputs、runs、sweeps、analysisを調査し、fixed/moving配列をSHA-256でdeduplicateした。保存済みreal runs/sweepsはすべて同じhashの1 pairだった。小型の別hashはsynthetic smoke testであり、real datasetに数えていない。

| Dataset | Fixed n | Moving n | Usable | Reason |
|---|---:|---:|---|---|
| serial_section_pair_20260904 | 1,979 (eval 1,830) | 1,842 (eval 1,754) | yes | HE nuclei、fluorescence/GeoJSON nuclei、座標・Affine・共通評価subsetあり |
| synthetic_smoke_20260904 | 80 (eval 65) | 80 (eval 66) | no | engineering syntheticであり独立real serial sectionではない |

詳細hashは [dataset_inventory.csv](dataset_inventory.csv)。usable real dataset countは **1**。以下のreal結果をdataset間robustnessとは一般化しない。

全methodは同じsaved Affine points、fixed points、filters、coordinate system、evaluation subset、bounds、grid、safety thresholdsから開始した。Affine再推定は行っていない。

## Real-data results

NN値はpoint-set proximity proxyであり、**ground-truth correspondence accuracyではない**。

| Metric | Affine | Density | UOT conservative | UOT primary | Joint |
|---|---:|---:|---:|---:|---:|
| Symmetric median NN (µm) | 10.0237 | 9.8238 | 8.6405 | 8.3365 | 9.4134 |
| Moving→fixed median NN (µm) | 9.8950 | 9.6586 | 8.6361 | 8.3136 | 9.0886 |
| Fixed→moving median NN (µm) | 10.1524 | 9.9890 | 8.6448 | 8.3595 | 9.7382 |
| Within 3 µm | 7.0344% | 8.0678% | 10.9151% | 9.6025% | 8.8221% |
| Within 5 µm | 16.7807% | 18.2322% | 24.4553% | 22.3659% | 20.3834% |
| Within 10 µm | 49.7719% | 51.4526% | 57.0320% | 61.1598% | 53.7035% |
| Mutual NN | 53.6272% | 53.5714% | 55.4129% | 58.3705% | 53.4040% |
| Displacement median (µm) | 0 | 0.2710 | 1.1790 | 2.1095 | 0.8914 |
| Displacement p95 (µm) | 0 | 0.5817 | 2.3929 | 4.8705 | 1.9277 |
| Displacement max (µm) | 0 | 1.0175 | 3.8545 | 7.8942 | 4.1617 |
| Jacobian min | 1.0000 | 0.8542 | 0.7253 | 0.3512 | 0.6130 |
| Jacobian p05 | 1.0000 | 0.9399 | 0.9173 | 0.8296 | 0.8344 |
| Jacobian median | 1.0000 | 1.0000 | 0.9941 | 0.9795 | 0.9884 |
| Jacobian p95 | 1.0000 | 1.0608 | 1.0824 | 1.1689 | 1.1791 |
| Jacobian max | 1.0000 | 1.1753 | 1.3240 | 1.7867 | 1.7017 |
| Foldover | 0 | 0 | 0 | 0 | 0 |
| Common audit | PASS | PASS | PASS | PASS | PASS |

UOT primaryとconservativeのsymmetric NN、displacement p95、Jacobian p05は全て前回sourceと差 `≤1.7e-14` で再現した。詳細は [real_reproduction_check.json](real_reproduction_check.json)。

## Safety / failure rates

Real n=1では全method 1/1 PASS。したがってreal PASS率100%という値はrobustness estimateではない。best/worst/variance/IQRは [real_dataset_summary.csv](real_dataset_summary.csv) に出したが、n=1のためvariance/IQRが0なのは機械的結果である。

Syntheticの最終適用出力:

| Method | PASS | Failures | Applied foldover cases | Applied p95 disp median / worst (µm) | Applied Jacobian min worst |
|---|---:|---:|---:|---:|---:|
| Density | 94% | 3/50 | 0 | 0.6754 / 2.1535 | 0.7637 |
| UOT conservative | 94% | 3/50 | 0 | 2.9427 / 5.2963 | 0.2018 |
| UOT primary | 64% | 18/50 | 0 | 4.0808 / 12.2961 | 0.2821 |
| Joint | 82% | 9/50 | 0 | 2.4520 / 3.2199 | 0.6765 |

Rejection時はAffine fieldへ戻すため、適用出力のfoldoverは0。attempted candidateではUOT conservative 1/50、UOT primary 6/50にfoldoverがあり、worst attempted Jacobian minはそれぞれ -0.1106 / -1.1475だった。Joint attempted foldoverは0。これはsafety gateを維持すべき根拠である。

主failure modeはDensityの`no_improving_safe_checkpoint` 3件、Jointの`no_joint_checkpoint_passed_final_application_safety` 9件、UOTのJacobian hard/percentile/foldover gateだった。実行例外は0件。

## Synthetic benchmark

1 seedあたり重複除去済み10 scenarios、5 seeds（1101–1105）、5 methods、計250 outputsをGPU Slurm job 213211で実行した。runtimeは共有GPU条件のためformal benchmarkとして解釈しない。

生成条件:

- 128×128 grid、2 µm spacing、known smooth nonlinear field
- mixture＋uniform samplingによるspatially nonuniform point density
- independent random moving deletion、spatial partial overlap、moving-only unmatched additions
- fixed-only section-specific additionsによるunequal counts
- coordinate noise
- 登録入力に使わない30 known landmarks
- nontrivial known affine matrix/translationをraw movingへ逆適用後、全fine methodの前に同一Affineを正確に復元

known affine reconstruction max errorは `6.36e-14 µm`。difficultyはEasy `(4 µm, missing 10%, unmatched 5%, overlap 100%, noise 0.25 µm)`、Medium `(8, 30%, 15%, 85%, 0.75)`、Hard `(12, 50%, 30%, 70%, 1.25)`。加えてmissingness 0–50%とamplitude 4/8/12 µmのcontrolled sensitivityを実施した。

field errorは既知fixed tissue support内で評価し、backgroundによるidentity有利を避けた。棄却methodは実際の適用結果と同様にAffine fallbackを評価した。attempted field error/safetyも別列に保持した。

| Method | PASS | Field EPE median (µm) | Field EPE p95 median | Endpoint mean median | Independent TRE median (µm) | Known matched TRE median | Jacobian abs. error median | Unmatched movement median (µm) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Affine | 100% | 2.9234 | 6.4333 | 3.1732 | 3.0560 | 3.6897 | 0.0378 | 0 |
| Density | 94% | 2.7239 | 6.0166 | 2.9425 | 2.7749 | 3.3435 | 0.0375 | 0.3618 |
| UOT conservative | 94% | **2.2889** | **5.4142** | **2.5066** | **2.2557** | **2.7703** | 0.0442 | 0.9901 |
| UOT primary | 64% | 2.9234 | 6.4333 | 3.1732 | 3.0560 | 3.4086 | 0.0565 | 1.2684 |
| Joint | 82% | 2.9234 | 5.5969 | 3.0122 | 3.0560 | 3.4231 | 0.0638 | 0.9405 |

UOT conservativeのworst-case EPEは4.7321 µm、IQR 0.8549 µm。Jointはworst 4.5306 µm、IQR 0.2848 µmで、UOT conservativeは中央値で優れる一方、ばらつきとworst caseはJointより大きい。

## Missingness sensitivity

amplitude 8 µm、unmatched 15%、overlap 85%、noise 0.75 µmを固定し、missingnessのみ0–50%へ変更した。

| Missing | Density EPE / PASS | UOT cons. EPE / PASS | UOT primary EPE / PASS | Joint EPE / PASS |
|---:|---:|---:|---:|---:|
| 0% | 2.6467 / 100% | **2.2241 / 100%** | 2.9234 / 40% | 2.7696 / 100% |
| 10% | 2.6586 / 100% | **2.4247 / 60%** | 2.9234 / 20% | 2.8806 / 80% |
| 20% | 2.7369 / 100% | **2.2060 / 100%** | 3.0202 / 100% | 2.9094 / 80% |
| 30% | 2.7248 / 100% | **2.0622 / 100%** | 3.2797 / 80% | 2.9478 / 80% |
| 40% | 2.7743 / 80% | **2.5010 / 80%** | 2.9234 / 40% | 3.0100 / 80% |
| 50% | 2.7767 / 100% | **2.3464 / 100%** | 3.5910 / 60% | 2.9657 / 80% |

UOT conservativeは全levelでJointより低いmedian EPEだったが、advantageは単調増加せず、10%で2/5 rejectionが発生した。したがって「missingnessが高いほどUOT優位が増える」は支持されない。

unmatched movementではUOT conservativeが0–20% missingでJointより小さく、30%と50%では大きかった。overall差は+0.0495 µm。partial correspondence handlingによる一貫したoutlier抑制とは結論しない。

## Deformation sensitivity

missing 30%、unmatched 15%、overlap 85%、noise 0.75 µmを固定した。

| Amplitude | Density EPE / PASS | UOT cons. EPE / PASS | UOT primary EPE / PASS | Joint EPE / PASS |
|---:|---:|---:|---:|---:|
| 4 µm | 1.3684 / 100% | **1.2603 / 100%** | 2.2558 / 80% | 1.7001 / 80% |
| 8 µm | 2.7248 / 100% | **2.0622 / 100%** | 3.2797 / 80% | 2.9478 / 80% |
| 12 µm | 4.1178 / 100% | **3.1362 / 100%** | 4.3850 / 80% | 4.2138 / 100% |

UOT conservativeは3 amplitudesすべて5/5 PASSで、EPE/TREともJointより良かった。UOT primaryは各level 4/5 PASSに留まり、大変形での採用候補にならなかった。

## Independent evaluation

real datasetには独立landmark/ground truthがなかったため、real-data independent accuracyは未評価。代わりにsyntheticで次をNN以外に評価した。

- full known displacement field EPE（median/p95/max/mean endpoint error）
- registration objectiveに使わない30 landmarksのTRE
- known matched-point TREとpoint displacement error
- known fieldに対するJacobian determinant error
- unmatched点の不要移動量

UOT conservativeはJointに対してfield EPE -0.6345 µm、独立TRE -0.8003 µm、known matched TRE -0.6528 µm。UOT primaryはNNとground truthがdiscordantなcaseを19/50生じた。

real datasetでは両UOTとJointのposition差の最大値で順位付けした12領域を、65 µm以上離して抽出した。[manual_review_candidates/candidates.csv](manual_review_candidates/candidates.csv) と同ディレクトリの12 PNGは、後続のblind pathologist review用であり、今回の評価結果には使用していない。

## UOT vs Joint

Real datasetのdeltaは `UOT - Joint`。負のmedianはUOTが良い。

| Comparison | Median delta (µm) | Within5 delta | Within10 delta | Mutual delta | Disp p95 delta (µm) | Jacobian p05 delta | Jacobian min delta | PASS |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| UOT conservative − Joint | -0.7729 | +4.0719 pp | +3.3285 pp | +2.0089 pp | +0.4653 | +0.0829 | +0.1123 | PASS/PASS |
| UOT primary − Joint | -1.0769 | +1.9825 pp | +7.4563 pp | +4.9665 pp | +2.9428 | -0.0047 | -0.2618 | PASS/PASS |

real NN proxy wins/losses/tiesは、UOT conservative 1/0/0、UOT primary 1/0/0。ただしn=1である。

Synthetic field EPE wins/losses/tiesは、UOT conservative 46/3/1、UOT primary 17/29/4。UOT primaryはNNがJointより低いのにGT field errorが高いcaseが19/50あり、NN proxyだけなら誤選択することを示した。UOT conservativeの同discordanceは1/50。

## Failure cases

- **UOT primary:** 18/50 reject。主因はJacobian percentile/foldover gate。attempted displacement p95最悪12.2961 µm、Jacobian min最悪 -1.1475、attempted foldover 6件。
- **UOT conservative:** 3/50 reject。Jacobian hard failure 2件、foldoverを伴うfailure 1件。safety gate後の適用fieldにはfoldoverなし。
- **Joint:** 9/50 reject。final application safetyを満たすcheckpointなし。attempted foldoverなし。
- **Density:** 3/50 reject。improving safe checkpointなし。attempted foldoverなし。
- 実行例外、NaN/Inf、solver non-convergenceは0。

## Limitations

1. usable real datasetが1つしかなく、real-data robustness/generalizationを推定できない。
2. real independent landmarks、病理医blind review、image-derived independent metricは未実施。
3. syntheticはknown truthを持つが、serial-section biologyの全てを表現しない。特にsynthetic RGB nuclear appearanceは実HE stain variationの代替ではない。
4. unmatched点の「不要移動=0」はover-pull検出用の明示的synthetic定義であり、実組織の物理的ground truthではない。
5. UOT結果はUOT plan単独ではなく、mass filtering、confidence weighting、linear RBF smoothingを含む固定pipelineの評価である。
6. shared GPU runtimeはformal benchmarkではない。
7. UOT→Joint cascadeは実行しておらず、hybrid architectureの優位は未検証。

## Adoption criteria

| Pre-fixed criterion | UOT conservative assessment |
|---|---|
| 1. Safety PASS rate ≥ Joint | real同率、synthetic +12 pp。tested dataではmet |
| 2. Multiple real datasetsでNN同等以上 | **not met: n=1** |
| 3. Within5/10に大きな悪化なし | 1 real datasetでは改善。generalization不明 |
| 4. Mutual NNに大きな悪化なし | 1 real datasetでは改善。generalization不明 |
| 5. Synthetic GT field errorでJoint同等以上 | met（2.2889 < 2.9234 µm） |
| 6. Missing/unmatchedで安定 | EPEは全levelで優位、10% stratumとunmatched movementは不完全 |
| 7. Independent evaluationで悪化なし | syntheticではmet、realではunavailable |
| 8. Foldoverなし | applied outputはmet、attempted 1件をgateがreject |
| 9. 病理学的に不合理でない | displacementはthreshold内。pathologist確認なし |

## Adoption decision

**B. UOT PROMISING, MORE VALIDATION NEEDED**

UOT conservativeは、今回利用できた1 real datasetと広いsynthetic条件ではJoint以上の有力なpoint-only候補である。しかし「複数real datasets」と「real independent evaluation」が欠けるため、Workflow C simplificationの正式根拠としては不足する。UOT primaryはNN再現には成功したが、synthetic safety/ground truthで保守条件より劣り、採用しない。

## Recommended next step

現時点では **Joint維持**。Workflow C本体、UI、Workflow Dを変更しない。

次はUOT conservativeだけを同じ固定parameterで、少なくとも追加2 real serial-section pairsへ適用する。同時に今回抽出した12領域をblind reviewし、可能なら独立landmarkまたはpoint objectiveに使わないtissue metricを追加する。これらを満たした後にのみ、将来の `Affine → UOT conservative → confidence-gated smooth field → safety gate` 構成を再判定する。

## Artifacts

- [real_dataset_metrics.csv](real_dataset_metrics.csv)
- [real_dataset_pairwise_deltas.csv](real_dataset_pairwise_deltas.csv)
- [synthetic_metrics.csv](synthetic_metrics.csv)
- [synthetic_summary.csv](synthetic_summary.csv)
- [independent_validation.csv](independent_validation.csv)
- [synthetic_uot_joint_deltas.csv](synthetic_uot_joint_deltas.csv)
- [uot_adoption_decision.md](uot_adoption_decision.md)
- [real_dataset_comparison.png](real_dataset_comparison.png)
- [pass_rate_comparison.png](pass_rate_comparison.png)
- [synthetic_ground_truth_error.png](synthetic_ground_truth_error.png)
- [missingness_sensitivity.png](missingness_sensitivity.png)
- [accuracy_deformation_pareto.png](accuracy_deformation_pareto.png)
