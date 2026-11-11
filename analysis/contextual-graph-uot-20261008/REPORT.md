# Multi-scale Contextual Graph-UOT Matching

実施日: 2026-10-08 JST  
対象branch: `feature/gpu-slurm`  
目的: registration距離ではなく、partial overlap下のpersistent cell correspondence recoveryを評価する。

## Primary synthetic result

| Method | Top-1 | Top-3 recall | Matched F1 | Unmatched F1 |
|---|---:|---:|---:|---:|
| M0 Spatial-only UOT | 0.6660 | 0.9158 | 0.6319 | 0.0835 |
| M1 Regional NCC only | 0.6545 | 0.8843 | 0.5814 | 0.0661 |
| M2 Fingerprint nearest candidate | 0.6302 | 0.8770 | 0.5954 | 0.0856 |
| M3 Fingerprint + UOT | 0.6826 | 0.8949 | 0.6422 | 0.2103 |
| M4 Fingerprint + graph + UOT | 0.7600 | 0.9285 | 0.7100 | 0.2680 |
| M5 Full contextual graph-UOT | 0.7829 | 0.9299 | 0.7290 | 0.3446 |

Primary tableはEasy/Medium/Hard × 5 seedsの15 casesのcase平均。median NNはrankingに使用していない。

## Candidate discovery

- Affine spatial: recall@10/20/30 = 0.9994 / 1.0000 / 1.0000
- Regional NCC: recall@10/20/30 = 0.9587 / 0.9881 / 0.9934
- Fingerprint: recall@10/20/30 = 0.6801 / 0.7921 / 0.8474
- Combined: recall@10/20/30 = 0.9860 / 0.9942 / 1.0000

## Regional NCC hypothesis

Gaussian 6/12 µmとDoG 6–18 µmを固定併用したlocal NCCのprimary casesにおけるcorrect-region top-1/top-3は 0.9038/0.9327。best-second margin中央値は 0.0473。

| Missingness | Correct-region Top-1 | Top-3 | Median NCC margin |
|---:|---:|---:|---:|
| 0% | 0.8649 | 0.9054 | 0.0510 |
| 10% | 0.8400 | 0.8667 | 0.0500 |
| 20% | 0.9041 | 0.9178 | 0.0529 |
| 30% | 0.8592 | 0.8592 | 0.0473 |
| 40% | 0.8406 | 0.8696 | 0.0471 |
| 50% | 0.8571 | 0.9206 | 0.0420 |

NCCはmissingness 50%でもregion top-1 0.8571を維持した。ただしAffine spatial candidate recall@10が既にほぼ1.0のため、今回のNCCの価値は正解を候補集合へ救済することより、候補costのlocal offset priorとして現れた。

## Fingerprint

Descriptorはsorted absolute/locally-normalized kNN distances、multi-radius density/count、rotation-invariant angular Fourier magnitudes、covariance anisotropy、近傍間距離とangular-gap quantileをrobust scalingした。画像・学習モデル・cell IDはdescriptor生成に使っていない。spatial top-80内の固定diagnosticは次の通り。

| Descriptor group | Recall@10 | Hard-negative AUC |
|---|---:|---:|
| all fingerprint components | 0.6801 | 0.8136 |
| absolute kNN distances | 0.6005 | 0.8171 |
| radial density/count | 0.4855 | 0.7160 |
| locally normalized kNN distances | 0.4688 | 0.7374 |
| angular Fourier | 0.4594 | 0.7529 |
| covariance anisotropy/scale | 0.4202 | 0.6773 |
| neighbor-pair geometry | 0.3520 | 0.6737 |

全component併用がrecall@10で最良。単独groupではabsolute kNN distanceが最も強かった。ただしM3のTop-1増分は+1.66ポイント、15 cases中8勝6敗1同率であり、fingerprint単独の効果は小さい。

## Graph consistency

M5 candidate pairで、graph supportによるtrue/false candidateのrank AUCはprimary 0.9505、全case 0.9473。mutual-10NN上でedge length・angle・neighbor candidate compatibilityをsoft集約し、完全なgraph isomorphismは要求していない。M3→M4はTop-1 +7.74ポイント、14勝1敗で最大の改善源だった。

## UOT as partial assignment

UOT conservativeのentropy=0.1、KL relaxation=0.5、row-mass threshold=0.25、translation-invariant Sinkhornをsource-of-truthどおり使用し、全45×4 UOT runsが収束した。UOTは候補発見器ではなく、候補外costを抑えたglobal partial-assignment layerとして使用した。top-1 massだけでなくrow entropy、margin、row mass、reciprocal、regional/fingerprint/graph componentを保存した。

M5はmatched F1を0.6319→0.7290、unmatched F1を0.0835→0.3446へ改善した。ただしunmatched recallは0.2438に留まり、unmatched detectionは未完成。confidence≥0.75のpredicted matchesはprecision 0.9576、common-cell coverage 0.4697だったが、ECE 0.1435なのでconfidenceを校正済み確率とは解釈しない。

## Ablation

| Stage | Top-1 | Δ vs M0 | Incremental Δ |
|---|---:|---:|---:|
| M0 Spatial-only UOT | 0.6660 | +0.0000 | — |
| M3 Fingerprint + UOT | 0.6826 | +0.0166 | +0.0166 |
| M4 Fingerprint + graph + UOT | 0.7600 | +0.0940 | +0.0774 |
| M5 Full contextual graph-UOT | 0.7829 | +0.1169 | +0.0229 |

## Real-data exploratory result

`serial_section_pair_20260904`へM5を探索的適用した。matched/ambiguous/unmatched = 25/434/1295。最大confidenceは0.6984でsyntheticのhigh-confidence閾値0.75に達せず、domain shiftまたはcost scale不適合を示唆する。ground-truth cell correspondenceがないためaccuracyは主張しない。相対confidence順のhigh/medium/low各12例をmanual review用CSVへ保存した。

## Research conclusion

**A. FULL TREE SUPPORTED**

M5はM0を15/15 primary casesで上回ったためfull pipelineの研究方向を支持する。ただし寄与は均等ではなくgraphが主因、fingerprint単独は小さく、regional priorは補助的だった。この分類はtrue cell correspondence recovery、candidate recall、unmatched detection、stage ablationに基づき、registration後のmedian NN低下を根拠にしていない。

## Recommended next step

追加の独立real serial-section pairで、固定済みの今回手法をblind manual review可能な形で評価する。Workflow C production algorithmへの統合は行わない。

## Scope and limitations

- synthetic identityは既知だが、実serial-section biologyを完全には表現しない。
- real datasetにはcell-level ground truthがない。
- weightsと閾値は固定baselineであり、大規模parameter sweepは実施していない。
- syntheticではAffine spatial recall@10がほぼ1.0で、candidate rescue能力の比較には易しすぎる側面がある。
- unmatched recallとconfidence calibrationは不十分で、realへの閾値移植は支持されない。
- optional displacement field reconstructionとJacobian rejectionは今回のcorrespondence評価に使用していない。
