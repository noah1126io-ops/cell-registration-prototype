# Experiment A — Graph-UOT cost decomposition and mass suppression

This is a candidate-fixed controlled ablation. Every A0--A6 run uses the exact Full-method 30-candidate indices. It is therefore not the original unrestricted Spatial-UOT baseline.

## Main result

| Cost | Total mass | Row mass median | Matched | Ambiguous | Unmatched | Synthetic Top-1 | Synthetic Matched F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Position | 0.3534 | 0.3789 | 1305 | 293 | 156 | 0.7142 | 0.6644 |
| Position+FP | 0.2333 | 0.2511 | 866 | 505 | 383 | 0.7391 | 0.7022 |
| Position+Region | 0.1842 | 0.1781 | 551 | 589 | 614 | 0.7914 | 0.7496 |
| Position+Graph | 0.2125 | 0.2223 | 716 | 698 | 340 | 0.8523 | 0.8071 |
| Position+FP+Graph | 0.1417 | 0.1489 | 212 | 851 | 691 | 0.8246 | 0.7864 |
| Position+FP+Region | 0.1220 | 0.1133 | 222 | 610 | 922 | 0.7910 | 0.7562 |
| Full | 0.0770 | 0.0694 | 25 | 434 | 1295 | 0.8362 | 0.7990 |

## Frozen implementation

- Position contribution: `0.45 * d^2/(2*10um^2)`.
- Fingerprint contribution: `0.85 * fingerprint_cost`.
- Region contribution: `0.70 * (regional_residual^2/(2*10um^2) + 0.25*(1-regional_confidence))`.
- Graph contribution: `0.75*(1-graph_support)`.
- Non-candidate dense entries retain cost 25 in A0--A6. UOT uses entropy 0.1, marginal KL 0.5, translation-invariant Sinkhorn, tolerance 1e-7, and unchanged status thresholds.
- A6 reproduces the saved real Full result: {'unmatched': 1295, 'ambiguous': 434, 'matched': 25}; maximum audited numeric error 9.988e-17.

All component values are nonnegative and finite. Across all 30 Real candidates, contribution medians (p95) are position 5.648 (21.430), fingerprint 0.369 (1.512), region 8.895 (33.472), and graph 0.743 (0.749). For each row's minimum-Full-cost candidate, the corresponding medians are 0.490, 0.387, 0.374, and 0.567; their summed Full minimum median is 2.044.

The graph term is particularly uplift-like on Real: its all-candidate row-mean median is 0.723, while its within-row SD median is only 0.051. It still changes relative ranking, but most candidates receive a similar positive addition.

## Q1. Why is transported mass so low?

**Absolute contextual cost suppresses UOT mass.** A0 Position has total mass 0.3534; A6 Full has 0.0770. Its median minimum candidate cost changes 0.220→2.044, while the median top1-top2 cost gap changes 0.288→0.720. Thus relative separation grows, but the absolute minimum grows much more. Median row-mass ratio changes 0.379→0.069; Full minimum-cost/row-mass Spearman is -0.962.

Objective decomposition confirms that the solver trades transport cost against KL destruction/creation of marginal mass; details are in `cost_ablation_metrics.csv`. This vendored POT version logs `total_cost` with backend `kl_div(..., mass=False)`, which is near zero at the solution; the report therefore also records a separately named nonnegative generalized-KL objective diagnostic. Row-normalized concentration remains separate from absolute row mass.

## Q2. Does Graph improve identity discrimination?

**Yes on the existing synthetic benchmark.** With fixed candidates, A1→A4 changes UOT Top-1 0.7391→0.8246 and raw-cost Top-1 0.6806→0.7968. Across common synthetic cells, UOT identity outcomes are improved=247, worsened=41, remained correct=1741; raw-cost improved=315, worsened=35. This supports discrimination on synthetic truth, not on unlabeled real data.

## Q3. Does Graph suppress UOT mass?

**Yes.** Real A1→A4 total mass changes 0.2333→0.1417; median row mass 0.251→0.149. Paired real patterns are rank-changed/mass-increased=0, rank-changed/mass-decreased=134, rank-unchanged/mass-increased=0, rank-unchanged/mass-decreased=1620. “Rank changed” is deliberately not called an improvement on real data.

## Q4. Are useful matches being rejected by the mass gate?

**Yes on synthetic, to the extent quantified by known IDs.** A4 has Top-1 accuracy 0.8246 but matched recall 0.8001; 59 pooled correct Top-1 assignments are not labeled matched. A1 has corresponding values 0.7391, 0.7155, and 57. Importantly, Graph does **not** reduce aggregate matched recall here: A1→A4 recall improves because its discrimination gain outweighs its mass penalty. The Full condition rejects 88 correct Top-1 rows and has matched recall 0.7997. Precision, recall, F1, wrong-match counts, and difficulty strata are retained in `synthetic_correspondence_metrics.csv`.

## Q5. Can mass and discrimination be separated?

**Mathematically yes; operational adoption still needs calibration.** B1 adds +1 to every dense cost and preserves every raw ranking/gap, yet total mass changes 0.0770→0.0310. B2 subtracts each row minimum while preserving within-row ranks/gaps and yields 0.4187. B3 rescales all costs while preserving ranks and yields 0.2325. UOT top-1 preservation versus B0 is 1.000, 0.934, and 0.926; global/row competition can alter the plan even when raw ranks are exact.

## Interpretation

**B. USEFUL DISCRIMINATION WITH MASS PENALTY** is closest. A1→A4 improves synthetic identity and lowers mass, although its net matched recall increases rather than decreases. Therefore the penalty is mathematically real but does not erase Graph's aggregate synthetic benefit. The larger A4→A6 contextual addition shows the clearer adoption loss: Top-1 rises 0.8246→0.8362, while correct-but-not-matched rows rise 59→88.

Real data have no truth, so no real correspondence-accuracy claim is made. Increased mass is not increased correctness, and row-normalized mass is not a match probability.

## Q6. What should change next?

The minimal research change to evaluate next is to decouple **relative candidate discrimination** from **absolute match adoption**: retain contextual relative costs, then calibrate the absolute cost baseline/mass decision on held-out persistent-ID synthetic data plus a small blinded real correspondence set. The B diagnostics are sensitivity analyses, not proposed winning algorithms; no offset/rescaling or threshold is adopted here.

No deformation field was generated. This experiment stops at Candidate → Cost → UOT plan → Mass → correspondence decision. It does not explain prior small warp displacement, which also involved confidence weighting, anchor filtering, smoothing, and safety constraints.
