# Cell Correspondence low matched-rate diagnosis

This is a diagnosis-only audit of the frozen `contextual-graph-uot-20261008` prototype. No algorithm, weight, threshold, production workflow, safety setting, or prior artifact was changed.

## Executive stage table

| Stage | Valid moving nuclei | Candidate median | Main rejection / degradation reason |
|---|---:|---:|---|
| Affine | 1,754 | 1,830 fixed search universe | No rejection |
| Regional NCC | 1,754 | 30 | Ranking/prior only; no moving nucleus deleted |
| Fingerprint | 1,754 | 30 | Ranking only; no moving nucleus deleted |
| Graph | 1,754 | 30 | Cost modification only; no candidate deleted |
| UOT | 1,754 | 30 | 1,295 rows have row mass ratio below 0.12 |
| Confidence classification | 25 matched | 30 | Status uses row mass/concentration/margin, not final confidence |

## Source-result verification

- Synthetic M0 Spatial-UOT: Top-1 66.60%, Top-3 91.58%, matched F1 0.632, unmatched F1 0.083.
- Synthetic M5 Full: Top-1 78.29%, Top-3 92.99%, matched F1 0.729, unmatched F1 0.345.
- Real: 1,754 moving nuclei, 25 matched (1.4%), 434 ambiguous (24.7%), 1295 unmatched (73.8%).
- Real maximum confidence is 0.6983803; confidence >=0.75 is 0.
- The recomputed real result agrees with the saved source CSV exactly for status/indices/reciprocity and within 0.00e+00 for confidence.

Synthetic Top-1 accuracy and real matched coverage are different metrics. Real has no cell-correspondence ground truth, so this report makes no real-data accuracy claim.

## Why only 25 matched?

The exact classifier path is:

1. initialize every row as `ambiguous`;
2. assign `unmatched` when `row_mass_ratio < 0.12`;
3. override as `matched` only when row mass ratio >=0.25, normalized top-1 mass >=0.16, and normalized top1-top2 mass margin >=0.01.

On real data, 1,754/1,754 pass concentration and 1,747/1,754 pass the margin, but only 25/1,754 pass the matched row-mass gate. The intersection is exactly 25. The direct bottleneck is therefore **absolute UOT row mass**.

Final confidence is additive, not multiplicative:

`0.30*clip(row_mass_ratio) + 0.25*clip(normalized_top1) + 0.15*clip(margin/0.25) + 0.15*graph_support + 0.15*regional_confidence`.

It is never read by the status code. Thus maximum confidence 0.6983803 and 25 matched are fully compatible: the 25 assignments pass the three independent status conditions even though none reaches 0.75. Reciprocity is also recorded only; fingerprint, graph, and region influence upstream candidate cost, but are not direct status gates.

### Exact frozen scoring and thresholds

- Candidate generation forms spatial top-30, regional top-30, and fingerprint top-30 (fingerprint is evaluated inside spatial top-80), unions the first 20 from each, and prunes to 30 by base cost. It is top-K rather than radius rejection, so every real row receives candidates.
- Base cost is `0.45*position_cost + 0.85*fingerprint_cost + 0.70*region_cost`, where position and regional residual use 10 um scales. Regional cost also includes `0.25*(1-regional_confidence)`.
- Regional confidence is `clip(0.5*(best_NCC+1) + 2*max(best-second margin,0), 0, 1)`. It affects regional cost and contributes 0.15 to final confidence.
- Graph adds `0.75*(1-support)` to base cost. It changes scores/ranks and therefore UOT mass; it has no hard support threshold.
- Candidate pairs populate the dense UOT matrix; all non-candidate pairs receive cost 25. UOT uses entropy 0.1, KL mass relaxation 0.5, and translation-invariant Sinkhorn.
- Each real source has nominal mass `1/max(1754,1830) = 0.00054645`. `row_mass_ratio` is absolute transported row mass divided by that nominal source mass. Top-1/top-2 normalized masses and entropy describe allocation *within* the row and are not correspondence probabilities.
- Reciprocal status is `moving i == argmax_moving(plan[:, top1_fixed_i])`; it is output only. Fingerprint similarity is output and enters cost through fingerprint cost, but is not separately thresholded.
- In this real run all 1,295 unmatched have ratio <0.12; all 434 ambiguous have ratio in [0.12,0.25); and all 25 matched pass ratio >=0.25 plus concentration >=0.16 and margin >=0.01.

## Stage audit

- All 1,754 moving nuclei have 30 spatial, 30 regional, and 30 fingerprint-ranked candidates.
- Their three top-20 lists have union p05/median/p95 32/37/42; base-cost pruning retains exactly 30 for every row.
- Regional best-offset norm p05/median/p95 is 4.24/18.25/25.46 um. 1,025/1,754 (58.4%) nuclei inherit an offset with x or y at the ±18 um search boundary, a direct search-window saturation warning. Regional rank-1 differs from spatial rank-1 for 1,253/1,754 rows; fingerprint rank-1 differs for 1,726/1,754.
- Contextual base cost changes spatial rank-1 for 927 rows; graph cost changes the base-cost minimum for 53; global UOT top-1 differs from the minimum graph-augmented local cost for 195. These are re-ranking counts, not evidence of correctness.
- Candidate-zero, NaN, and Inf counts are all zero.
- Graph calculates soft support for all retained candidates and adds `0.75*(1-support)` to cost. It does not prune.
- Real row-mass-ratio p05/median/p95 is 0.0005/0.0694/0.2065.
- 1,290/1,295 unmatched rows nevertheless pass both within-row concentration and margin conditions. A sharp row-normalized peak is not the same as large absolute mass or a calibrated correspondence probability.

## Is candidate discovery working?

**PARTIALLY.** Mechanically it works: every real nucleus reaches UOT with 30 candidates and all index mappings are valid. However, 1,025 regional offsets hit the search boundary, so the regional prior is frequently search-window-limited. Real scientific candidate recall is still **unknown**, because no true IDs exist. Synthetic mean candidate recall@10/20/30 is:

| Source | @10 | @20 | @30 |
|---|---:|---:|---:|
| Affine spatial | 99.98% | 100.00% | 100.00% |
| Regional NCC | 95.00% | 98.73% | 99.49% |
| Fingerprint | 66.70% | 78.43% | 84.86% |
| Combined | 99.09% | 99.61% | 100.00% |

Near-perfect synthetic affine recall shows that the synthetic candidate problem is easier than the unknown real one.

## Is UOT discarding plausible matches?

**UNKNOWN scientifically; YES as the direct status gate.** All 1,295 unmatched rows are labeled so because row mass ratio is below 0.12. Without real truth or expert labels, sharp rejected candidates cannot be called plausible true matches. 194/1,295 unmatched rows also have minimum contextual cost above the Synthetic Hard p95 (4.0185); this comparison is diagnostic only.

The requested A–F transport phenotypes are separated as follows (flags can overlap):

| Phenotype | Real count | Interpretation |
|---|---:|---|
| A. No candidate | 0 | Not observed |
| B. Cost above Synthetic Hard p95 | 194 | Comparative high-cost flag, not a formal rejection |
| C. Low mass although cost is within Hard p95 | 1101 | Low absolute mass without the comparative high-cost flag |
| D. Mass present but diffuse | 7 | Within-row concentration/margin failure |
| E. Clear top-1 but confidence <0.75 | 1747 | Diagnostic overlap; 0.75 is not a status threshold |
| F. Confidence >=0.75 but rejected | 0 | Not observed |

## Is confidence overly conservative?

**NO as an explanation for 25 matched.** Confidence is not a classifier input. A hypothetical confidence >=0.75 filter covers 0 real nuclei (0.00%), but real precision cannot be inferred. Synthetic calibration/precision-coverage results are in `confidence_analysis.csv`; source M5 ECE is 0.1435.

## Synthetic-Real gap

| Median feature | Easy | Medium | Hard | Real |
|---|---:|---:|---:|---:|
| Nuclear density (/um2) | 0.0092 | 0.0080 | 0.0050 | 0.0019 |
| kNN distance (um) | 4.441 | 4.693 | 5.715 | 11.953 |
| Top-1 spatial distance (um) | 1.851 | 3.780 | 5.392 | 14.831 |
| Regional NCC | 0.961 | 0.890 | 0.807 | 0.360 |
| Fingerprint similarity | 0.954 | 0.893 | 0.838 | 0.794 |
| Graph support | 0.965 | 0.838 | 0.679 | 0.242 |
| Row mass ratio | 0.544 | 0.456 | 0.376 | 0.069 |
| Row entropy | 0.116 | 0.228 | 0.306 | 0.034 |
| Confidence | 0.818 | 0.717 | 0.598 | 0.541 |

The largest practical shifts are much larger real top-candidate distance, lower NCC and graph support, and lower absolute UOT row mass. Real entropy is lower, not higher: the characteristic failure is often small total mass concentrated on one candidate, not broad mass diffusion. Distribution shift does not prove real correspondence error.

## Negative controls

The translation control destroys cross-cloud spatial/regional alignment while preserving internal point-cloud geometry. The deterministic Y-shuffle preserves coordinate marginals/cardinality but destroys 2-D neighborhoods and regional arrangements.

| Median score | Normal | Translated | Y-shuffled |
|---|---:|---:|---:|
| NCC | 0.360 | 0.264 | 0.395 |
| Fingerprint similarity | 0.794 | 0.740 | 0.766 |
| Graph support | 0.242 | 0.125 | 0.227 |
| UOT confidence | 0.541 | 0.504 | 0.541 |

The large translation lowers NCC, graph support, row mass, and confidence, so the pipeline does respond when gross cross-cloud alignment is destroyed. The Y-shuffle is more concerning: median NCC increases from 0.360 to 0.395, graph support changes only 0.242 to 0.227, and confidence remains 0.541. Under this density-preserving control, the current scores provide weak discrimination. This supports a descriptor/domain-gap concern, but it still cannot establish correctness without real labels. Full KS/Wasserstein comparisons are in `negative_control_summary.csv`; controls were not used for tuning.

## Actual unmatched reasons

| Cause | Nuclei | Fraction |
|---|---:|---:|
| Candidateなし | 0 | 0.0% |
| Cost不適合 (explicit rule) | 0 | 0.0% |
| UOT mass不足 | 1295 | 100.0% |
| 候補が曖昧 (explicit unmatched rule) | 0 | 0.0% |
| Graph support不足 (explicit rule) | 0 | 0.0% |
| Reciprocal条件不成立 (explicit rule) | 0 | 0.0% |
| Confidence threshold不達 (explicit rule) | 0 | 0.0% |
| その他 | 0 | 0.0% |

This table follows the actual exclusive classifier. Overlapping upstream/comparative flags are separate in `classification_reasons.csv`. Non-reciprocity occurs in 586/1,295 unmatched rows but cannot be called the reject reason.

## Root-cause ranking

1. **C. UOT MASS / COST PROBLEM — direct observed bottleneck.** Absolute row-mass gates explain all unmatched labels and leave only 25 matched; concentration/margin almost always pass.
2. **B. DESCRIPTOR DOMAIN GAP — strong distribution and control evidence.** Real density, NCC, graph, distance, cost, and mass differ materially from synthetic; the Y-shuffle barely changes graph/confidence and even raises NCC. These findings raise concern about real-domain discrimination, but do not establish individual errors.
3. **A. CANDIDATE DISCOVERY / REGIONAL PRIOR — partially degraded, correctness unresolved.** Numerical coverage is complete, but 1,025 offsets hit the regional search boundary. True real candidate recall remains unidentifiable.
4. **D. CONFIDENCE / CLASSIFICATION PROBLEM — semantic mismatch.** Classification is mass-gated; the reported composite confidence does not classify. It therefore cannot explain or contradict the count.
5. **F. INTRINSIC AMBIGUITY — possible, unconfirmed.** Point clouds may be insufficient, but low entropy/high within-row peaks do not support simple candidate diffusion as the main observed mechanism.

**E. IMPLEMENTATION / DATA INTEGRITY PROBLEM is not supported:** counts, stable IDs, indices, finite values, confidence reconstruction, saved-result reproduction, and synthetic IDs pass checks.

## Facts, inference, unknowns

- **Observed:** every row has candidates; graph never prunes; all unmatched rows fail the absolute-mass floor; confidence/reciprocity are not status inputs; real score distributions differ from synthetic.
- **Inference:** lower regional/graph evidence and larger distances raise cost and likely contribute to low UOT mass. This follows the cost path but is not evidence of real accuracy.
- **Unknown:** whether any real top-1 candidate is the same cell, whether a rejected sharp candidate is biologically plausible, and real candidate recall/precision.

## Recommended minimal fix (proposal only)

The operational cause is confirmed, but biological validity is not. Before changing a threshold or cost, label a small blinded real correspondence set and measure current candidate recall and precision across row mass/cost/confidence. Only after that evidence should row-mass classification be recalibrated for the real cost distribution. No fix was implemented here.
