| Method | Top-1 | Matched Precision | Matched Recall | Matched F1 | Unmatched F1 | Coverage at 95% Precision |
|---|---:|---:|---:|---:|---:|---:|
| Position + Graph | 0.9568 | 0.7923 | 0.9401 | 0.8599 | 0.4721 | 0.4762 |
| Relative-cost Graph-UOT | 0.9471 | 0.6386 | 0.9401 | 0.7606 | 0.0000 | 0.3849 |
| Separated Matching / Adoption | 0.9471 | 0.9530 | 0.7061 | 0.8112 | 0.8253 | 0.5070 |
| Full contextual UOT | 0.9694 | 0.9211 | 0.8287 | 0.8724 | 0.7604 | 0.4557 |

The table is the pooled held-out Real-like Test set. The last column is the descriptive held-out precision--coverage envelope, not a threshold retuned for deployment. Validation-fixed threshold performance is reported separately below.

# Main finding

The pre-specified research decision is **A. SEPARATED MATCHING SUPPORTED**.

Relative row-centering changed Graph-UOT Top-1 from 0.9568 to 0.9471 and native matched F1 from 0.8599 to 0.7606. The separated method used the relative Graph-UOT Top-1, but replaced the absolute-mass gate with a synthetic-only logistic adoption model. Its matched precision/recall/F1 were 0.9530/0.7061/0.8112.

The B2 match model was fit only on five Train base patterns. Its threshold was selected only on five Validation base patterns at target precision 0.95, then frozen. On held-out Test, that threshold produced precision 0.9530 and coverage 0.4958; target-achieved=True. Test labels were used to evaluate, never to refit or select the operational threshold.

# Is Graph still useful?

Position-only Top-1 was 0.9039; Position+Graph-UOT was 0.9568, a change of +0.0529. Candidate-level Graph AUC averaged 0.9853.

| Strength | Graph AUC | True support median | Other support median |
|---|---:|---:|---:|
| Moderate | 0.9802 | 0.8018 | 0.0072 |
| Strong | 0.9766 | 0.7748 | 0.0073 |
| Weak | 0.9948 | 0.8194 | 0.0074 |

Candidate-source sensitivity on held-out cases:

| Candidate source | Recall@30 | Raw Position+Graph Top-1 | Graph AUC |
|---|---:|---:|---:|
| Full contextual candidate-30 | 1.0000 | 0.9430 | 0.9853 |
| Spatial candidate-30 | 1.0000 | 0.9388 | 0.9818 |

Graph remains useful only to the extent supported by these Real-like persistent IDs. It is not evidence of Real correspondence accuracy.

# Is UOT necessary?

Graph ranking without UOT achieved Top-1 0.9457; relative Graph-UOT achieved 0.9471. Their matched F1 values after their respective adoption logic were 0.8666 and 0.8112. Across all rows, within-case duplicate-destination rates were 0.2619 for raw Graph ranking and 0.2171 for relative Graph-UOT. Relative UOT changed Top-1 by +0.0014; absolute-cost B0 changed it by +0.0111. Thus UOT adds little to ranking in the row-centered path, but B2 is the only Graph-centered operational rule that transferred the Validation-selected 95% precision target to Test; raw Graph adoption reached 0.9383, below 0.95.

Duplicate destinations are algorithmic many-to-one warnings, not proof of biological error; real 2-D detections may include split/merge exceptions.

# Does adoption separation help?

At the descriptive held-out 95% precision envelope, B0/B1/B2 coverages were 0.4762, 0.3849, and 0.5070. B2 minus B0 was +0.0308.

Negative controls generated 303 B0, 581 B1, and 75 B2 false accepted matches across the three controls. Identity permutation preserves geometry while declaring all IDs unrelated, so high confidence in that control exposes the fundamental inability of geometry-only matching to detect identity replacement; it is deliberately confounded rather than a realistic serial-section sample.

# Synthetic--Real gap

| Variable | Real-like Test median | Real median |
|---|---:|---:|
| candidate_distance_um | 4.9799 | 10.4809 |
| candidate_margin | 0.4231 | 0.3371 |
| confidence | 0.8036 | 0.1718 |
| graph_support | 0.7183 | 0.2273 |
| knn_distance_um | 10.3190 | 11.9526 |
| minimum_contextual_cost | 0.2896 | 0.8043 |
| nuclear_density_per_um2 | 0.0020 | 0.0019 |
| predicted_unmatched_fraction | 0.3038 | 0.8660 |
| row_mass_ratio | 0.4667 | 0.2223 |

The Real-like generator matched density by construction while retaining spatial nonuniformity, partial overlap, independent local missing clusters, section-specific nuclei, unequal cardinality, nonlinear deformation, and centroid noise. Any residual differences above limit external validity; Real values were not transformed to resemble synthetic.

# Real application

Real has no cell-level truth. Counts therefore measure adoption behavior only.

| Method | Matched | Ambiguous | Unmatched | Coverage | Confidence median | Reciprocal | Duplicate destinations |
|---|---:|---:|---:|---:|---:|---:|---:|
| Position + Graph | 716 | 698 | 340 | 0.4082 | 0.5547 | 0.7127 | 6 |
| Relative-cost Graph-UOT | 1691 | 63 | 0 | 0.9641 | 0.6284 | 0.7412 | 402 |
| Separated Matching / Adoption | 26 | 209 | 1519 | 0.0148 | 0.1718 | 0.7412 | 0 |
| Full contextual UOT | 25 | 434 | 1295 | 0.0143 | 0.5411 | 0.6585 | 0 |

B2 confidence is derived exclusively from synthetic Train data and is not claimed to be a calibrated Real correctness probability. More Real matches are not interpreted as higher accuracy.

# Existing candidate-fixed baseline reproduction

The prior 15-case candidate-fixed results were read from the frozen Experiment A outputs and checked against their known values: Position+Graph Top-1/F1 0.8523/0.8071; Full 0.8362/0.7990. These are not mixed with the new held-out Real-like table.

# Research decision

**A. SEPARATED MATCHING SUPPORTED**

The decision rule was fixed in code before interpreting Real results. Real adoption count is not part of the decision.

# Recommended next step

1. Repeat the frozen B2 calibrator on at least one independently designed low-density generator family; the current split prevents seed/template leakage but remains one generator family.
2. Obtain a small blinded Real correspondence set and evaluate candidate recall plus the frozen precision--coverage curve before changing any production threshold.
3. If UOT adds little beyond raw Graph ranking, test a simpler one-to-one conflict resolver as a separate, pre-registered experiment rather than weakening the mass gate ad hoc.

# Scope

No deformation field was generated. Workflow C, Streamlit UI, Workflow D, production safety gates, and production algorithms were not changed. Row centering is a controlled diagnostic/matching prototype, not an adopted production algorithm.
