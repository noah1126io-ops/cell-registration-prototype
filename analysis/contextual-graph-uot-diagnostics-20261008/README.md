# Contextual Graph-UOT low matched-rate diagnostics

Analysis-only audit of the 2026-10-08 contextual graph-UOT experiment. This
directory diagnoses the frozen implementation and outputs without changing any
matching weight, threshold, production workflow, or prior analysis artifact.

Run:

```bash
/home/noa.ganeko/miniconda3/envs/cell-rg/bin/python \
  analysis/contextual-graph-uot-diagnostics-20261008/diagnose.py
```

The normal real-data pipeline is recomputed and checked field-by-field against
the saved source result before any diagnostic result is accepted. Negative
controls are separate diagnostic runs and are never used to tune the method.

Primary outputs:

- `REPORT.md`: interpretation with observed facts, inference, and unknowns kept separate.
- `per_nucleus_trace.csv`: all 1,754 moving IDs, top-3 candidates, costs, UOT mass, confidence components, and final reason.
- `stage_diagnostics.csv`, `classification_reasons.csv`, `synthetic_real_distribution.csv`, and `confidence_analysis.csv`: requested machine-readable audits.
- `individual_nucleus_examples.csv`: 20 examples per available category; the no-candidate group is empty because every row has 30 candidates.
- Six requested PNG diagnostics, including local point-cloud candidate overlays. Candidate lines in the real-data figure are proposals, not ground truth.

Tests:

```bash
/home/noa.ganeko/miniconda3/envs/cell-rg/bin/python -m pytest -q \
  analysis/contextual-graph-uot-diagnostics-20261008/tests
```
