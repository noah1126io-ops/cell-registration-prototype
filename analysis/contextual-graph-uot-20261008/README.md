# Multi-scale Contextual Graph-UOT Matching

Analysis-only prototype for discovering persistent cell correspondences between
affine-aligned serial-section nuclear point clouds. It is deliberately isolated
from Workflow C/D and the Streamlit application.

The primary endpoint is ground-truth cell correspondence recovery, not nearest-
neighbor registration distance. The fixed ablation is:

- M0 Spatial-only UOT
- M1 Regional NCC only
- M2 Fingerprint nearest candidate
- M3 Fingerprint + UOT
- M4 Fingerprint + graph consistency + UOT
- M5 Regional NCC + fingerprint + graph consistency + UOT

The conservative UOT configuration is read from the prior source-of-truth
record and frozen at cost scale 10 um, entropy 0.1, KL mass relaxation 0.5,
row-mass threshold 0.25, and translation-invariant Sinkhorn.

Run in the project environment:

```bash
conda run -n cell-rg python analysis/contextual-graph-uot-20261008/run_experiment.py
conda run -n cell-rg python -m pytest -q analysis/contextual-graph-uot-20261008/tests
```

The script refuses to overwrite completed tabular outputs unless `--force` is
passed. Existing `runs/` and `sweeps/` are never read as output targets.

