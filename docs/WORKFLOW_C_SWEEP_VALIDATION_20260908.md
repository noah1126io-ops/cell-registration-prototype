# Workflow C sweep validation — 2026-09-08

Validated on `feature/gpu-slurm`, based on `c03777c`. Sweep manifests record the implementation source SHA-256 hashes and the pre-commit dirty state. Numerical registration and existing worker files were unchanged.

- `python -m compileall -q .`: passed.
- `python -m pytest -q`: **233 passed, 4 skipped** (GPU-dependent tests without a local CUDA device).
- `bash -n slurm/workflow_c_sweep_gpu.sbatch`: passed.
- Added 25 tests covering grid order, baseline/default expansion and CPU result equality, duplicates, trial count guard, forbidden thresholds, relative input paths and shared staging, input/config/source integrity, array mapping, duplicate submission, status transitions, scratch cleanup, failure isolation, artifact parameter/finite checks, ranking/deltas, and incomplete/corrupt results. Unit tests prohibit real sbatch calls.

## Slurm smoke results

Synthetic: array **180384**, `0-3%2`, 4/4 completed and valid. Search dimensions are Stage A learning rate `[0.10, 0.15]` and Stage B learning rate `[0.05, 0.075]`, with baseline `0000` included in the grid.

Real data: array **180509**, `0-1%2`, 2/2 completed and valid. Base run is `runs/20260904-161247-698fe6c9/config.json`; only Stage B learning rate varies `[0.05, 0.06]`. Original fixed nuclei: 1979; moving nuclei: 1842. Existing filtering is preserved: optimization fixed points 1830; success-metric fixed/moving points 1830/1754. The input directory is staged once per sweep.

All six tasks used **NVIDIA RTX 4000 SFF Ada Generation**. Resource policy was the requested 1 GPU / 4 CPUs / 16G / 30 minutes in partition `mib-dbia`; the scheduler selected the nodes without model-specific code.

| Array task | Trial | State | Validity | Node | Symmetric median distance (µm) |
|---|---|---|---|---|---|
| 180384_0 | 0000 | completed | VALID | mib-dbia-u01 | 1.195515240645 |
| 180384_1 | 0001 | completed | VALID | mib-dbia-u06 | 1.170912712241 |
| 180384_2 | 0002 | completed | VALID | mib-dbia-u01 | 1.164449037870 |
| 180384_3 | 0003 | completed | VALID | mib-dbia-u06 | 1.127312661358 |
| 180509_0 | 0000 | completed | VALID | mib-dbia-u01 | 9.469851957431 |
| 180509_1 | 0001 | completed | VALID | mib-dbia-u06 | 9.413604716189 |

These are correctness and orchestration smoke tests, not a performance benchmark or a recommendation to adopt the best candidate.

## Baseline reproduction and concurrency

The real baseline artifact matches the prior validated run with `rtol=1e-10`, `atol=1e-10`. All numeric artifact arrays were compared; maximum transformed-point difference was `1.1368683772161603e-13 µm`, and maximum displacement-component difference was `2.3092638912203256e-14 µm`. Input points, affine matrix/translation, grids and bounds were exactly equal. Before/attempted/applied metrics, safety, Joint Flow stage/final diagnostics and optimization history matched within the same tolerance, with identical application status. The JSON evidence is in the real sweep's `summary/baseline_parity.json`.

Persistent status transition intervals show a peak of **two running trials** for both arrays. Each `summary/smoke_validation.json` records task IDs, nodes, GPU models, artifact SHA-256 values, baseline identity and final counts. Synthetic ranking order is `0003, 0002, 0001, 0000`; real ranking order is `0001, 0000`. No safety constraint or baseline was changed to produce these rankings.

## Artifacts and CSVs

Relative to the repository:

- `sweeps/synthetic-array-validation-20260908-v3/summary/trials.csv`
- `sweeps/synthetic-array-validation-20260908-v3/summary/valid_trials.csv`
- `sweeps/synthetic-array-validation-20260908-v3/summary/ranking.csv`
- `sweeps/real-array-validation-20260908/summary/trials.csv`
- `sweeps/real-array-validation-20260908/summary/valid_trials.csv`
- `sweeps/real-array-validation-20260908/summary/ranking.csv`

Each sweep has artifacts at `trials/<trial_id>/result/workflow_c_registration_result.zip`, plus config/status/Slurm metadata and logs. Generated sweeps, inputs and artifacts are git-ignored; only code, example and documentation are committed.

## Operational limits

A preparation-only synthetic directory and a failed initial submission were retained for diagnosis; successful validation uses the `v3` directory above. The failed submission did not create a successful array. Duplicate submission is deliberately blocked rather than automatically retried.

Sweep counts are refreshed by the summarize command. A hard node loss/SIGKILL may leave a stale running status requiring Slurm accounting review. Existing artifact status/rejection fields are derived from `fine_applied` with their source explicitly recorded; see [sweep documentation](WORKFLOW_C_SWEEPS.md). No Streamlit sweep UI, adaptive search, Workflow D GPU changes, or numerical algorithm changes were made.
