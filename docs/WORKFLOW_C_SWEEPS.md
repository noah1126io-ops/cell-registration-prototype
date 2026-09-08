# Workflow C Joint Flow parameter sweeps

This CLI runs deterministic grid search through one Slurm array. Each task reuses the validated Workflow C CUDA worker and produces its existing result ZIP. Registration, float64 arithmetic, filtering, checkpoint selection, and safety/application rules are unchanged. No Streamlit sweep UI or adaptive optimizer is included.

## Create, submit, summarize

Activate the existing `cell-rg` environment, then run from the repository:

```bash
python scripts/create_workflow_c_sweep.py \
  --sweep-config examples/workflow_c_sweep_synthetic.json \
  --output-root sweeps
python scripts/submit_workflow_c_sweep.py --sweep-dir sweeps/<sweep_id>
python scripts/summarize_workflow_c_sweep.py --sweep-dir sweeps/<sweep_id>
```

Creation prints the Cartesian product count, total including baseline, and concurrency. It never submits a job. `--base-config path/to/config.json` overrides the specification's `base_config`; otherwise that path is relative to the sweep specification. Input paths are relative to the base config, or absolute. The example produces exactly four trials with concurrency two.

```json
{
  "base_config": "workflow_c_synthetic_gpu.json",
  "method": "grid",
  "parameters": {
    "stage_a_learning_rate": [0.10, 0.15],
    "stage_b_learning_rate": [0.05, 0.075]
  },
  "max_trials": 100,
  "concurrency": 2
}
```

The eight supported keys are `stage_a_learning_rate`, `stage_b_learning_rate`, `stage_a_update_smoothing`, `stage_b_update_smoothing`, `density_weight`, `support_weight`, `structure_weight`, and `soft_jacobian_weight`. Values must be finite nonnegative numbers; learning rates must be positive. Scale-list searches and all safety-threshold searches are rejected. Stage-specific overrides already present in the base retain their existing semantics: for example, a non-null `stage_b_structure_weight` takes precedence over `structure_weight`. Choose a search dimension that actually controls the base case.

Parameter names and unique numeric values are sorted. The baseline is always index `0`, `is_baseline=true`; remaining Cartesian combinations follow in deterministic order with the baseline combination removed if present. If the baseline lies outside the grid, it adds one trial. Both grid count and total including baseline must fit the limit (default 100). A deliberate `--max-trials N` or specification `max_trials` overrides that limit. `--concurrency K` overrides the specification (default two).

The base is expanded using the current Joint Flow/public-wrapper/density function defaults; supplied values, inputs, affine, and point selection are preserved. Every trial config contains the full exposed numerical parameter set, including CUDA and float64. Optional null parameters retain their documented fallback semantics. Preset names are metadata, not a substitute for parameter values. Only explicit search dimensions differ across trials. The manifest retains both the original and expanded base configs.

## Resources, staging, reproducibility

The array is `0-(N-1)%K`, requesting `partition=mib-dbia`, `gres=gpu:1`, `nodes=1`, `ntasks=1`, `cpus-per-task=4`, `mem=16G`, and `time=00:30:00`. It does not request exclusive GPU/node use or alter behavior by GPU model. `CELLREG_PYTHON` can override the existing cluster Python interpreter path.

```text
sweeps/<sweep_id>/
  sweep_config.json
  manifest.json
  status.json
  submission_started.json
  input/                         # staged once, read-only files
  slurm-<array>_<task>.stdout.log
  slurm-<array>_<task>.stderr.log
  trials/0000/
    config.json                  # complete effective config
    input -> ../../input
    status.json
    slurm_job.json
    execution_started.json
    stdout.log
    stderr.log
    result/workflow_c_registration_result.zip
  summary/
    summary.json
    trials.csv
    valid_trials.csv
    ranking.csv
```

Repeated references to the same source file are staged once. SHA-256 and byte size verify each copy; read-only staged inputs and config checksums are checked before execution. The manifest records git SHA, dirty state, source-file hashes, creation time, original/expanded base, search space, order, input identities, Python/platform, concurrency, and array job ID. Changing captured Python/batch source after creation requires creating a new sweep. This prevents silently running a different implementation against an old manifest; it is not a source-code snapshot. Retain the recorded checkout/commit for reproducibility.

Each trial's input symlink is copied by the existing worker to node-local `$SLURM_TMPDIR/<trial_id>`, or `/tmp/cellreg_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}/<trial_id>`. There is no per-trial NFS image copy. The worker computes locally, writes the artifact atomically on NFS, reloads it to verify the copy, then deletes that trial's scratch directory. On failure scratch remains for diagnosis. Logs, status, configs and final artifacts remain on shared storage.

## Status and failures

Trial status files record `queued`, `running`, `completed`, or `failed`, with transition history. `queued` initially means prepared (possibly not yet submitted). The summarize command refreshes sweep-level counts from these persistent files; the sweep-level file is a snapshot at the last refresh, not a continuously running monitor. Completion is never inferred from disappearance from `squeue`.

The submission marker prevents duplicate or concurrent submission, including ambiguous `sbatch` responses. A definite submission failure marks queued trials failed and saves the error. No automatic retry is attempted. Each worker has a separate execution marker and artifact directory. A worker exception records a failure and preserves its stderr without affecting other tasks. Launcher errors and catchable termination signals also mark failures. A hard node loss or uncatchable SIGKILL can leave the last persistent state at running; consult Slurm accounting before manually resolving such a stale status. No accounting-based success inference is used.

## Aggregation and ranking

Aggregation loads completed ZIPs without rerunning registration. It verifies artifact structure, numeric-array finiteness, and the artifact parameter set against the trial configuration. Missing/corrupt artifacts and failed trials remain visible but cannot rank. Incomplete trials are `PENDING` and the remaining completed results can still be summarized.

CSV columns include effective parameters, applied/rejected status, rejection reason, symmetric median distance, within 3/5/10 µm, mutual nearest fraction, displacement median/p95/max, Jacobian min/p05/median/p95/max, foldover fraction, runtime, node, GPU model, Slurm identifiers, artifact/config paths, and baseline deltas. Point metrics are from the applied result; displacement/Jacobian QC describes the attempted final field so a rejected unsafe attempt is not hidden by an identity applied field. Runtime is diagnostic only, not a speed benchmark.

Existing Workflow C artifacts store `manifest.fine_applied`, but not a separate `FineWarpResult` status/message/rejection reason. The CSV explicitly identifies `fine_status_source=artifact.manifest.fine_applied`; status is derived as applied/rejected, and Joint Flow's existing generic rejection reason is reported on rejection. No more detailed reason is invented.

A valid run requires the existing application gate to pass, `finite_output=true`, finite numeric arrays and metrics, zero foldover, and positive minimum Jacobian. Thresholds are not relaxed. The main ranking contains only valid runs, ordered by:

1. Symmetric median distance ascending.
2. Within 5 µm descending.
3. Mutual nearest fraction descending.
4. Displacement p95 ascending.
5. Array index ascending to break exact ties.

No composite score is used. Baseline deltas are candidate minus baseline for median distance, within 5 µm, mutual nearest, and displacement p95. Deltas are blank when the baseline or candidate metrics are unavailable. Rejected baselines are retained as references but excluded from the valid ranking.
