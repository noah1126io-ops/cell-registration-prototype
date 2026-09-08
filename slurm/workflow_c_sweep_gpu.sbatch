#!/bin/bash
#SBATCH --job-name=cellreg-sweep
#SBATCH --partition=mib-dbia
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
REPOSITORY_ROOT="$1"
SWEEP_DIR="$2"
PYTHON_BIN="${CELLREG_PYTHON:-/home/noa.ganeko/miniconda3/envs/cell-rg/bin/python}"
mark_launch_failed() {
  "$PYTHON_BIN" -c 'import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); from src.workflow_c_sweep import trial_path, read_json; from src.workflow_c_slurm import update_status; p=trial_path(Path(sys.argv[2]), int(sys.argv[3])); state=read_json(p / "status.json")["state"]; update_status(p, "failed", error="array launcher failed or terminated; see stderr.log") if state in ("queued", "running") else None' "$REPOSITORY_ROOT" "$SWEEP_DIR" "$SLURM_ARRAY_TASK_ID" || true
}
trap mark_launch_failed ERR
trap 'mark_launch_failed; exit 143' TERM INT HUP
TRIAL_ID="$(printf '%04d' "$SLURM_ARRAY_TASK_ID")"
cd "$REPOSITORY_ROOT"
"$PYTHON_BIN" scripts/run_workflow_c_sweep_trial.py --sweep-dir "$SWEEP_DIR" > "$SWEEP_DIR/trials/$TRIAL_ID/stdout.log" 2> "$SWEEP_DIR/trials/$TRIAL_ID/stderr.log"
