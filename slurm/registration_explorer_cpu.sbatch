#!/bin/bash
#SBATCH --job-name=cellreg-explorer
#SBATCH --partition=mib-dbia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=00:45:00

# Research-mode Registration Explorer job (CPU only). Production Workflow C/D are not involved.
set -euo pipefail

RUN_DIR="$1"
# sbatch runs a spooled copy of this file, so the repository root comes from the submit directory (--chdir)
REPOSITORY_ROOT="${CELLREG_REPO:-${SLURM_SUBMIT_DIR:?submit with --chdir=<repository root>}}"
PYTHON_BIN="${CELLREG_PYTHON:-/mnt/mib-dbia08_1/users/noa.ganeko/miniconda3/envs/cell-rg/bin/python}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

mark_launch_failed() {
  "$PYTHON_BIN" -c \
    'import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); from src.workflow_c_slurm import update_status; update_status(Path(sys.argv[2]), "failed", error="batch launcher failed before job completion")' \
    "$REPOSITORY_ROOT" "$RUN_DIR" || true
}
trap mark_launch_failed ERR

cd "$REPOSITORY_ROOT"
exec "$PYTHON_BIN" -B scripts/run_registration_explorer_job.py --run-dir "$RUN_DIR"
