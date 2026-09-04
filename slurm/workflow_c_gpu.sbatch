#!/bin/bash
#SBATCH --job-name=cellreg-cuda
#SBATCH --partition=mib-dbia
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

CONFIG_PATH="$1"
RUN_DIR="$(cd "$(dirname "$CONFIG_PATH")" && pwd)"
REPOSITORY_ROOT="$(cd "$RUN_DIR/../.." && pwd)"
PYTHON_BIN="${CELLREG_PYTHON:-/home/noa.ganeko/miniconda3/envs/cell-rg/bin/python}"

mark_launch_failed() {
  "$PYTHON_BIN" -c \
    'import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); from src.workflow_c_slurm import update_status; update_status(Path(sys.argv[2]), "failed", error="batch launcher failed before worker completion")' \
    "$REPOSITORY_ROOT" "$RUN_DIR" || true
}
trap mark_launch_failed ERR

cd "$REPOSITORY_ROOT"
exec "$PYTHON_BIN" scripts/run_workflow_c.py --config "$CONFIG_PATH"
