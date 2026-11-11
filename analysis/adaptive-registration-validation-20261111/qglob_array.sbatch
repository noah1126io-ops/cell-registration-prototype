#!/bin/bash
#SBATCH --job-name=arv-qglob
#SBATCH --partition=mib-dbia
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=logs/qglob_%A_%a.out
# one (case, method) per array task; task list = `python benchmark_runner.py --list`
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/mnt/mib-dbia08_1/users/noa.ganeko/miniconda3/envs/cell-rg/bin/python
cd /mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu/analysis/adaptive-registration-validation-20261111
echo "host $(hostname) start $(date -Is)"
/usr/bin/time -v $PY -B quadrant_global.py --index "$SLURM_ARRAY_TASK_ID"
