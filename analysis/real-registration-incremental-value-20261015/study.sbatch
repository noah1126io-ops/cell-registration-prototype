#!/bin/bash
#SBATCH --partition=mib-dbia
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu/analysis/real-registration-incremental-value-20261015
exec /home/noa.ganeko/miniconda3/envs/cell-rg/bin/python -B -u run_study.py "$1" --index "${SLURM_ARRAY_TASK_ID:-0}"
