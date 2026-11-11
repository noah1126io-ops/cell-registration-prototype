#!/bin/bash
#SBATCH --job-name=oir-syn
#SBATCH --partition=mib-dbia
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec /mnt/mib-dbia08_1/users/noa.ganeko/miniconda3/envs/cell-rg/bin/python -B run_synthetic.py --case "$SLURM_ARRAY_TASK_ID"
