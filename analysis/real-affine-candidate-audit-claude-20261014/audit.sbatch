#!/bin/bash
#SBATCH --job-name=claude-affine-audit
#SBATCH --partition=mib-dbia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu/analysis/real-affine-candidate-audit-claude-20261014
hostname
exec /home/noa.ganeko/miniconda3/envs/cell-rg/bin/python -B -u run_audit.py --stage "${1:-all}"
