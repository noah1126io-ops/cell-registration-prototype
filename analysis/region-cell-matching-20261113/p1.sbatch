#!/bin/bash
#SBATCH --job-name=rcm-p1
#SBATCH --partition=mib-dbia
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/p1_%A_%a.out
# usage: sbatch --array=0-N%10 p1.sbatch <case-list-file>
export OMP_NUM_THREADS=4
cd /mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu/analysis/region-cell-matching-20261113
CASE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $1)
/mnt/mib-dbia08_1/users/noa.ganeko/miniconda3/envs/cell-rg/bin/python -B run_phase1.py --case "$CASE"
