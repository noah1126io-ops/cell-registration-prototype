#!/bin/bash
#SBATCH --job-name=rcm-p2known
#SBATCH --partition=mib-dbia
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-10
#SBATCH --output=logs/p2known_%a.out
export OMP_NUM_THREADS=4
cd /mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu/analysis/region-cell-matching-20261113
/mnt/mib-dbia08_1/users/noa.ganeko/miniconda3/envs/cell-rg/bin/python -B run_phase2_known.py --index $SLURM_ARRAY_TASK_ID --chunks 11
