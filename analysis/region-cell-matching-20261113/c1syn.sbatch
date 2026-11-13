#!/bin/bash
#SBATCH --job-name=rcm-c1syn
#SBATCH --partition=mib-dbia
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --array=0-38%10
#SBATCH --output=logs/c1syn_%a.out
export OMP_NUM_THREADS=4
cd /mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu/analysis/region-cell-matching-20261113
/mnt/mib-dbia08_1/users/noa.ganeko/miniconda3/envs/cell-rg/bin/python -B run_c1_synthetic.py --index $SLURM_ARRAY_TASK_ID
