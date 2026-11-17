#!/bin/bash
#SBATCH --job-name=crx-fig
#SBATCH --partition=mib-dbia
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/fig_%j.out
# usage: sbatch smoke.sbatch 031-8_6 [extra args]
export OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
cd /mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu/analysis/cell-review-explorer-20261117
/mnt/mib-dbia08_1/users/noa.ganeko/miniconda3/envs/cell-rg/bin/python -B make_figures.py
