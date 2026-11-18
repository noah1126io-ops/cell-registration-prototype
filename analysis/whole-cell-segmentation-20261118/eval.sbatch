#!/bin/bash
#SBATCH --job-name=wcs-ev
#SBATCH --partition=mib-dbia
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_%j.out
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
/mnt/mib-dbia08_1/users/noa.ganeko/miniconda3/envs/cell-rg/bin/python /mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu/analysis/whole-cell-segmentation-20261118/evaluate.py
