#!/bin/bash
#SBATCH --job-name=wcs-ex
#SBATCH --partition=mib-dbia
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/export_%j.out
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
E=/mnt/mib-dbia08_1/users/noa.ganeko/miniconda3/envs/cell-rg/bin/python
D=/mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu/analysis/whole-cell-segmentation-20261118
$E $D/evaluate.py && $E $D/full_section.py export
