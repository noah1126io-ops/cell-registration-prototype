#!/bin/bash
#SBATCH --job-name=wcs-full
#SBATCH --partition=mib-dbia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/full_%j.out
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
E=/mnt/mib-dbia08_1/users/noa.ganeko/miniconda3/envs
D=/mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu/analysis/whole-cell-segmentation-20261118
$E/cell-rg/bin/python $D/full_section.py crop && \
$E/cellseg-cp/bin/python $D/full_section.py seg 2>&1 | grep -v "%|" && \
$E/cell-rg/bin/python $D/full_section.py export
