#!/bin/bash
#SBATCH --job-name=arv-pytest
#SBATCH --partition=mib-dbia
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pytest_%j.out
export OMP_NUM_THREADS=4
PY=/mnt/mib-dbia08_1/users/noa.ganeko/miniconda3/envs/cell-rg/bin/python
cd /mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu
$PY -m pytest -q -p no:cacheprovider tests 2>&1 | tail -15
echo "== study tests"
$PY -m pytest -q -p no:cacheprovider analysis/adaptive-registration-validation-20261111/test_adaptive_validation.py 2>&1 | tail -5
