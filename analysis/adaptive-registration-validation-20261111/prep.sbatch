#!/bin/bash
#SBATCH --job-name=arv-prep
#SBATCH --partition=mib-dbia
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=logs/prep_%j.out
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/mnt/mib-dbia08_1/users/noa.ganeko/miniconda3/envs/cell-rg/bin/python
cd /mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu/analysis/adaptive-registration-validation-20261111
/usr/bin/time -v $PY -B prep.py features
$PY -B prep.py cases
$PY -B prep.py manifest
