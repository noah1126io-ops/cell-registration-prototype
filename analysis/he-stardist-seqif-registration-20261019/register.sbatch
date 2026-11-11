#!/bin/bash
#SBATCH --job-name=he-seqif-reg
#SBATCH --partition=mib-dbia
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
S="$1"; DATA=/mnt/mib-dbia08_1/users/noa.ganeko/projects/$S
PY=/mnt/mib-dbia08_1/users/noa.ganeko/miniconda3/envs/cell-rg/bin/python
$PY -u register.py --sample $S --data $DATA --nuclei out/$S/he_nuclei.csv --out out/$S/registration
$PY -u qc_overlay.py --sample $S --data $DATA --reg out/$S/registration --nuclei out/$S/he_nuclei.csv
