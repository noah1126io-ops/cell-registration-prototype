#!/bin/bash
#SBATCH --job-name=he-stardist
#SBATCH --partition=mib-dbia
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
S="$1"; DATA=/mnt/mib-dbia08_1/users/noa.ganeko/projects/$S
IMG=$(ls $DATA/${S}_HE*.ome.tif)
exec /mnt/mib-dbia08_1/users/noa.ganeko/miniconda3/envs/stardist/bin/python -u detect_he_nuclei.py --image "$IMG" --out out/$S
