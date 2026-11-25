#!/bin/bash
#SBATCH --job-name=cro-insp
#SBATCH --partition=mib-dbia
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/inspect_%j.out
export OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
PY=/mnt/mib-dbia08_1/users/noa.ganeko/miniconda3/envs/cell-rg/bin/python
UI=/mnt/mib-dbia08_1/users/noa.ganeko/envs/uitest/bin/python
cd /mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu/analysis/cell-review-ui-operation-20261125
B=/tmp/cro-insp-$SLURM_JOB_ID; mkdir -p $B; $PY browser/prepare.py $B 031-6_2
$UI browser/e2e.py $B logs/inspect.json --port 8601 --inspect
rm -rf $B
