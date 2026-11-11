#!/bin/bash
#SBATCH --job-name=claude-audit-tests
#SBATCH --partition=mib-dbia
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=analysis/real-affine-candidate-audit-claude-20261014/logs/%x-%j.out
export PYTHONDONTWRITEBYTECODE=1
cd /mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu
PY=/home/noa.ganeko/miniconda3/envs/cell-rg/bin/python
echo "== analysis tests"; $PY -B -m pytest -q -p no:cacheprovider analysis/real-affine-candidate-audit-claude-20261014/test_audit.py 2>&1 | tail -5
echo "== repository tests"; $PY -B -m pytest -q -p no:cacheprovider tests 2>&1 | tail -5
