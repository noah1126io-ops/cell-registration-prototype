#!/bin/bash
#SBATCH --job-name=crx-pytest
#SBATCH --partition=mib-dbia
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/pytest_%j.out
# usage: sbatch pytest.sbatch [new|all]   (new = only tests_crx; all = existing suites + tests_crx incl. slow UI tests)
export OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
PY=/mnt/mib-dbia08_1/users/noa.ganeko/miniconda3/envs/cell-rg/bin/python
cd /mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu
echo "== HEAD $(git rev-parse HEAD)"; git status --porcelain | head -20
if [ "$1" = "all" ]; then
  echo "== existing suite (tests/)"
  $PY -m pytest -q -p no:cacheprovider tests 2>&1 | tail -6
  echo "== frozen study tests: graph-matching-adoption-separation-20261014"
  (cd analysis/graph-matching-adoption-separation-20261014 && $PY -m pytest -q -p no:cacheprovider tests 2>&1 | tail -4)
  echo "== frozen study tests: contextual-graph-uot-20261008"
  (cd analysis/contextual-graph-uot-20261008 && $PY -m pytest -q -p no:cacheprovider tests 2>&1 | tail -4)
  echo "== region-cell-matching-20261113 (tests_rcm)"
  $PY -m pytest -q -p no:cacheprovider analysis/region-cell-matching-20261113/tests_rcm 2>&1 | tail -8
  echo "== this study (tests_crx, all)"
  $PY -m pytest -q -p no:cacheprovider analysis/cell-review-explorer-20261117/tests_crx 2>&1 | tail -30
else
  echo "== this study (tests_crx, not slow)"
  $PY -m pytest -q -p no:cacheprovider -m "not slow" analysis/cell-review-explorer-20261117/tests_crx 2>&1 | tail -60
fi
