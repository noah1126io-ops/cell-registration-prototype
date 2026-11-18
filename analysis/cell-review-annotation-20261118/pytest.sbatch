#!/bin/bash
#SBATCH --job-name=cra-pytest
#SBATCH --partition=mib-dbia
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/pytest_%j.out
# usage: sbatch pytest.sbatch [new|all]   (new = only tests_cra incl. slow; all = every existing suite + tests_crx + tests_cra)
export OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
PY=/mnt/mib-dbia08_1/users/noa.ganeko/miniconda3/envs/cell-rg/bin/python
cd /mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu
echo "== HEAD $(git rev-parse HEAD)"; git status --porcelain | head -20
if [ "$1" = "all" ]; then
  echo "== existing suite (tests/)"
  $PY -m pytest -q -p no:cacheprovider tests 2>&1 | tail -4
  echo "== frozen study tests: graph-matching-adoption-separation-20261014"
  (cd analysis/graph-matching-adoption-separation-20261014 && $PY -m pytest -q -p no:cacheprovider tests 2>&1 | tail -3)
  echo "== frozen study tests: contextual-graph-uot-20261008"
  (cd analysis/contextual-graph-uot-20261008 && $PY -m pytest -q -p no:cacheprovider tests 2>&1 | tail -3)
  echo "== region-cell-matching-20261113 (tests_rcm)"
  $PY -m pytest -q -p no:cacheprovider analysis/region-cell-matching-20261113/tests_rcm 2>&1 | tail -4
  echo "== cell-review-explorer-20261117 (tests_crx, all)"
  $PY -m pytest -q -p no:cacheprovider analysis/cell-review-explorer-20261117/tests_crx 2>&1 | tail -30
fi
echo "== this study (tests_cra, all)"
$PY -m pytest -q -p no:cacheprovider analysis/cell-review-annotation-20261118/tests_cra -W ignore 2>&1 | grep -vE "pyarrow|Arrow|^  File|^    |Traceback|st.iframe|components.v1.html|ScriptRunContext|^\s*\^" | tail -120
