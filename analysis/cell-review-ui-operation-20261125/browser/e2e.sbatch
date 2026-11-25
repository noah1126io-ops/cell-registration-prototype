#!/bin/bash
#SBATCH --job-name=cro-e2e
#SBATCH --partition=mib-dbia
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=logs/e2e_%j.out
# usage (from the study dir): sbatch browser/e2e.sbatch [prepare]   -- real-browser checks (headless Chromium) at dpr 1 and 2,
# with the tab hidden behind another tab, and with IF whole-cell outlines shown.  Outputs: logs/e2e_*.json (not committed), figures/*.jpg
export OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
PY=/mnt/mib-dbia08_1/users/noa.ganeko/miniconda3/envs/cell-rg/bin/python
UI=/mnt/mib-dbia08_1/users/noa.ganeko/envs/uitest/bin/python
D=/mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu/analysis/cell-review-ui-operation-20261125
cd $D
echo "== HEAD $(git rev-parse HEAD) on $(hostname)"; git status --porcelain | head -20
BASE=/tmp/cro-e2e-$SLURM_JOB_ID; mkdir -p $BASE/auto
$PY browser/prepare.py $BASE/auto 031-6_2 || exit 1
run() {  # name, port, extra args
  local R=$BASE/$1; rm -rf $R; mkdir -p $R; cp -r $BASE/auto/031-6_2 $R/
  $UI browser/e2e.py $R logs/e2e_$1.json --port $2 "${@:3}"; echo "== $1 rc=$?"
}
run dpr1 8597 --dpr 1 --shots figures
run dpr2 8598 --dpr 2 --shots figures
run cells 8599 --dpr 1 --cells
run tabs 8600 --dpr 1 --tabs
$PY - <<'P'
import json, glob
for f in sorted(glob.glob("logs/e2e_*.json")):
    R = json.load(open(f)); print(f, sum(r["ok"] for r in R), "/", len(R), [r["check"] for r in R if not r["ok"]])
P
rm -rf $BASE
