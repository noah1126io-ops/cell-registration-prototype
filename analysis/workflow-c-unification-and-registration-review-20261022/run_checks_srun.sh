#!/bin/bash
# Run the parity / end-to-end checks: one short srun per task (CPU, partition mib-dbia).
set -u
cd "$(dirname "$0")"
PY=/mnt/mib-dbia08_1/users/noa.ganeko/miniconda3/envs/cell-rg/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
for t in "$@"; do
  srun -p mib-dbia -J core-check -c 4 --mem=8G -t 01:30:00 -o logs/$t.out $PY -B run_core_checks.py --task $t &
done
wait
echo CHECKS_DONE "$@"
