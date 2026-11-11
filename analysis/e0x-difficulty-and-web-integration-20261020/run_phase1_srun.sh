#!/bin/bash
# Phase 1 driver: each task is one short srun job (backfill-friendly); concurrency capped by xargs -P.
set -u
cd "$(dirname "$0")"
PY=/mnt/mib-dbia08_1/users/noa.ganeko/miniconda3/envs/cell-rg/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
seq 0 15 | xargs -P 6 -I{} srun -p mib-dbia -J e0x-abl -c 4 --mem=4G -t 00:45:00 -o logs/abl-srun-{}.out $PY -B run_ablation.py --task {} &
seq 0 59 | xargs -P 10 -I{} srun -p mib-dbia -J e0x-syn -c 4 --mem=4G -t 00:45:00 -o logs/syn-srun-{}.out $PY -B synth.py --task {} &
wait
echo PHASE1_DONE
