#!/bin/bash
#SBATCH --job-name=uot-adopt-v2
#SBATCH --partition=mib-dbia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-4
#SBATCH --output=/mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu/analysis/uot-adoption-validation-20260911/synthetic_shards_v2/slurm-%A_%a.stdout.log
#SBATCH --error=/mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu/analysis/uot-adoption-validation-20260911/synthetic_shards_v2/slurm-%A_%a.stderr.log

set -euo pipefail
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1

cd /mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu
exec /mnt/mib-dbia08_1/users/noa.ganeko/miniconda3/envs/cell-rg/bin/python -B -u \
  analysis/uot-adoption-validation-20260911/validate_synthetic.py \
  --seed-index "${SLURM_ARRAY_TASK_ID}" --device cuda \
  --output-dir analysis/uot-adoption-validation-20260911/synthetic_shards_v2
