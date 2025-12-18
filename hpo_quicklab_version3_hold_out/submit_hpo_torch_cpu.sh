#!/bin/bash
#SBATCH --job-name=HPO_TorchNN_CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16gb
#SBATCH --time=01:00:00
#SBATCH --output=__experi_%j_output/hpo_cpu_%j.out
#SBATCH --error=__experi_%j_output/hpo_cpu_%j.err

set -euo pipefail

hostname; date; pwd
mkdir -p "__experi_${SLURM_JOB_ID}_output"

export XDG_RUNTIME_DIR="${SLURM_TMPDIR:-/tmp}/${USER}/xdg_${SLURM_JOB_ID}"
mkdir -p "$XDG_RUNTIME_DIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export TORCH_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_THREADING_LAYER=GNU

module load conda || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ml_exp_torch

WORKSPACE="${1:-$PWD}"
cd "$WORKSPACE"
echo "Workspace is $(realpath "$WORKSPACE")"

DATADIR="${WORKSPACE}/data"

# FashionMNIST on CPU
python run.py \
  --dataset fashion_mnist \
  --data-dir "${DATADIR}" \
  --budget 600 \
  --print-every 10 \
  --algos grid,random,bayes,genetic \
  --seed 42 \
  --val-frac 0.15 \
  --max-batches-per-epoch 200 |& tee "__experi_${SLURM_JOB_ID}_output/console.log"
