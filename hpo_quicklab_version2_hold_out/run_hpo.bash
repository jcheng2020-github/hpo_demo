#!/bin/bash
#SBATCH --job-name=HHPO_QuickLab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8gb
#SBATCH --time=01:00:00
#SBATCH --output=__experi_%j_output/hpo_%j.out
#SBATCH --error=__experi_%j_output/hpo_%j.err

set -euo pipefail

hostname; date; pwd

# Make sure output directory exists
mkdir -p "__experi_${SLURM_JOB_ID}_output"

# Some clusters require this to avoid runtime-dir errors
export XDG_RUNTIME_DIR="${SLURM_TMPDIR:-/tmp}/${USER}"
mkdir -p "$XDG_RUNTIME_DIR"

# Optional: keep BLAS from oversubscribing threads (often helps on HPC)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Make sklearn use all CPUs for cross_val_score (joblib backend)
export JOBLIB_TEMP_FOLDER="${SLURM_TMPDIR:-/tmp}/${USER}/joblib_${SLURM_JOB_ID}"
mkdir -p "$JOBLIB_TEMP_FOLDER"

# Load conda (adjust module name if your cluster differs)
module load conda || true

# Activate env
# (use the one that matches your cluster’s conda setup)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ml_exp

# Workspace / project root passed as first argument
WORKSPACE="${1:-$PWD}"
cd "$WORKSPACE"
echo "Workspace is $(realpath "$WORKSPACE")"

# Example: run all 4 algorithms, each with 10 min budget (600s)
# Total worst-case time ~ 40 min + overhead
python run.py \
  --budget 600 \
  --print-every 10 \
  --algos grid,random,bayes,genetic \
  --seed 42
