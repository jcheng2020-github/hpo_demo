#!/bin/bash
#SBATCH --job-name=HPO_RepCV_n50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16gb
#SBATCH --time=04:00:00
#SBATCH --output=__experi_%j_output/repeatedcv_%j.out
#SBATCH --error=__experi_%j_output/repeatedcv_%j.err

set -euo pipefail

hostname
date
pwd

# ============================
# Output & runtime directories
# ============================
OUTDIR="__experi_${SLURM_JOB_ID}_output"
mkdir -p "$OUTDIR"

# Some clusters require this to avoid runtime-dir errors
export XDG_RUNTIME_DIR="${SLURM_TMPDIR:-/tmp}/${USER}"
mkdir -p "$XDG_RUNTIME_DIR"

# ============================
# Threading / BLAS safety
# ============================
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Joblib temp folder (important for sklearn CV)
export JOBLIB_TEMP_FOLDER="${SLURM_TMPDIR:-/tmp}/${USER}/joblib_${SLURM_JOB_ID}"
mkdir -p "$JOBLIB_TEMP_FOLDER"

# ============================
# Conda environment
# ============================
module load conda || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ml_exp

# ============================
# Workspace
# ============================
WORKSPACE="${1:-$PWD}"
cd "$WORKSPACE"
echo "Workspace: $(realpath "$WORKSPACE")"

# ============================
# Run experiment
# ============================
python run_repeatedcv_n50.py \
  --budget 16 \
  --cv 3 \
  --test-size 0.25 \
  --n-total 50 \
  --repetitions 1,3,5,10,50,100 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --countdown-every 30 \
  --out-csv "${OUTDIR}/n50_repeatedcv_results.csv"

date
echo "Job ${SLURM_JOB_ID} finished."
