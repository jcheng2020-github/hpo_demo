#!/bin/bash
#SBATCH --job-name=HPO_SampleSize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16gb
#SBATCH --time=04:00:00
#SBATCH --output=__experi_%j_output/sample_size_%j.out
#SBATCH --error=__experi_%j_output/sample_size_%j.err

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
python hpo_sample_size.py \
  --algo random \
  --budget 18 \
  --cv 3 \
  --test-size 0.25 \
  --sample-sizes 50,500,1000,2000,5000,10000,20000,40000 \
  --seeds 0,1,2,3,4 \
  --countdown-every 30 \
  --out-csv "${OUTDIR}/sample_size_hpo_results.csv"

date
echo "Job ${SLURM_JOB_ID} finished."
