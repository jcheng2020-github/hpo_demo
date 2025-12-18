#!/bin/bash
set -euo pipefail

# Optional: activate your env (adjust as needed)
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate ml_exp

# Example run (random search)
python hpo_sample_size.py \
  --algo random \
  --budget 18 \
  --cv 3 \
  --test-size 0.25 \
  --sample-sizes 50,500,1000,2000,5000,10000,20000,40000 \
  --seeds 0,1,2,3,4 \
  --countdown-every 30 \
  --out-csv sample_size_hpo_results.csv
