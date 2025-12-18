#!/bin/bash
set -euo pipefail

# Optional: activate your env (adjust as needed)
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate ml_exp

python run_repeatedcv_n50.py \
  --budget 16 \
  --cv 3 \
  --test-size 0.25 \
  --n-total 50 \
  --repetitions 1,50,100 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --countdown-every 30 \
  --out-csv n50_repeatedcv_results.csv
