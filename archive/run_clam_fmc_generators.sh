#!/usr/bin/env bash
set -euo pipefail

source /home/huskypaul/miniforge3/etc/profile.d/conda.sh
conda activate dacon-cu128
cd /mnt/c/Users/husky_gp7j99y/Documents/ChatGPT/DACON

for generator in TTM02 TTM03 TTM04 TTM05; do
  echo "START ${generator}"
  python clam_fmc_eval.py \
    --manifest "/home/huskypaul/dacon/musicdet/label_available/${generator}_eval.txt" \
    --real-dir /home/huskypaul/dacon/musicdet/data_raw/musiccaps \
    --fake-dir /home/huskypaul/dacon/musicdet/data_raw/fmc \
    --mert-dir /home/huskypaul/dacon/clam/models/MERT-v1-95M \
    --wav2vec-dir /home/huskypaul/dacon/clam/models/wav2vec2-base-100k-gtzan-music-genres \
    --checkpoint /home/huskypaul/dacon/clam/MoM-CLAM/model_wts/best_model_triplet_loss_margin_0.2.pth \
    --output "/home/huskypaul/dacon/clam/validation/${generator}_eval_50x2_d90.csv" \
    --per-class 50 \
    --durations 90
  python eat_fmc_scores.py \
    --manifest "/home/huskypaul/dacon/musicdet/label_available/${generator}_eval.txt" \
    --real-dir /home/huskypaul/dacon/musicdet/data_raw/musiccaps \
    --fake-dir /home/huskypaul/dacon/musicdet/data_raw/fmc \
    --eat-dir /home/huskypaul/dacon/submissions/eat75_musicdet25_fmc_v1/model/eat_fmc \
    --ensemble-script /home/huskypaul/dacon/submissions/eat75_musicdet25_fmc_v1/script.py \
    --output "/home/huskypaul/dacon/clam/validation/${generator}_eval_50x2_eat.csv" \
    --per-class 50
done
