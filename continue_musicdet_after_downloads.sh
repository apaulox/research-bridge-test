#!/usr/bin/env bash
set -euo pipefail
project=/home/huskypaul/dacon/musicdet
python_bin=/home/huskypaul/miniforge3/envs/dacon-cu128/bin/python
log="$project/logs/continuation.log"
while pgrep -f "$project/download_fmc.py" >/dev/null || pgrep -f "$project/download_musiccaps_real.py" >/dev/null; do
  date -Is >> "$log"
  sleep 60
done
if [[ ! -f "$project/data_archives/FakeMusicCaps.zip" ]]; then
  echo 'FMC archive did not finish; stopping before preparation.' >> "$log"
  exit 1
fi
# Retry unavailable/bad MusicCaps items twice; valid files are skipped cheaply.
for retry in 1 2; do
  echo "MusicCaps retry pass $retry" >> "$log"
  "$python_bin" "$project/download_musiccaps_real.py" \
    --manifest "$project/data_preparation/required_real_audio.csv" \
    --output-dir "$project/data_raw/musiccaps" --sleep 1 --workers 2 \
    >> "$log" 2>&1
done
"$python_bin" "$project/prepare_musicdet_fmc_dataset.py" \
  --repo "$project/vendor/MusicDET" \
  --archive "$project/data_archives/FakeMusicCaps.zip" \
  --real-dir "$project/data_raw/musiccaps" \
  --raw-fake-dir "$project/data_raw/fmc" \
  --output-dir "$project/vendor/MusicDET/datasets/fakemusiccaps/all_audio_wav" \
  --available-protocol-dir "$project/label_available" --workers 8 \
  >> "$log" 2>&1
run_dir="$project/runs/musicdet_fmc_real_only_seed688_v1"
if [[ -e "$run_dir" ]]; then
  echo "Run directory exists; refusing to overwrite: $run_dir" >> "$log"
  exit 1
fi
"$python_bin" "$project/train_musicdet_real_only.py" \
  --repo "$project/vendor/MusicDET" \
  --audio-dir "$project/vendor/MusicDET/datasets/fakemusiccaps/all_audio_wav" \
  --train-protocol "$project/label_available/TTM01_train.txt" \
  --dev-protocol "$project/label_available/TTM01_dev_paired.txt" \
  --output-dir "$run_dir" --epochs 10 --batch-size 16 --workers 4 \
  >> "$log" 2>&1
"$python_bin" "$project/evaluate_musicdet_fmc.py" \
  --repo "$project/vendor/MusicDET" --checkpoint-dir "$run_dir" \
  --audio-dir "$project/vendor/MusicDET/datasets/fakemusiccaps/all_audio_wav" \
  --protocol-dir "$project/label_available" \
  --output-dir "$run_dir/evaluation" --batch-size 64 --workers 4 \
  >> "$log" 2>&1
echo 'MusicDET real-only training completed.' >> "$log"
