#!/usr/bin/env bash
set -euo pipefail
project=/home/huskypaul/dacon/musicdet
python_bin=/home/huskypaul/miniforge3/envs/dacon-cu128/bin/python
mkdir -p "$project/logs" "$project/data_archives" "$project/data_raw/musiccaps"
if pgrep -f "$project/download_fmc.py" >/dev/null; then
  echo 'FMC downloader is already running'
else
  nohup "$python_bin" "$project/download_fmc.py" \
    --output "$project/data_archives/FakeMusicCaps.zip" \
    > "$project/logs/fmc-download.log" 2>&1 &
  echo "$!" > "$project/logs/fmc-download.pid"
fi
if pgrep -f "$project/download_musiccaps_real.py" >/dev/null; then
  echo 'MusicCaps downloader is already running'
else
  nohup "$python_bin" "$project/download_musiccaps_real.py" \
    --manifest "$project/data_preparation/required_real_audio.csv" \
    --output-dir "$project/data_raw/musiccaps" --sleep 0.5 --workers 4 \
    > "$project/logs/musiccaps-download.log" 2>&1 &
  echo "$!" > "$project/logs/musiccaps-download.pid"
fi
sleep 2
pgrep -af 'download_fmc.py|download_musiccaps_real.py'
