#!/usr/bin/env bash
set -u
project=/home/huskypaul/dacon/musicdet
echo 'Processes:'
pgrep -af 'download_fmc.py|download_musiccaps_real.py|continue_musicdet_after_downloads.sh|resume_musicdet_after_prepare_fix.sh|prepare_musicdet_fmc_dataset.py|train_musicdet_real_only.py|evaluate_musicdet_fmc.py' || true
echo
echo 'Downloaded data:'
du -sh "$project/data_archives" "$project/data_raw/musiccaps" 2>/dev/null || true
printf 'MusicCaps WAV files: '
find "$project/data_raw/musiccaps" -maxdepth 1 -type f -name '*.wav' 2>/dev/null | wc -l
echo
echo 'Recent MusicCaps log:'
tail -n 8 "$project/logs/musiccaps-download.log" 2>/dev/null || true
echo
echo 'Recent continuation log:'
tail -n 8 "$project/logs/continuation.log" 2>/dev/null || true
echo
echo 'Training outputs:'
find "$project/runs" -maxdepth 2 -type f 2>/dev/null | sort | tail -n 12 || true
