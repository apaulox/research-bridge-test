#!/usr/bin/env bash
set -euo pipefail
project=/home/huskypaul/dacon/musicdet
mkdir -p "$project/logs"
if pgrep -f "$project/continue_musicdet_after_downloads.sh" >/dev/null; then
  echo 'Continuation is already running'
  exit 0
fi
nohup bash "$project/continue_musicdet_after_downloads.sh" \
  > "$project/logs/continuation-launch.log" 2>&1 &
echo "$!" > "$project/logs/continuation.pid"
sleep 1
ps -o pid,etime,cmd -p "$(cat "$project/logs/continuation.pid")"
