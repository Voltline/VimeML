#!/usr/bin/env bash
# Start all eight V4.5 students on one node, one GPU each, detached from this shell.
# GPUs default to 0-7; override with VIMEML_V45_GPUS. With fewer free GPUs, run waves:
# VIMEML_V45_ARMS="geo-v42 geo-s1 geo-s2 geo-s3" VIMEML_V45_GPUS="0 1 2 3".
# Rerunning resumes arms whose process has stopped and skips arms still running.
set -euo pipefail
cd "$(dirname "$0")/../.."
read -r -a arms <<< "${VIMEML_V45_ARMS:-geo-v42 geo-s1 geo-s2 geo-s3 mix-v42 mix-s1 mix-s2 mix-s3}"
read -r -a gpus <<< "${VIMEML_V45_GPUS:-0 1 2 3 4 5 6 7}"
(( ${#gpus[@]} == ${#arms[@]} )) || { echo "Need ${#arms[@]} GPUs, got ${#gpus[@]}" >&2; exit 2; }
mkdir -p logs/v45
for i in "${!arms[@]}"; do
  arm="${arms[$i]}"
  pidfile="logs/v45/${arm}.pid"
  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "$arm already running (pid $(cat "$pidfile"))"; continue
  fi
  setsid nohup bash scripts/training/cluster_v45.sh "$arm" "${gpus[$i]}" \
    >> "logs/v45/${arm}.log" 2>&1 < /dev/null &
  echo $! > "$pidfile"
  echo "$arm -> GPU ${gpus[$i]}, pid $!, log logs/v45/${arm}.log"
done
