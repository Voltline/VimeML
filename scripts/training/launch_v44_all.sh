#!/usr/bin/env bash
# Start all six V4.4 arms on one node, one GPU each, detached from this shell.
# GPUs default to 0-5; override with VIMEML_V44_GPUS="2 3 4 5 6 7".
# Rerunning resumes arms whose process has stopped and skips arms still running.
set -euo pipefail
cd "$(dirname "$0")/../.."
arms=(s1 s2 s3 chat0 chat10 crop60)
read -r -a gpus <<< "${VIMEML_V44_GPUS:-0 1 2 3 4 5}"
(( ${#gpus[@]} == ${#arms[@]} )) || { echo "Need ${#arms[@]} GPUs, got ${#gpus[@]}" >&2; exit 2; }
mkdir -p logs/v44
for i in "${!arms[@]}"; do
  arm="${arms[$i]}"
  pidfile="logs/v44/${arm}.pid"
  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "$arm already running (pid $(cat "$pidfile"))"; continue
  fi
  setsid nohup bash scripts/training/cluster_v44.sh "$arm" "${gpus[$i]}" \
    >> "logs/v44/${arm}.log" 2>&1 < /dev/null &
  echo $! > "$pidfile"
  echo "$arm -> GPU ${gpus[$i]}, pid $!, log logs/v44/${arm}.log"
done
