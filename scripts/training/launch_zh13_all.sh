#!/usr/bin/env bash
# Start all eight vimeml-tiny-zh-v1 arms on one node, one GPU each, detached from this shell.
# GPUs default to 0-7; override with VIMEML_ZH13_GPUS="0 1 2 3 4 5 6 7".
# Rerunning resumes arms whose process has stopped and skips arms still running.
set -euo pipefail
cd "$(dirname "$0")/../.."
arms=(12k-fresh-s1 12k-fresh-s2 12k-fresh-s3 12k-ep5 12k-ep20 8k-fresh 8k-ep5 8k-ep20)
read -r -a gpus <<< "${VIMEML_ZH13_GPUS:-0 1 2 3 4 5 6 7}"
(( ${#gpus[@]} == ${#arms[@]} )) || { echo "Need ${#arms[@]} GPUs, got ${#gpus[@]}" >&2; exit 2; }
mkdir -p logs/zh13
for i in "${!arms[@]}"; do
  arm="${arms[$i]}"
  pidfile="logs/zh13/${arm}.pid"
  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "$arm already running (pid $(cat "$pidfile"))"; continue
  fi
  setsid nohup bash scripts/training/cluster_zh13.sh "$arm" "${gpus[$i]}" \
    >> "logs/zh13/${arm}.log" 2>&1 < /dev/null &
  echo $! > "$pidfile"
  echo "$arm -> GPU ${gpus[$i]}, pid $!, log logs/zh13/${arm}.log"
done
