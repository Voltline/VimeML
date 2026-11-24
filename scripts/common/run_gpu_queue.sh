#!/usr/bin/env bash
# Run a list of shell commands on the GPUs of one node, one command per GPU at a time.
#   run_gpu_queue.sh JOBS [GPUS] [LOGDIR]
# JOBS: one command per line, run from the repository root with CUDA_VISIBLE_DEVICES set to the GPU it got
# (so every command sees one device, cuda:0). Empty lines and lines starting with # are skipped; a line
# "wait" waits for every running command before the next starts (shards before their merge, say).
# GPUS: comma-separated ids (default: every GPU nvidia-smi lists). Each command's output goes to
# LOGDIR/NNN.log (default outputs/gpu-queue/<JOBS name>). Exit status: the number of failed commands.
set -uo pipefail
cd "$(dirname "$0")/../.."
jobs_file="${1:?usage: run_gpu_queue.sh JOBS [GPUS] [LOGDIR]}"
if [[ -n "${2:-}" ]]; then IFS=, read -ra gpus <<< "$2"
else mapfile -t gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader); fi
(( ${#gpus[@]} )) || { echo "No GPUs" >&2; exit 2; }
logs="${3:-outputs/gpu-queue/$(basename "$jobs_file" .txt)}"
mkdir -p "$logs"
declare -A busy=()          # pid -> gpu
declare -A command_of=()    # pid -> job number
failed=0 n=0

reap() {  # wait for one command to end and free its GPU
  local pid status
  wait -n -p pid; status=$?
  [[ -n "${pid:-}" ]] || return 0
  if (( status )); then failed=$((failed + 1)); echo "$(date '+%F %T') FAIL #${command_of[$pid]} (status $status)";
  else echo "$(date '+%F %T') done #${command_of[$pid]}"; fi
  unset "busy[$pid]" "command_of[$pid]"
}

free_gpu() {
  local g used
  for g in "${gpus[@]}"; do
    used=0
    for pid in "${!busy[@]}"; do [[ ${busy[$pid]} == "$g" ]] && used=1; done
    (( used )) || { echo "$g"; return; }
  done
}

while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "${line// }" || "$line" == \#* ]] && continue
  if [[ "$line" == wait ]]; then
    while (( ${#busy[@]} )); do reap; done
    continue
  fi
  while (( ${#busy[@]} >= ${#gpus[@]} )); do reap; done
  g=$(free_gpu)
  n=$((n + 1))
  log=$(printf '%s/%03d.log' "$logs" "$n")
  echo "$(date '+%F %T') start #$n on GPU $g: $line"
  printf '%s\n' "$line" > "$log"
  CUDA_VISIBLE_DEVICES="$g" bash -c "$line" >> "$log" 2>&1 &
  busy[$!]=$g
  command_of[$!]=$n
done < "$jobs_file"
while (( ${#busy[@]} )); do reap; done
echo "$(date '+%F %T') all done: $n commands, $failed failed"
exit "$failed"
