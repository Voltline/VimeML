#!/usr/bin/env bash
# Cluster launcher for the zh-800m model: builds (or verifies) its frozen
# mixture, then trains, auto-resuming from last.pt when resubmitted.
# Usage: cluster_zh800m.sh [arm] [gpu] (arm defaults to 12k-c512); without [gpu] the scheduler's CUDA_VISIBLE_DEVICES is kept.
# Data parallel: set VIMEML_NPROC to the number of GPUs the job sees (see the end of this file).
# Token stores and c512 indexes must exist first (scripts/tokenizer/encode_zh800m.py).
set -euo pipefail
cd "$(dirname "$0")/../.."
arm="${1:-12k-c512}"
config="configs/train-zh800m-${arm}.toml"
[[ -f "$config" ]] || { echo "No config for arm $arm" >&2; exit 2; }
if [[ -n "${2:-}" ]]; then export CUDA_VISIBLE_DEVICES="$2"; fi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
# The image exports PYTHONPATH to its own Python tree, which shadows the venv's triton.
unset PYTHONPATH
# Per-arm compile caches so concurrent compilations never share a directory.
export TORCHINDUCTOR_CACHE_DIR="/tmp/vimeml-zh800m-${arm}/torchinductor-cache"
export TRITON_CACHE_DIR="/tmp/vimeml-zh800m-${arm}/triton-cache"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
output_dir=$(grep '^output_dir' "$config" | head -1 | sed 's/.*= *"\(.*\)"/\1/')
# W&B never uses the machine's login: drop every inherited WANDB_* variable, point
# config/cache/data/netrc into this arm's own directory, then read only the private file.
while read -r name; do unset "$name"; done < <(compgen -e | grep '^WANDB_' || true)
tracking="$PWD/artifacts/tracking/$(basename "$output_dir")"
mkdir -p "$tracking"/{config,cache,data}
export WANDB_DIR="$tracking" WANDB_CONFIG_DIR="$tracking/config" WANDB_CACHE_DIR="$tracking/cache"
export WANDB_DATA_DIR="$tracking/data" NETRC="$tracking/netrc"
credentials="${VIMEML_WANDB_ENV:-$HOME/.vimeml-wandb.env}"
if [[ -f "$credentials" ]]; then
  if [[ -n "$(find "$credentials" -perm /077)" ]]; then
    echo "Refusing $credentials: readable by others; chmod 600 it" >&2; exit 2
  fi
  set -a; source "$credentials"; set +a
else
  echo "No W&B credential file at $credentials; this arm logs offline" >&2
fi
python_bin="${VIMEML_PYTHON:-python3}"
"$python_bin" -X utf8 -u scripts/corpus/prepare_zh800m.py --arm "$arm"
resume=()
if [[ -f "$output_dir/last.pt" ]]; then
  resume=(--resume)
fi
# VIMEML_NPROC > 1: one process per visible GPU on this machine (torchrun, data parallel; the
# global batch and data order stay those of one GPU). Unset or 1: one process, as before.
nproc="${VIMEML_NPROC:-1}"
if [[ "$nproc" -gt 1 ]]; then
  exec "$python_bin" -X utf8 -u -m torch.distributed.run --standalone --nproc_per_node "$nproc" \
    scripts/training/train_zh200m.py --config "$config" "${resume[@]}"
fi
exec "$python_bin" -X utf8 -u scripts/training/train_zh200m.py --config "$config" "${resume[@]}"
