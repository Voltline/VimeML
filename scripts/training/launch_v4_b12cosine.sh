#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# Optional private W&B environment file, kept outside the repository.
if [[ -n "${VIMEML_WANDB_ENV:-}" ]]; then source "$VIMEML_WANDB_ENV"; fi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export WANDB_DIR="$PWD/artifacts/tracking"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache}"
mkdir -p "$WANDB_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
python_bin="${VIMEML_PYTHON:-python3}"
exec "$python_bin" -X utf8 -u scripts/training/train_wandb.py --config configs/train-v4-b12cosine.toml "$@"
