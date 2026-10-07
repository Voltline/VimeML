#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source /root/autodl-tmp/vimeml-wandb.env
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export WANDB_DIR="$PWD/artifacts/tracking"
export TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/torchinductor-cache
export TRITON_CACHE_DIR=/root/autodl-tmp/triton-cache
mkdir -p "$WANDB_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
python_bin="${VIMEML_PYTHON:-/root/miniconda3/bin/python}"
exec "$python_bin" -X utf8 -u scripts/training/train_wandb.py --config configs/train-v21-extend.toml "$@"
