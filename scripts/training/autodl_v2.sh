#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source /root/autodl-tmp/vimeml-wandb.env
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export WANDB_DIR="$PWD/artifacts/tracking"
mkdir -p "$WANDB_DIR"
python_bin="${VIMEML_PYTHON:-/root/miniconda3/bin/python}"
exec "$python_bin" -X utf8 -u scripts/training/train.py --config configs/train-v2.toml "$@"
