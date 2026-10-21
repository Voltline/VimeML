#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# Optional private W&B environment file, kept outside the repository.
if [[ -n "${VIMEML_WANDB_ENV:-}" ]]; then source "$VIMEML_WANDB_ENV"; fi
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export WANDB_DIR="$PWD/artifacts/tracking"
mkdir -p "$WANDB_DIR"
python_bin="${VIMEML_PYTHON:-python3}"
exec "$python_bin" -X utf8 -u scripts/training/train.py --config configs/train-v2.toml "$@"
