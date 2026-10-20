#!/usr/bin/env bash
# Cluster launcher for V4.2: auto-resumes from last.pt when resubmitted.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
# The image exports PYTHONPATH to its own Python 3.10 tree, which shadows
# the training venv's triton (cp312) and breaks torch.compile.
unset PYTHONPATH
# Keep torch.compile caches on local disk; they are ephemeral but fast.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
python_bin="${VIMEML_PYTHON:-python3}"
resume=()
if [[ -f artifacts/models/tiny-ja-v42-e8k-d384-l7/last.pt ]]; then
  resume=(--resume)
fi
exec "$python_bin" -X utf8 -u scripts/training/train_v42.py --config configs/train-v42.toml "${resume[@]}" "$@"
