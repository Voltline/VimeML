#!/usr/bin/env bash
# Cluster launcher for V5: auto-resumes from last.pt when resubmitted.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
# The image exports PYTHONPATH to its own Python 3.10 tree, which shadows
# the training venv's triton (cp312) and breaks torch.compile.
unset PYTHONPATH
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
python_bin="${VIMEML_PYTHON:-python3}"
# Resolve the last --config argument to find this run's own output directory.
config="configs/train-v5.toml"
prev=""
for arg in "$@"; do
  if [[ "$prev" == "--config" ]]; then config="$arg"; fi
  prev="$arg"
done
output_dir=$(grep '^output_dir' "$config" | head -1 | sed 's/.*= *"\(.*\)"/\1/')
resume=()
if [[ -f "${output_dir:-artifacts/models/tiny-ja-v5-e8k-d512-l8}/last.pt" ]]; then
  resume=(--resume)
fi
exec "$python_bin" -X utf8 -u scripts/training/train_v5.py --config "$config" "${resume[@]}" "$@"
