#!/usr/bin/env bash
# Real-data performance experiment; copied model updates are discarded.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/torchinductor-cache
export TRITON_CACHE_DIR=/root/autodl-tmp/triton-cache
python_bin="${VIMEML_PYTHON:-/root/miniconda3/bin/python}"
output_dir="${1:-$PWD/outputs/model-checks/v21-real-data-performance-$(date +%Y%m%d-%H%M%S)}"
if [[ -e "$output_dir" ]]; then
  echo "Use a fresh output directory: $output_dir" >&2
  exit 2
fi
mkdir -p "$output_dir" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
nvidia-smi --query-gpu=uuid,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,temperature.gpu --format=csv -l 1 > "$output_dir/gpu.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT
"$python_bin" -X utf8 -u scripts/training/benchmark_v21.py \
  --config configs/train-v21.toml --output "$output_dir/measurements" \
  --batch-sizes 256 512 --warmup 32 --steps 128 --workers 4 \
  2>&1 | tee "$output_dir/benchmark.log"
