#!/usr/bin/env bash
# Cluster launcher for one V4.5 distillation student: verifies the teachers, then
# trains, auto-resuming from last.pt when resubmitted. The twin mixtures already exist
# (V4.2 and V4.4 s1-s3) and are verified by the trainer against their pinned SHAs.
# Usage: cluster_v45.sh <arm> [gpu]; without [gpu] the scheduler's CUDA_VISIBLE_DEVICES is kept.
set -euo pipefail
cd "$(dirname "$0")/../.."
arm="${1:?usage: cluster_v45.sh <geo|mix>-<v42|s1|s2|s3> [gpu]}"
config="configs/train-v45-${arm}.toml"
[[ -f "$config" ]] || { echo "No config for arm $arm" >&2; exit 2; }
if [[ -n "${2:-}" ]]; then export CUDA_VISIBLE_DEVICES="$2"; fi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
# The image exports PYTHONPATH to its own Python 3.10 tree, which shadows
# the training venv's triton (cp312) and breaks torch.compile.
unset PYTHONPATH
# Per-arm compile caches so eight concurrent compilations never share a directory.
export TORCHINDUCTOR_CACHE_DIR="/tmp/vimeml-v45-${arm}/torchinductor-cache"
export TRITON_CACHE_DIR="/tmp/vimeml-v45-${arm}/triton-cache"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
output_dir=$(grep '^output_dir' "$config" | head -1 | sed 's/.*= *"\(.*\)"/\1/')
# W&B never uses the machine's login: drop every inherited WANDB_* variable (key,
# entity, service socket), point config/cache/data/netrc into this arm's own
# directory, then read only the private credential file.
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
"$python_bin" -X utf8 -u scripts/training/prepare_v45_teachers.py --verify-only
resume=()
if [[ -f "$output_dir/last.pt" ]]; then
  resume=(--resume)
fi
exec "$python_bin" -X utf8 -u scripts/training/train_v45.py --config "$config" "${resume[@]}"
