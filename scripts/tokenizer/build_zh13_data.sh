#!/usr/bin/env bash
# Build the vimeml-tiny-zh-v1 training data once, where the frozen export lives (docs/zh-CN/training-zh-v1-plan.md):
# verify export-r1 against SHA256SUMS -> measure pools -> encode 8K and 12K -> eight arm mixtures,
# then write configs/data-zh13.sha256, the list the cluster checks after downloading.
# Needs outputs/zh-corpus/export-r1 and the formal tokenizers in artifacts/tokenizers/zh-v13.
# Finished steps are skipped on a rerun; a half-written output directory must be removed by hand.
set -euo pipefail
cd "$(dirname "$0")/../.."
unset PYTHONPATH
python_bin="${VIMEML_PYTHON:-python3}"
export_dir="outputs/zh-corpus/export-r1"
( cd "$export_dir" && sha256sum --quiet -c SHA256SUMS )
echo "export-r1 matches SHA256SUMS"
if [[ ! -f artifacts/training-data/zh13-pools/pools.json ]]; then
  "$python_bin" -I -X utf8 -u scripts/tokenizer/encode_zh13.py --no-verify-inputs measure
fi
for vocab in 8192 12288; do
  tag=$([[ $vocab == 8192 ]] && echo 8k || echo 12k)
  if [[ -f "artifacts/training-data/zh13-chat-${tag}-c128/manifest.json" ]]; then
    echo "zh13 ${tag} stores already built"; continue
  fi
  "$python_bin" -I -X utf8 -u scripts/tokenizer/encode_zh13.py --no-verify-inputs encode --vocab "$vocab"
done
arms=(12k-fresh-s1 12k-fresh-s2 12k-fresh-s3 12k-ep5 12k-ep20 8k-fresh 8k-ep5 8k-ep20)
for arm in "${arms[@]}"; do
  "$python_bin" -X utf8 -u scripts/corpus/prepare_zh13.py --arm "$arm" | tail -1
done
# Everything the cluster downloads, relative to the repository root. run.json only holds wall time.
find artifacts/tokenizers/zh-v13 artifacts/training-data/zh13-* artifacts/token-data/zh13-* -type f \
    ! -name run.json ! -name '*.tmp' | LC_ALL=C sort | xargs sha256sum | sed 's/ \*/  /' > configs/data-zh13.sha256
echo "wrote configs/data-zh13.sha256 ($(wc -l < configs/data-zh13.sha256) files)"
