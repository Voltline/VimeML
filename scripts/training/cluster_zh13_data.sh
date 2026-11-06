#!/usr/bin/env bash
# Build the Vime 1.3 zh token stores on the cluster, once, before launch_zh13_all.sh.
# Needs the frozen export at outputs/zh-corpus/export-r1 (every file checked against
# SHA256SUMS) and the formal tokenizers copied to artifacts/tokenizers/zh-v13 (their
# SHA256 is checked against the tokenizer manifest by the encoder).
# Steps: verify SHA256SUMS -> measure pools -> encode 8K and 12K. Finished steps are
# skipped on a rerun; a half-written output directory must be removed by hand.
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
