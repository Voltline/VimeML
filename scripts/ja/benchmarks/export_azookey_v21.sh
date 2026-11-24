#!/usr/bin/env bash
# Run this copy from the extracted handoff package on the Mac.
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash export_azookey.sh /path/to/AzooKeyKanaKanjiConverter [results-directory]' >&2
  exit 2
fi
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
converter_dir="$(cd -- "$1" && pwd)"
results_dir="${2:-$package_dir/azookey-results}"
expected_converter='d59a28e4c7ca049aef04f29a91eae9677a7753f2'
actual_converter="$(git -C "$converter_dir" rev-parse HEAD)"
if [[ "$actual_converter" != "$expected_converter" ]]; then
  echo "Converter version differs: $actual_converter; expected $expected_converter. Use the existing benchmark checkout." >&2
  exit 2
fi
if [[ ! -x "$converter_dir/.build/release/CliTool" ]]; then
  echo 'Build CliTool in the converter checkout: swift build -c release --product CliTool -Xcxx -xobjective-c++' >&2
  exit 2
fi
mkdir -p -- "$results_dir"
results_dir="$(cd -- "$results_dir" && pwd)"
cd -- "$converter_dir"
for role in development blind; do
  result="$results_dir/$role"
  if [[ -e "$result/azookey-candidates.json" ]]; then
    if [[ -f "$result/completed.txt" ]] && \
      cmp -s "$package_dir/$role/ajimee-input.json" "$result/ajimee-input.json" && \
      [[ "$(cat "$result/converter-version.txt")" == "$actual_converter" ]]; then
      echo "Keeping completed export: $role"
      continue
    fi
    echo "Already has candidate output: $result. Keep it and choose a fresh results directory." >&2
    exit 2
  fi
  mkdir -p -- "$result"
  cp -- "$package_dir/$role/ajimee-input.json" "$result/ajimee-input.json"
  cp -- "$package_dir/$role/case-map.json" "$result/case-map.json"
  git rev-parse HEAD > "$result/converter-version.txt"
  git submodule status --recursive > "$result/dictionary-versions.txt"
  swift --version > "$result/swift-version.txt"
  printf '%s\n' 'n_best=20' 'typo_mode=off' 'Zenzai=disabled (no model)' 'stable=off' > "$result/export-flags.txt"
  echo "Exporting $role ..."
  .build/release/CliTool evaluate "$result/ajimee-input.json" \
    --config_n_best 20 --config_typo_mode off \
    --output "$result/azookey-candidates.json" 2>&1 | tee "$result/export.log"
  printf '%s\n' 'cli_exit=0' > "$result/completed.txt"
done
echo "Done. Return this complete directory: $results_dir"
