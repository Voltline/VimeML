#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
converter_dir="$(cd -- "${1:-$HOME/Sources/AzooKeyKanaKanjiConverter-ajimee}" && pwd)"
result_name="${2:-azookey-results}"
if [[ "$result_name" == */* || "$result_name" == '.' || "$result_name" == '..' ]]; then
  echo 'Results directory must be a simple name inside this package.' >&2
  exit 2
fi
if [[ "$(git -C "$converter_dir" rev-parse HEAD)" != 'd59a28e4c7ca049aef04f29a91eae9677a7753f2' ]]; then
  echo 'Use the frozen converter checkout recorded in README_CN.md.' >&2
  exit 2
fi
dictionary_versions="$(git -C "$converter_dir" submodule status --recursive)"
for expected in '4d418525b090cf49c219819d05a7e3cc2a4346eb' '67b822603391b01238d7b80b8b61b63f966cf357'; do
  if [[ "$dictionary_versions" != *" $expected "* ]]; then
    echo "Dictionary snapshot differs or is uninitialized: $expected" >&2
    exit 2
  fi
done
if [[ ! -x "$converter_dir/.build/release/CliTool" ]]; then
  (cd -- "$converter_dir" && swift build -c release --product CliTool -Xcxx -xobjective-c++)
fi
bash "$package_dir/export_azookey.sh" "$converter_dir" "$package_dir/$result_name"
archive="$package_dir/ime-standard-ja-v1-azookey-results.zip"
if [[ -e "$archive" ]]; then
  echo "Keep existing archive: $archive. Successful exports remain in $result_name." >&2
  exit 2
fi
# Archive name is stable even when recovering into a fresh local directory.
if [[ "$result_name" == 'azookey-results' ]]; then
  (cd -- "$package_dir" && zip -rq "$archive" azookey-results)
else
  python3 - "$package_dir/$result_name" "$archive" <<'PY'
from pathlib import Path
import sys, zipfile
root, target = Path(sys.argv[1]), Path(sys.argv[2])
with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as out:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out.write(path, "azookey-results/" + path.relative_to(root).as_posix())
PY
fi
echo "Return: $archive"
