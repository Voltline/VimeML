"""Train the vimeml-tiny-zh-v1 8K and 12K tokenizers on the frozen export.

Formal runs read configs/tokenizer-zh13.toml, require the export to be frozen
(manifest.json + SHA256SUMS) and hash every input against SHA256SUMS. --unfrozen is
for local smoke tests only and must write somewhere other than the formal directory.

usage:
  python -I -X utf8 scripts/tokenizer/train_zh13.py
  python -I -X utf8 scripts/tokenizer/train_zh13.py --unfrozen --output OUT [--export DIR]
"""
import argparse
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.tokenizer.zh13 import run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tokenizer-zh13.toml")
    parser.add_argument("--export", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--unfrozen", action="store_true", help="smoke test on an unfrozen export")
    parser.add_argument("--no-verify-inputs", action="store_true", help="check names in SHA256SUMS but skip hashing")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text("utf-8"))
    export = args.export or ROOT / config["export_dir"]
    output = args.output or ROOT / config["output_dir"]
    if args.unfrozen and output.resolve() == (ROOT / config["output_dir"]).resolve():
        parser.error("--unfrozen must not write the formal tokenizer directory")
    run(config, export, output, frozen=not args.unfrozen, verify_inputs=not args.no_verify_inputs)


if __name__ == "__main__":
    main()
