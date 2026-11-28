"""Run the preregistered test confirmation of the ja-gen 20M candidate colloq-c40 against the V4.2 deployment,
once (docs/zh-CN/ja/benchmarks/confirmation-ja-gen-plan.md). One process, one GPU.

usage:
  python -X utf8 scripts/ja/benchmarks/confirm_ja_gen.py --preflight-only   # development only
  python -X utf8 scripts/ja/benchmarks/confirm_ja_gen.py                    # opens the test splits
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.benchmarks.ja_gen_confirm import run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = run(ROOT, args.device, args.preflight_only)
    print(json.dumps(result.get("decision", result), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
