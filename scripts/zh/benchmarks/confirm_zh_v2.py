"""Run the preregistered test-split confirmation of vimeml-zh-v2-20m, -200m or -800m (see its plan, once per model).

usage:
  python -I -X utf8 scripts/zh/benchmarks/confirm_zh_v2.py open v2-20m --preflight-only   # development only
  python -I -X utf8 scripts/zh/benchmarks/confirm_zh_v2.py open v2-20m      # opens test: contexts, generation
  python -I -X utf8 scripts/zh/benchmarks/confirm_zh_v2.py finish v2-20m --production ROWS
  python -I -X utf8 scripts/zh/benchmarks/confirm_zh_v2.py open v2-200m && ... finish v2-200m
  python -I -X utf8 scripts/zh/benchmarks/confirm_zh_v2.py open v2-800m && ... finish v2-800m
After `open v2-20m`, the local Rime harness reads RUN/test-contexts.jsonl and the test records and writes the
production rows ({"id", "input", "v1-prod": [...]}) given to `finish`.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.benchmarks.zh_v2_confirm import MODELS, finish, open_test  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["open", "finish"])
    parser.add_argument("model", choices=sorted(MODELS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--production")
    args = parser.parse_args()
    if args.step == "open":
        out = open_test(ROOT, args.model, args.device, args.preflight_only)
        print(f"{'preflight passed, test not opened' if args.preflight_only else 'test generated'}: {out}")
    else:
        result = finish(ROOT, args.model, args.production, args.device)
        print(json.dumps(result["decision"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
