"""Run the preregistered blind confirmation of vimeml-tiny-zh-v1 (see its plan, once only).

usage:
  python -I -X utf8 scripts/benchmarks/confirm_zh_v1.py --preflight-only   # development only, blind stays closed
  python -I -X utf8 scripts/benchmarks/confirm_zh_v1.py                    # needs the plan and code committed
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.benchmarks.zh_v1_confirm import run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = run(ROOT, args.device, args.preflight_only)
    if result:
        print(json.dumps(result["decision"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
