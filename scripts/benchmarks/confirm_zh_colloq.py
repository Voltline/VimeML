"""Run the preregistered test confirmation of colloq-c40 against vimeml-zh-v2-20m (kd-cont), once
(docs/zh-CN/benchmarks/confirmation-zh-colloq-plan.md).

usage:
  python -I -X utf8 scripts/benchmarks/confirm_zh_colloq.py --preflight-only   # development only
  python -I -X utf8 scripts/benchmarks/confirm_zh_colloq.py                    # opens the test splits
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.benchmarks.zh_colloq_confirm import run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = run(ROOT, args.device, args.preflight_only)
    print(json.dumps(result.get("decision", result), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
