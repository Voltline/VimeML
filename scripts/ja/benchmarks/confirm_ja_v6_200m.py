"""Run the preregistered test confirmation of vimeml-ja-v6-200m against the V4.2 deployment, with
vimeml-ja-v6-20m as the guard, once (docs/zh-CN/ja/benchmarks/confirmation-ja-v6-200m-plan.md).
One process, one GPU. Needs the finished 20M confirmation (outputs/benchmarks/ja-gen-confirmation/run-01).

usage:
  python -X utf8 scripts/ja/benchmarks/confirm_ja_v6_200m.py --preflight-only   # development only
  python -X utf8 scripts/ja/benchmarks/confirm_ja_v6_200m.py                    # opens the test splits
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.benchmarks.ja_v6_200m_confirm import run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = run(ROOT, args.device, args.preflight_only)
    print(json.dumps(result.get("decision", result), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
