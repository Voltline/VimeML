"""Validate, freeze and evaluate a returned native-speaker Chinese review packet."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.benchmarks.chinese_review_eval import prepare, capture_and_score, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "report", "all"))
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts/chinese-human-eval-20261017-184402")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/chinese-human-eval-20261017-184402")
    args = parser.parse_args()
    if args.action in {"prepare", "all"}:
        if not args.annotations:
            parser.error("--annotations is required when freezing the returned packet")
        prepare(ROOT, args.annotations, args.artifacts, args.output)
    if args.action in {"run", "all"}:
        capture_and_score(ROOT, args.artifacts, args.output)
    if args.action in {"report", "all"}:
        report(ROOT, args.artifacts, args.output)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
