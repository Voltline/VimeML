"""Prepare the local 100-case native-speaker review, without model outputs."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.benchmarks.chinese_review import build_packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-cases", type=Path, default=ROOT / "artifacts/chinese-human-review-20261017/new-cases-private.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/chinese-human-review-20261017/dist")
    args = parser.parse_args()
    payload = build_packet(ROOT / "artifacts/chinese-ime-research-20261017/cases.jsonl", args.new_cases,
        ROOT / "templates/benchmarks/chinese-ime-human-review.html", args.output,
        ROOT / "artifacts/chinese-human-review-20261017")
    print(payload["dataset_id"], len(payload["cases"]))
    print(args.output / "index.html")


if __name__ == "__main__":
    main()
