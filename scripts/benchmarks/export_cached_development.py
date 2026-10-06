"""Export successful generation caches for review, even if other batches failed."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.benchmarks.development import export_cached_drafts
from vimeml.review.multi_key import output_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, default=ROOT / "artifacts/benchmarks/ajimee-jwtd-v2-v1/evaluation_items.json")
    args = parser.parse_args()
    try:
        with output_lock(args.output):
            stats = export_cached_drafts(args.source, args.output, args.exclude)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
