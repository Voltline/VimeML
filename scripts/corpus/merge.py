"""Merge completed cleaning parts using the verified parallel exporter."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.data.merge_parts_fast import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/corpus-parallel.toml")
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--parts", nargs="+", type=Path)
    sources.add_argument("--parts-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--buckets", type=int, default=64, choices=(1, 4, 16, 64, 256))
    parser.add_argument("--cache-mb", type=int, default=128)
    args = parser.parse_args()
    parts = args.parts if args.parts is not None else sorted(p.parent for p in args.parts_dir.glob("*/manifest.json"))
    if not parts:
        parser.error("No completed parts found")
    run(args.config, parts, args.output, workers=args.workers, buckets=args.buckets,
        cache_mb=args.cache_mb, work=args.work)


if __name__ == "__main__":
    main()
