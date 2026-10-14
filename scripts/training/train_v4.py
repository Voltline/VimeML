"""Explicit V4 training entry; importing it never starts training."""
import sys
import argparse
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.training.model_factory_v4 import register_v4
from vimeml.training.train import main as training_main


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train-v4-b.toml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    if config.get("architecture") != "tiny_gpt_v4":
        parser.error("This entry point accepts only tiny_gpt_v4 configurations")
    register_v4()
    # Reuse the established tracking dispatch, including its offline mode.
    forwarded = ["--config", str(args.config.resolve())]
    if args.resume:
        forwarded.append("--resume")
    if args.dry_run:
        forwarded.append("--dry-run")
    return training_main(forwarded)

if __name__ == "__main__":
    main()
