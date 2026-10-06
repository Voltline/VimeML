"""Inspect local raw data; this command never modifies the inputs."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.tools.inspect_data import main


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
