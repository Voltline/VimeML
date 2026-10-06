"""Command entry point; implementation lives in src/vimeml."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.data.preprocess_part import main

if __name__ == "__main__":
    main()
