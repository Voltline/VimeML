"""Check the existing corpus without requiring cleaned-up worker directories."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.tools.check_corpus import main

if __name__ == "__main__":
    main()
