"""Command entry point for reviewing with independently limited SJTU accounts."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.review.multi_key import main

if __name__ == "__main__":
    main()
