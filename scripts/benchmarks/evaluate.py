"""Score a frozen, verified synthetic benchmark with a local checkpoint."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.benchmarks.evaluate import main

if __name__ == "__main__":
    main()
