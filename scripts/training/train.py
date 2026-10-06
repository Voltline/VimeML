"""Manual training entry point, safe for Windows worker spawn."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.training.train import main

if __name__ == "__main__":
    main()
