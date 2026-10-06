"""Small local candidate ranking and phrase suggestion diagnostic."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.training.evaluate_ime import main

if __name__ == "__main__":
    main()
