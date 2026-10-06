"""Evaluate a frozen AJIMEE candidate export using the local Tiny Japanese LM."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.benchmarks.evaluate_ajimee import main

if __name__ == "__main__":
    main()
