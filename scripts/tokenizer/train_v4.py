"""Train and verify the isolated V4-a tokenizer experiment."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.tokenizer.v4 import main

if __name__ == "__main__":
    main()
